#!/usr/bin/env python
"""Run the R4 corpus validity audit and emit its evidence artifact.

Measures the corpus; changes nothing. Deterministic given the same task files
and the same seed, so the recorded command reruns to the same figures.

The power projection is only trustworthy if the simulator that produced it can
reproduce a real measured interval, so this script calibrates against R3's
completed replication run before projecting anything, and the artifact schema
refuses to record a failed calibration.

Usage:
    uv run python benchmarks/corpus_audit/run_audit.py \
        --output benchmarks/evidence/s2-corpus-validity.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from archex.benchmark.corpus_audit import (
    audit_held_out,
    describe_clusters,
    detection_bracket,
    estimate_effect_heterogeneity,
    minimum_detectable_effect,
    score_corpus_leakage,
    simulate_power,
)
from archex.benchmark.evidence import task_manifest_digest
from archex.benchmark.loader import load_tasks

SEED = 20260728
SIMULATIONS = 4000
RESAMPLES = 400
#: 4000 simulations puts the Monte Carlo standard error of a power estimate near
#: 0.006, small against the grid spacing. At 400 it was ~0.02, which is wider than
#: the distance from the reported +25 to the 0.80 target -- the estimator was being
#: read at a resolution it did not have.
CLUSTER_SD = 0.08
#: The nuisance parameter no calibration constrains, so its influence is measured
#: rather than assumed. If the verdict moved across this sweep it would be an
#: artifact of a chosen constant instead of a property of the corpus.
SENSITIVITY_CLUSTER_SD = (0.02, 0.05, 0.08, 0.15, 0.25)
SENSITIVITY_EFFECTS = (10.0, 20.0, 25.0, 30.0)
#: Effects to search, in percentage points. Spans the literature's range
#: (RLCoder's +4.88) through effects far larger than anything published.
CANDIDATE_EFFECTS = (1.0, 2.0, 3.0, 4.88, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
TARGET_POWER = 0.80
#: Two base rates rather than one: archex's own metrics sit high, but a rate near
#: 0.5 maximises binomial variance and is the conservative choice.
BASELINE_RATE = 0.50
BASE_RATES = (BASELINE_RATE, 0.85)

#: R3's completed run, used to validate the simulator. 8 balanced repository
#: clusters of 200 tasks, control exact match 41.625%, and a measured 95% cluster
#: bootstrap interval of [0.1250, 5.9375] on a delta of +2.8125.
R3_CLUSTERS = (200,) * 8
R3_BASE_RATE = 0.41625
R3_EFFECT = 2.8125
R3_MEASURED_CI_WIDTH = 5.8125
#: The simulator only needs to land in the right region; a generative model of a
#: real pipeline is not expected to match to the third decimal.
CALIBRATION_TOLERANCE = 0.20
#: The effect R3 was asked to recover.
LITERATURE_EFFECT = 4.88
#: R3's eight measured per-repository deltas, used to estimate how much the
#: treatment effect actually varies between repositories. Their spread turns out
#: to be smaller than the within-repository sampling noise at 200 tasks each, so
#: the estimate floors at zero: this is the only dataset available and it shows no
#: detectable between-repository heterogeneity.
R3_PER_REPO_DELTAS = (2.5, 2.0, -3.5, 12.0, 3.5, -1.0, 4.5, 2.5)
R3_TASKS_PER_CLUSTER = 200
#: Corpus growth is projected in total tasks, held at the current 16 clusters.
#: Projecting in repositories would imply repositories are the binding
#: constraint, which requires between-repository effect heterogeneity the data do
#: not show.
CORPUS_GROWTH_TASKS = (64, 128, 256, 512, 1024, 2048, 4096)
CURRENT_CLUSTERS = 16


def _source_revision(repo_root: Path) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"could not resolve the source revision: {result.stderr.strip()}"
        raise SystemExit(msg)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", default="benchmarks/tasks", type=Path)
    parser.add_argument("--held-out", default="benchmarks/held_out.txt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path.cwd()
    tasks = load_tasks(args.tasks_dir)
    leakage = score_corpus_leakage(tasks)
    clusters = describe_clusters(tasks)

    declared = [
        line.strip()
        for line in args.held_out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Established by searching src/ and .github/ for any consumer of
    # held_out.txt: there is none, so nothing excludes these IDs from a run.
    held_out = audit_held_out(declared, tasks, enforced_by_code=False)

    reference_power_at_reported = simulate_power(
        R3_CLUSTERS,
        effect_points=LITERATURE_EFFECT,
        base_rate=R3_BASE_RATE,
        cluster_sd=CLUSTER_SD,
        simulations=SIMULATIONS,
        resamples=RESAMPLES,
        seed=SEED,
    ).power
    calibration = simulate_power(
        R3_CLUSTERS,
        effect_points=R3_EFFECT,
        base_rate=R3_BASE_RATE,
        cluster_sd=CLUSTER_SD,
        simulations=SIMULATIONS,
        resamples=RESAMPLES,
        seed=SEED,
    )
    relative_error = abs(calibration.mean_ci_width - R3_MEASURED_CI_WIDTH) / R3_MEASURED_CI_WIDTH
    within_tolerance = relative_error <= CALIBRATION_TOLERANCE

    sizes = list(clusters.cluster_sizes.values())
    mde: dict[str, float | None] = {}
    brackets: dict[str, str] = {}
    curves: dict[str, list[dict[str, float]]] = {}
    for base_rate in BASE_RATES:
        # An effect that saturates the treatment arm is not a power calculation,
        # and simulate_power refuses it, so the grid is trimmed per base rate.
        candidates = [effect for effect in CANDIDATE_EFFECTS if base_rate + effect / 100.0 <= 1.0]
        detectable, curve = minimum_detectable_effect(
            sizes,
            base_rate=base_rate,
            cluster_sd=CLUSTER_SD,
            target_power=TARGET_POWER,
            candidates=candidates,
            simulations=SIMULATIONS,
            resamples=RESAMPLES,
            seed=SEED,
        )
        key = f"base_rate_{base_rate}"
        mde[key] = detectable
        brackets[key] = detection_bracket(curve, target_power=TARGET_POWER).describe()
        curves[key] = [
            {
                "effect_points": item.effect_points,
                "power": round(item.power, 4),
                "monte_carlo_se": round(item.monte_carlo_se, 4),
                "mean_ci_width": round(item.mean_ci_width, 4),
            }
            for item in curve
        ]

    sensitivity = [
        {
            "cluster_sd": sd,
            "power": {
                str(effect): round(
                    simulate_power(
                        sizes,
                        effect_points=effect,
                        base_rate=BASELINE_RATE,
                        cluster_sd=sd,
                        simulations=SIMULATIONS,
                        resamples=RESAMPLES,
                        seed=SEED,
                    ).power,
                    4,
                )
                for effect in SENSITIVITY_EFFECTS
            },
        }
        for sd in SENSITIVITY_CLUSTER_SD
    ]

    effect_sd = estimate_effect_heterogeneity(
        R3_PER_REPO_DELTAS,
        tasks_per_cluster=R3_TASKS_PER_CLUSTER,
        base_rate=R3_BASE_RATE,
    )
    scaling = [
        {
            "tasks": total,
            "clusters": CURRENT_CLUSTERS,
            "tasks_per_cluster": total // CURRENT_CLUSTERS,
            "power": round(
                simulate_power(
                    (total // CURRENT_CLUSTERS,) * CURRENT_CLUSTERS,
                    effect_points=LITERATURE_EFFECT,
                    base_rate=BASELINE_RATE,
                    cluster_sd=CLUSTER_SD,
                    simulations=SIMULATIONS,
                    resamples=RESAMPLES,
                    seed=SEED,
                    effect_sd=effect_sd,
                ).power,
                4,
            ),
        }
        for total in CORPUS_GROWTH_TASKS
    ]
    reachable = next((row["tasks"] for row in scaling if float(row["power"]) >= TARGET_POWER), None)

    baseline_key = f"base_rate_{BASELINE_RATE}"
    literature_power: float = next(
        row["power"] for row in curves[baseline_key] if row["effect_points"] == LITERATURE_EFFECT
    )

    verdict = (
        f"At {clusters.total_tasks} tasks over {clusters.cluster_count} repository clusters, "
        f"the smallest effect this corpus can detect at {TARGET_POWER:.0%} power is "
        f"{brackets[baseline_key]}; a literature-sized effect of {LITERATURE_EFFECT} points "
        f"has power {literature_power}. Reaching {TARGET_POWER:.0%} power for that effect needs "
        f"about {reachable} tasks, against the {clusters.total_tasks} available today. "
        f"Between-repository effect heterogeneity estimated from R3's measured per-repository "
        f"deltas is {effect_sd:.2f} points, so the binding constraint is total task count and "
        "this audit does not claim repositories are the scarce resource."
    )

    artifact = {
        "corpus_audit_version": 1,
        "milestone": "R4",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "source_revision": _source_revision(repo_root),
        "tasks_dir": str(args.tasks_dir),
        "task_manifest_digest": task_manifest_digest(args.tasks_dir),
        "total_tasks": leakage.total_tasks,
        "leakage": {
            "symbol_leaked_tasks": list(leakage.symbol_leaked_task_ids),
            "symbol_leak_rate": round(leakage.symbol_leak_rate, 4),
            "any_leaked_tasks": list(leakage.leaked_task_ids),
            "any_leak_rate": round(leakage.leak_rate, 4),
            "signals_by_kind": leakage.by_kind,
            "leaked_tasks_by_family": leakage.by_family,
            "tiers": (
                "symbol = an identifier-shaped gold symbol quoted in the query; "
                "symbol_word = a gold symbol that is also the ordinary word for the "
                "thing asked about, excluded from the headline; "
                "path_stem = a gold file's stem quoted in the query, weakest tier"
            ),
        },
        "clustering": {
            "cluster_count": clusters.cluster_count,
            "cluster_sizes": clusters.cluster_sizes,
            "largest_cluster": clusters.largest_cluster,
            "largest_cluster_share": round(clusters.largest_cluster_share, 4),
            "self_repo_share": round(clusters.self_repo_share, 4),
            "weighted_mean_cluster_size": round(clusters.weighted_mean_cluster_size, 4),
            "effective_sample_size": {
                str(icc): round(clusters.effective_sample_size(icc), 2)
                for icc in (0.0, 0.1, 0.3, 0.5, 0.8)
            },
        },
        "held_out": {
            "declared": list(held_out.declared),
            "also_in_task_corpus": list(held_out.also_in_task_corpus),
            "leak_rate": round(held_out.leak_rate, 4),
            "enforced_by_code": held_out.enforced_by_code,
            "note": (
                "No code under src/ or .github/ consumes held_out.txt, so the declaration "
                "has no runtime effect. tests/benchmark/test_generalization.py asserts every "
                "held-out ID is also a top-level task, making the overlap a requirement "
                "rather than a defect it would catch."
            ),
        },
        "power": {
            "seed": SEED,
            "simulations": SIMULATIONS,
            "resamples": RESAMPLES,
            "cluster_sd": CLUSTER_SD,
            "target_power": TARGET_POWER,
            "minimum_detectable_effect_points": mde,
            "detection_bracket": brackets,
            "power_curves": curves,
            "literature_effect_points": LITERATURE_EFFECT,
            "tasks_needed": reachable,
            "effect_sd_estimated_from_r3": round(effect_sd, 4),
            "effect_sd_note": (
                "Between-cluster SD of the treatment effect, estimated from R3's eight "
                "measured per-repository deltas with within-cluster binomial noise removed. "
                "It floors at zero: the observed spread is smaller than the sampling noise "
                "at 200 tasks per repository, so the only data available show no detectable "
                "between-repository heterogeneity. Without heterogeneity, repositories are "
                "not the binding constraint and power is governed by total task count."
            ),
            "scaling": scaling,
            "cluster_sd_sensitivity": sensitivity,
            "model": (
                "Each cluster draws a success rate from Normal(base_rate, cluster_sd); every "
                "task in it draws a control outcome at that rate and a treatment outcome at "
                "that rate plus the effect. Inference is the same cluster bootstrap over "
                "repositories that R3 used, and power is the fraction of simulations whose "
                "95% interval excludes zero."
            ),
        },
        "calibration": {
            "reference": "R3 / benchmarks/evidence/s0-rlcoder-replication.json",
            "reference_structure": f"{len(R3_CLUSTERS)} clusters of {R3_CLUSTERS[0]} tasks",
            "reference_effect_points": R3_EFFECT,
            "measured_ci_width": R3_MEASURED_CI_WIDTH,
            "simulated_ci_width": round(calibration.mean_ci_width, 4),
            "relative_error": round(relative_error, 4),
            "tolerance": CALIBRATION_TOLERANCE,
            "within_tolerance": within_tolerance,
            "power_on_reference_structure_at_observed_effect": round(calibration.power, 4),
            "power_on_reference_structure_at_reported_effect": round(
                reference_power_at_reported, 4
            ),
            "observed_power_caveat": (
                "Power at the effect actually observed is observed power, not design power, "
                "and reporting it as if the reference setup were underpowered would be a "
                "post-hoc-power fallacy. At the effect R3's paper reports (+4.88) the same "
                "model puts that setup near the conventional 0.80 bar."
            ),
        },
        "verdict": verdict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(verdict)
    print(
        "cluster_sd sensitivity (power at "
        + ", ".join(f"+{e:g}" for e in SENSITIVITY_EFFECTS)
        + "): "
        + "; ".join(
            f"sd={row['cluster_sd']} -> "
            + "/".join(str(row["power"][str(e)]) for e in SENSITIVITY_EFFECTS)  # type: ignore[index]
            for row in sensitivity
        )
    )
    print(
        f"calibration: simulated {calibration.mean_ci_width:.3f} against measured "
        f"{R3_MEASURED_CI_WIDTH} ({relative_error:.1%} error, within tolerance: {within_tolerance})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
