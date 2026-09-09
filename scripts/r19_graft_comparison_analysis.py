"""Recompute R19's frozen Graft-vs-archex analysis from the checked-in artifacts.

Every number quoted in `benchmarks/headtohead/GRAFT_COMPARISON_R19.md` comes from
this script, so any reader can re-derive it:

```bash
uv run python scripts/r19_graft_comparison_analysis.py \
    --output benchmarks/evidence/r19-graft-graph-memory-comparison.json
```

The analysis is exactly the one frozen in
`benchmarks/preregistrations/R19-graft-graph-memory-comparison.md` before the
first Graft cell existed: primary metric mean required-file recall, treatment
`graft_query_warm` minus control `archex_query`, a 10 000-resample bootstrap over
the 15 source repositories with seed 20260909, evaluated against MWG +0.05,
NIM -0.05 and EQM ±0.03; plus the pre-declared depth-tier-only secondary. No cell
is excluded, and every other quantity is exploratory.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from archex.benchmark.headtohead import (
    load_headtohead_manifest,
    load_headtohead_results,
    reports_with_graph_memory_lanes,
)
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy

PRIMARY_TREATMENT = "graft_query_warm"
PRIMARY_CONTROL = "archex"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260909
MWG = 0.05
NIM = -0.05
EQM = 0.03


def _lane_results(reports: list[BenchmarkReport]) -> dict[str, dict[str, BenchmarkResult]]:
    """Map lane label -> task_id -> result."""
    lanes: dict[str, dict[str, BenchmarkResult]] = {}
    for report in reports:
        for result in report.results:
            if result.strategy is Strategy.ARCHEX_QUERY:
                label = "archex"
            elif result.strategy is Strategy.EXTERNAL_MCP:
                label = result.strategy_label or result.provenance.get("external_tool", "external")
            else:
                label = result.strategy.value
            lanes.setdefault(label, {})[report.task_id] = result
    return lanes


def _cluster_bootstrap(
    deltas: dict[str, float],
    clusters: dict[str, list[str]],
) -> tuple[float, float]:
    """Percentile 95% CI for the mean delta, resampling clusters with replacement."""
    rng = random.Random(BOOTSTRAP_SEED)
    cluster_ids = sorted(clusters)
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled: list[float] = []
        for _ in cluster_ids:
            cluster = rng.choice(cluster_ids)
            sampled.extend(deltas[task_id] for task_id in clusters[cluster])
        means.append(statistics.fmean(sampled))
    means.sort()
    low = means[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))]
    high = means[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))]
    return low, high


def _verdict(mean: float, low: float, high: float) -> dict[str, object]:
    return {
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "minimum_worthwhile_gain_met": low > MWG,
        "non_inferior": low > NIM,
        "practically_equivalent": low > -EQM and high < EQM,
    }


def _comparison(
    task_ids: list[str],
    treatment: dict[str, BenchmarkResult],
    control: dict[str, BenchmarkResult],
    repo_by_task: dict[str, str],
) -> dict[str, object]:
    deltas = {
        task_id: treatment[task_id].required_file_recall - control[task_id].required_file_recall
        for task_id in task_ids
    }
    clusters: dict[str, list[str]] = {}
    for task_id in task_ids:
        clusters.setdefault(repo_by_task[task_id], []).append(task_id)
    mean_delta = statistics.fmean(deltas.values())
    low, high = _cluster_bootstrap(deltas, clusters)
    return {
        "tasks": len(task_ids),
        "clusters": len(clusters),
        "treatment_mean": statistics.fmean(
            treatment[task_id].required_file_recall for task_id in task_ids
        ),
        "control_mean": statistics.fmean(
            control[task_id].required_file_recall for task_id in task_ids
        ),
        "per_task_difference": deltas,
        **_verdict(mean_delta, low, high),
    }


def _exploratory(lane: dict[str, BenchmarkResult], task_ids: list[str]) -> dict[str, object]:
    results = [lane[task_id] for task_id in task_ids]
    warm = sorted(result.warm_latency_ms or 0.0 for result in results)
    return {
        "recall": statistics.fmean(result.recall for result in results),
        "precision": statistics.fmean(result.precision for result in results),
        "f1_score": statistics.fmean(result.f1_score for result in results),
        "mrr": statistics.fmean(result.mrr for result in results),
        "ndcg": statistics.fmean(result.ndcg for result in results),
        "map_score": statistics.fmean(result.map_score for result in results),
        "required_file_recall": statistics.fmean(result.required_file_recall for result in results),
        "all_required_files_present_rate": statistics.fmean(
            1.0 if result.all_required_files_present else 0.0 for result in results
        ),
        "token_efficiency": statistics.fmean(result.token_efficiency for result in results),
        "token_efficiency_with_completion": statistics.fmean(
            result.token_efficiency_with_completion for result in results
        ),
        "tokens_output_mean": statistics.fmean(result.tokens_output for result in results),
        "cold_start_ms_mean": statistics.fmean(result.cold_start_ms or 0.0 for result in results),
        "warm_latency_p50_ms": warm[len(warm) // 2],
        "warm_latency_p95_ms": warm[min(len(warm) - 1, int(0.95 * (len(warm) - 1)))],
        "freshness_latency_ms_mean": statistics.fmean(
            result.freshness_latency_ms or 0.0 for result in results
        ),
        "freshness_correct_rate": statistics.fmean(
            1.0 if result.freshness_correct else 0.0 for result in results
        ),
        "returned_units_total": sum(
            int(result.provenance.get("graft_returned_units", "0")) for result in results
        ),
        "symbol_hits_total": sum(
            int(result.provenance.get("graft_symbol_hits", "0")) for result in results
        ),
        "whole_file_hits_total": sum(
            int(result.provenance.get("graft_whole_file_hits", "0")) for result in results
        ),
        "returned_source_units_total": sum(
            int(result.provenance.get("graft_returned_source_units", "0")) for result in results
        ),
        "query_modes": sorted(
            {result.provenance.get("graft_query_mode", "") for result in results}
        ),
        "extraction_tiers": {
            tier: sum(
                1 for result in results if result.provenance.get("graft_extraction_tier") == tier
            )
            for tier in ("depth", "breadth")
        },
        "statuses": {
            status: sum(1 for result in results if result.provenance.get("graft_status") == status)
            for status in ("ok", "failed")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmarks/headtohead/results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_headtohead_manifest(args.input / "manifest.yaml")
    reports = reports_with_graph_memory_lanes(manifest, load_headtohead_results(args.input))
    repo_by_task = {report.task_id: report.repo for report in reports}
    task_ids = [report.task_id for report in reports]
    lanes = _lane_results(reports)

    for label in (PRIMARY_TREATMENT, PRIMARY_CONTROL, "graft_build_plus_query"):
        missing = [task_id for task_id in task_ids if task_id not in lanes.get(label, {})]
        if missing:
            message = f"lane {label!r} is missing cells: {', '.join(missing)}"
            raise SystemExit(message)

    treatment = lanes[PRIMARY_TREATMENT]
    control = lanes[PRIMARY_CONTROL]
    depth_task_ids = [
        task_id
        for task_id in task_ids
        if treatment[task_id].provenance.get("graft_extraction_tier") == "depth"
    ]

    payload: dict[str, object] = {
        "spike_id": "R19",
        "preregistration": "benchmarks/preregistrations/R19-graft-graph-memory-comparison.md",
        "evidence_class": "original",
        "treatment_lane": PRIMARY_TREATMENT,
        "control_lane": PRIMARY_CONTROL,
        "primary_metric": "mean required-file recall over the 19 tasks",
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "repository",
        },
        "margins": {"minimum_worthwhile_gain": MWG, "non_inferiority": NIM, "equivalence": EQM},
        "primary": _comparison(task_ids, treatment, control, repo_by_task),
        "secondary_depth_tier_only": _comparison(depth_task_ids, treatment, control, repo_by_task),
        "exploratory": {
            "graft_query_warm": _exploratory(treatment, task_ids),
            "graft_build_plus_query": _exploratory(lanes["graft_build_plus_query"], task_ids),
        },
        "coverage": {
            "graft_query_warm_cells": len(treatment),
            "graft_build_plus_query_cells": len(lanes["graft_build_plus_query"]),
            "recorded_failures": sum(
                1
                for label in ("graft_query_warm", "graft_build_plus_query")
                for result in lanes[label].values()
                if result.provenance.get("graft_status") == "failed"
            ),
        },
    }

    document = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    else:
        print(document, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
