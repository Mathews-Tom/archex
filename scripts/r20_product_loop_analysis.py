"""Derive the R20 paired product-loop analysis from raw per-cell artifacts.

The analysis is exactly the one frozen in
`benchmarks/preregistrations/R20-product-loop-agent-baseline.md` before the first
cell existed: mean required-file completeness of the agent's final answer,
treatment `graft_product_loop` minus control `archex_product_loop`, averaged over
the three repetitions inside each task and then over the 19 tasks, with a
10 000-resample cluster bootstrap over the 15 source repositories, seed
`20260909`, read against MWG `+0.05`, NIM `-0.05`, and EQM `±0.03`.

The margins are reported so the interval can be placed against a fixed scale.
They are **not** used to declare a verdict: R20 is registered as a descriptive
baseline, and no cross-tool superiority, non-inferiority, or equivalence claim is
published from it at any observed value.

Pre-declared secondaries, all labelled exploratory: the same comparison over
product-using cells only; the efficiency family over all cells and over
both-complete comparison units; per-task repetition spread; and the answer-flag
and product-use rates per arm.

Usage:

```bash
uv run python scripts/r20_product_loop_analysis.py \
    --input benchmarks/product_loop/results \
    --output benchmarks/evidence/r20-product-loop-agent-baseline.json
```

Output is deterministic: sorted keys, two-space indent, trailing newline, so the
published artifact regenerates byte-for-byte from the retained cells.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from archex.benchmark.product_loop import (
    AGENT_MODEL,
    AGENT_NAME,
    AGENT_VERSION,
    BILLING_MODE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COST_CEILING_USD,
    PREREGISTRATION_PATH,
    REPETITIONS,
    ProductLoopAnswerFlag,
    ProductLoopArm,
    ProductLoopCellStatus,
    ProductLoopError,
    load_product_loop_artifact,
    validate_product_loop_directory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from archex.benchmark.product_loop import ProductLoopCellArtifact

CONTROL = ProductLoopArm.ARCHEX
TREATMENT = ProductLoopArm.GRAFT
MWG = 0.05
NIM = -0.05
EQM = 0.03


def _load_cells(directory: Path) -> list[ProductLoopCellArtifact]:
    cells: list[ProductLoopCellArtifact] = []
    for arm in ProductLoopArm:
        for path in sorted((directory / arm.value).glob("*.json")):
            cells.append(load_product_loop_artifact(path))
    return cells


def _task_means(
    cells: Sequence[ProductLoopCellArtifact], *, product_using_only: bool
) -> dict[tuple[ProductLoopArm, str], float]:
    """Mean completeness per task and arm, averaged across repetitions.

    Every cell counts, including recorded failures, which enter as `0.0`. The
    pre-registration forbids dropping any of them.
    """
    buckets: dict[tuple[ProductLoopArm, str], list[float]] = defaultdict(list)
    for cell in cells:
        if product_using_only and cell.no_product_use:
            continue
        buckets[(cell.arm, cell.task_id)].append(cell.completeness)
    return {key: statistics.fmean(values) for key, values in buckets.items() if values}


def _paired_differences(
    means: dict[tuple[ProductLoopArm, str], float], task_repo: dict[str, str]
) -> dict[str, list[float]]:
    """Treatment-minus-control differences, grouped by source repository."""
    by_repo: dict[str, list[float]] = defaultdict(list)
    for task_id, repo in sorted(task_repo.items()):
        control = means.get((CONTROL, task_id))
        treatment = means.get((TREATMENT, task_id))
        if control is None or treatment is None:
            continue
        by_repo[repo].append(treatment - control)
    return dict(by_repo)


def _cluster_bootstrap(by_repo: dict[str, list[float]]) -> tuple[float, float, float]:
    """Mean difference and a 95% cluster-bootstrap interval over repositories.

    Repositories are the resampling unit, so both tasks of a two-task repository
    always move together; repetitions were already averaged inside their task and
    are not additional independent evidence.
    """
    repos = sorted(by_repo)
    flat = [value for repo in repos for value in by_repo[repo]]
    if not flat:
        return 0.0, 0.0, 0.0
    observed = statistics.fmean(flat)

    rng = random.Random(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample: list[float] = []
        for _ in repos:
            sample.extend(by_repo[repos[rng.randrange(len(repos))]])
        draws.append(statistics.fmean(sample))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return observed, low, high


def _efficiency(cells: Sequence[ProductLoopCellArtifact]) -> dict[str, float]:
    if not cells:
        return {}
    return {
        "mean_modelled_cost_usd": statistics.fmean(c.modelled_cost_usd for c in cells),
        "mean_setup_seconds": statistics.fmean(c.setup_seconds for c in cells),
        "mean_tool_calls": statistics.fmean(float(c.tool_calls) for c in cells),
        "mean_total_tokens": statistics.fmean(
            float(c.input_tokens + c.output_tokens + c.cache_read_tokens + c.cache_creation_tokens)
            for c in cells
        ),
        "mean_wall_seconds": statistics.fmean(c.wall_seconds for c in cells),
    }


def _both_complete_units(cells: Sequence[ProductLoopCellArtifact]) -> set[tuple[str, int]]:
    """`(task, repetition index)` units whose control and treatment both scored 1.0.

    Because repetitions are unseeded this is an index alignment, not a matched
    pair. It exists because an arm that answers less completely can look cheaper.
    """
    complete: dict[ProductLoopArm, set[tuple[str, int]]] = {arm: set() for arm in ProductLoopArm}
    for cell in cells:
        if cell.completeness == 1.0:
            complete[cell.arm].add((cell.task_id, cell.repetition))
    return complete[CONTROL] & complete[TREATMENT]


def _arm_rates(cells: Sequence[ProductLoopCellArtifact]) -> dict[str, float]:
    total = len(cells)
    if total == 0:
        return {}
    return {
        "answer_over_broad_rate": sum(
            1 for c in cells if c.answer_flag is ProductLoopAnswerFlag.ANSWER_OVER_BROAD
        )
        / total,
        "answer_unparsed_rate": sum(
            1 for c in cells if c.answer_flag is ProductLoopAnswerFlag.ANSWER_UNPARSED
        )
        / total,
        "failed_rate": sum(1 for c in cells if c.status is ProductLoopCellStatus.FAILED) / total,
        "hook_invocation_mean": statistics.fmean(float(c.hook_invocations) for c in cells),
        "mean_answer_precision": statistics.fmean(c.answer_precision for c in cells),
        "no_product_use_rate": sum(1 for c in cells if c.no_product_use) / total,
        "stale_index_rate": sum(1 for c in cells if c.stale_index_event) / total,
    }


def _repetition_spread(cells: Sequence[ProductLoopCellArtifact]) -> dict[str, float]:
    """How much of any difference is agent nondeterminism rather than the product."""
    buckets: dict[tuple[ProductLoopArm, str], list[float]] = defaultdict(list)
    for cell in cells:
        buckets[(cell.arm, cell.task_id)].append(cell.completeness)
    spreads: dict[ProductLoopArm, list[float]] = defaultdict(list)
    for (arm, _task_id), values in buckets.items():
        if len(values) > 1:
            spreads[arm].append(max(values) - min(values))
    return {
        f"{arm.value}_mean_within_task_range": statistics.fmean(values)
        for arm, values in sorted(spreads.items(), key=lambda item: item[0].value)
        if values
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmarks/product_loop/results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cells = _load_cells(args.input)
    if not cells:
        print(f"no product-loop cells under {args.input}", file=sys.stderr)
        return 1
    task_repo = {cell.task_id: cell.repo for cell in cells}

    try:
        coverage = validate_product_loop_directory(args.input, task_ids=sorted(task_repo))
    except ProductLoopError as exc:
        print(f"artifact directory is not publishable: {exc}", file=sys.stderr)
        return 1

    all_means = _task_means(cells, product_using_only=False)
    observed, low, high = _cluster_bootstrap(_paired_differences(all_means, task_repo))
    using_means = _task_means(cells, product_using_only=True)
    using_observed, using_low, using_high = _cluster_bootstrap(
        _paired_differences(using_means, task_repo)
    )

    units = _both_complete_units(cells)
    by_arm = {arm: [c for c in cells if c.arm is arm] for arm in ProductLoopArm}
    fingerprints = sorted(
        {c.ambient_tool_fingerprint for c in cells if c.status is ProductLoopCellStatus.OK}
    )

    payload = {
        "analysis_version": 1,
        "spike_id": "R20",
        "preregistration": PREREGISTRATION_PATH,
        "evidence_class": "original-descriptive",
        "agent": {
            "name": AGENT_NAME,
            "version": AGENT_VERSION,
            "model": AGENT_MODEL,
            "billing_mode": BILLING_MODE,
            "ambient_tool_fingerprints": fingerprints,
        },
        "coverage": {
            "cells": coverage.cells,
            "planned_cells": coverage.planned_cells,
            "ok_cells": coverage.ok_cells,
            "failed_cells": coverage.failed_cells,
            "repetitions": REPETITIONS,
            "tasks": len(task_repo),
            "repositories": len(set(task_repo.values())),
        },
        "cost": {
            "ceiling_usd": COST_CEILING_USD,
            "is_modelled_not_billed": True,
            "total_modelled_usd": coverage.total_modelled_cost_usd,
        },
        "primary": {
            "metric": "mean required-file completeness of the agent's final answer",
            "control_arm": CONTROL.value,
            "treatment_arm": TREATMENT.value,
            "control_mean": statistics.fmean(
                value for (arm, _task_id), value in all_means.items() if arm is CONTROL
            ),
            "treatment_mean": statistics.fmean(
                value for (arm, _task_id), value in all_means.items() if arm is TREATMENT
            ),
            "mean_difference": observed,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "clustering_unit": "source repository",
            "margins": {"mwg": MWG, "nim": NIM, "eqm": EQM},
            "verdict": (
                "descriptive baseline; no cross-tool claim is published from this population"
            ),
        },
        "secondary_product_using_only": {
            "exploratory": True,
            "mean_difference": using_observed,
            "ci_low": using_low,
            "ci_high": using_high,
        },
        "secondary_efficiency": {
            "exploratory": True,
            "all_cells": {
                arm.value: _efficiency(by_arm[arm])
                for arm in sorted(ProductLoopArm, key=lambda a: a.value)
            },
            "both_complete_units": {
                "unit_count": len(units),
                **{
                    arm.value: _efficiency(
                        [c for c in by_arm[arm] if (c.task_id, c.repetition) in units]
                    )
                    for arm in sorted(ProductLoopArm, key=lambda a: a.value)
                },
            },
        },
        "secondary_rates": {
            "exploratory": True,
            **{
                arm.value: _arm_rates(by_arm[arm])
                for arm in sorted(ProductLoopArm, key=lambda a: a.value)
            },
        },
        "secondary_repetition_spread": {"exploratory": True, **_repetition_spread(cells)},
    }

    document = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(document)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
