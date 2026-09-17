"""Measure what archex's bundle buys per token, across the whole budget ladder.

`scripts/swebench_pro_token_headroom.py` establishes the constraint: in a
long-horizon agent loop every front-loaded token is re-sent on every later call,
so a retrieval bundle is only net-positive below a break-even size. That number
is useless without its other half — **how much retrieval quality archex actually
gives up at each bundle size.** This script measures that curve on the repo's own
task corpus, using the product `archex_query` path and the product metrics.

```bash
uv run python scripts/archex_bundle_budget_curve.py collect \
    --tasks-dir benchmarks/tasks \
    --output benchmarks/swebench_pro/bundle-budget-records.json

uv run python scripts/archex_bundle_budget_curve.py analyze \
    --records benchmarks/swebench_pro/bundle-budget-records.json \
    --headroom benchmarks/evidence/swebench-pro-token-headroom.json \
    --output benchmarks/evidence/archex-bundle-budget-curve.json
```

Protocol notes that matter for reading the numbers:

* Only `token_budget` varies. The task, the repository revision, the index
  configuration and the strategy are identical across the ladder, so a
  difference between two rungs is a budget effect and nothing else.
* The index cache is enabled for the sweep (`warm_cache`), because the same
  repository is queried once per rung. This makes latency here **not**
  comparable to the cold-index `archex_query` figures elsewhere in
  `benchmarks/`; retrieval quality and bundle size are unaffected.
* `tokens_total` is the bundle archex actually returned, which is not the
  requested budget — it is what the assembler spent. The curve is plotted
  against the delivered size, since that is what an agent pays for.
* Inference is the house standard: 10 000-resample percentile cluster bootstrap
  over source repositories, seed 20260909.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkRetrievalOptions, BenchmarkTask
from archex.benchmark.runner import repo_path_for_task
from archex.benchmark.strategies import (
    reset_benchmark_retrieval_options,
    run_archex_query,
    set_benchmark_retrieval_options,
)

BUDGET_LADDER = (1_024, 2_048, 3_072, 4_096, 5_120, 6_144, 8_192, 12_288, 16_384)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260909


def collect(tasks_dir: Path, output: Path, self_only: bool, family: str | None) -> None:
    tasks = load_tasks(tasks_dir)
    if self_only:
        tasks = [task for task in tasks if task.repo == "."]
    if family is not None:
        tasks = [task for task in tasks if task.family.value == family]
    if not tasks:
        raise SystemExit("no tasks selected")

    records: list[dict[str, Any]] = []
    if output.exists():
        records = json.loads(output.read_text(encoding="utf-8"))["records"]
    measured = {record["task_id"] for record in records}
    pending = [task for task in tasks if task.task_id not in measured]
    print(
        f"{len(tasks)} tasks selected, {len(measured)} already measured, {len(pending)} to run",
        file=sys.stderr,
    )

    def flush() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "tasks": len({record["task_id"] for record in records}),
                    "budgets": list(BUDGET_LADDER),
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup: list[Path] = []
    token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(warm_cache=True))
    try:
        for position, task in enumerate(pending, start=1):
            repo_path = repo_path_for_task(task, repo_cache, cleanup)
            print(
                f"[{position}/{len(pending)}] {task.task_id} ({task.repo})",
                file=sys.stderr,
                flush=True,
            )
            for budget in BUDGET_LADDER:
                rung: BenchmarkTask = task.model_copy(update={"token_budget": budget})
                started = time.perf_counter()
                result = run_archex_query(rung, repo_path)
                records.append(
                    {
                        "task_id": task.task_id,
                        "repo": task.repo,
                        "family": task.family.value,
                        "expected_files": len(task.expected_files),
                        "requested_budget": budget,
                        "bundle_tokens": result.tokens_total,
                        "recall": result.recall,
                        "precision": result.precision,
                        "f1_score": result.f1_score,
                        "required_file_recall": result.required_file_recall,
                        "all_required_files_present": result.all_required_files_present,
                        "missed_required_task": result.missed_required_task_rate >= 1.0,
                        "bundle_completion_tokens": result.bundle_completion_tokens,
                        "token_efficiency": result.token_efficiency,
                        "token_efficiency_with_completion": (
                            result.token_efficiency_with_completion
                        ),
                        "region_recall": result.region_recall,
                        "line_recall": result.line_recall,
                        "wall_time_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )
            flush()
    finally:
        reset_benchmark_retrieval_options(token)
        flush()
        for path in cleanup:
            shutil.rmtree(path, ignore_errors=True)
    print(f"wrote {output} ({len(records)} cells)", file=sys.stderr)


def _cluster_bootstrap(by_repo: dict[str, list[float]]) -> tuple[float, float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    repos = sorted(by_repo)
    observed = statistics.fmean(value for repo in repos for value in by_repo[repo])
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled: list[float] = []
        for _ in repos:
            sampled.extend(by_repo[rng.choice(repos)])
        means.append(statistics.fmean(sampled))
    means.sort()
    return (
        observed,
        means[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))],
        means[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))],
    )


def _slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bundle size and required-file recall for one subset of a rung."""
    return {
        "cells": len(rows),
        "mean_bundle_tokens": statistics.fmean(row["bundle_tokens"] for row in rows),
        "mean_required_file_recall": statistics.fmean(row["required_file_recall"] for row in rows),
        "all_required_present_rate": statistics.fmean(
            1.0 if row["all_required_files_present"] else 0.0 for row in rows
        ),
    }


def _rung(records: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    rows = [record for record in records if record["requested_budget"] == budget]
    by_repo: dict[str, list[float]] = {}
    for row in rows:
        by_repo.setdefault(row["repo"], []).append(row["required_file_recall"])
    mean, low, high = _cluster_bootstrap(by_repo)
    labelled = [row for row in rows if row["region_recall"] is not None]
    return {
        "requested_budget": budget,
        "cells": len(rows),
        "mean_bundle_tokens": statistics.fmean(row["bundle_tokens"] for row in rows),
        "median_bundle_tokens": statistics.median(row["bundle_tokens"] for row in rows),
        "mean_required_file_recall": mean,
        "ci95_low": low,
        "ci95_high": high,
        "all_required_present_rate": statistics.fmean(
            1.0 if row["all_required_files_present"] else 0.0 for row in rows
        ),
        "missed_required_task_rate": statistics.fmean(
            1.0 if row["missed_required_task"] else 0.0 for row in rows
        ),
        "mean_recall": statistics.fmean(row["recall"] for row in rows),
        "mean_precision": statistics.fmean(row["precision"] for row in rows),
        "mean_bundle_completion_tokens": statistics.fmean(
            row["bundle_completion_tokens"] for row in rows
        ),
        "mean_total_tokens": statistics.fmean(
            row["bundle_tokens"] + row["bundle_completion_tokens"] for row in rows
        ),
        "mean_region_recall": (
            statistics.fmean(row["region_recall"] for row in labelled) if labelled else None
        ),
        "labelled_region_cells": len(labelled),
        "by_corpus": {
            label: _slice(subset)
            for label, subset in (
                ("self", [row for row in rows if row["repo"] == "."]),
                ("external", [row for row in rows if row["repo"] != "."]),
            )
            if subset
        },
        "by_family": {
            family: _slice([row for row in rows if row["family"] == family])
            for family in sorted({row["family"] for row in rows})
        },
    }


def _largest_affordable(curve: list[dict[str, Any]], budget_tokens: float) -> dict[str, Any] | None:
    affordable = [rung for rung in curve if rung["mean_bundle_tokens"] <= budget_tokens]
    return affordable[-1] if affordable else None


def _per_intent(records: list[dict[str, Any]], tasks_dir: Path) -> list[dict[str, Any]]:
    """Saturation per query intent, against the shipped preset for that intent.

    The product does not apply one budget: ``archex.api.query`` routes an
    unspecified budget through ``token_budget_for_query``, so the only
    actionable question is which *preset* sits above its own saturation point.
    """
    from archex.benchmark.loader import load_tasks
    from archex.serve.intent import INTENT_TOKEN_BUDGETS, classify_intent

    intent_of = {task.task_id: classify_intent(task.question) for task in load_tasks(tasks_dir)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        intent = intent_of.get(record["task_id"])
        if intent is not None:
            grouped.setdefault(intent.value, []).append(record)

    out: list[dict[str, Any]] = []
    for intent_value in sorted(grouped):
        rows = grouped[intent_value]
        budgets = sorted({row["requested_budget"] for row in rows})
        rungs = {
            budget: _slice([r for r in rows if r["requested_budget"] == budget])
            for budget in budgets
        }
        regions = {
            budget: [
                r["region_recall"]
                for r in rows
                if r["requested_budget"] == budget and r["region_recall"] is not None
            ]
            for budget in budgets
        }
        best = max(rung["mean_required_file_recall"] for rung in rungs.values())
        saturation = next(
            budget
            for budget in budgets
            if rungs[budget]["mean_required_file_recall"] >= best - 1e-9
        )
        preset = INTENT_TOKEN_BUDGETS[
            next(i for i in INTENT_TOKEN_BUDGETS if i.value == intent_value)
        ]
        nearest = min(budgets, key=lambda budget: abs(budget - preset))
        out.append(
            {
                "intent": intent_value,
                "shipped_preset": preset,
                "tasks": len(rows) // len(budgets),
                "saturation_budget": saturation,
                "bundle_at_saturation": rungs[saturation]["mean_bundle_tokens"],
                "required_file_recall_at_saturation": rungs[saturation][
                    "mean_required_file_recall"
                ],
                "bundle_at_preset_rung": rungs[nearest]["mean_bundle_tokens"],
                "required_file_recall_at_preset_rung": rungs[nearest]["mean_required_file_recall"],
                "nearest_preset_rung": nearest,
                "labelled_region_tasks": len(regions[budgets[0]]),
                "region_recall_by_budget": {
                    str(budget): (statistics.fmean(regions[budget]) if regions[budget] else None)
                    for budget in budgets
                },
            }
        )
    return out


def analyze(records_path: Path, headroom_path: Path | None, output: Path, tasks_dir: Path) -> None:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = payload["records"]
    curve = [_rung(records, budget) for budget in payload["budgets"]]
    default_rung = next(rung for rung in curve if rung["requested_budget"] == 8_192)
    best_recall = max(rung["mean_required_file_recall"] for rung in curve)
    saturation = next(
        rung for rung in curve if rung["mean_required_file_recall"] >= best_recall - 1e-9
    )
    cheapest = min(curve, key=lambda rung: rung["mean_total_tokens"])

    constraints: list[dict[str, Any]] = []
    if headroom_path is not None:
        headroom = json.loads(headroom_path.read_text(encoding="utf-8"))
        for run in headroom["runs"]:
            breakeven = run["breakeven_bundle_tokens"]["mean"]
            affordable = _largest_affordable(curve, breakeven)
            constraints.append(
                {
                    "run": run["run"],
                    "breakeven_bundle_tokens": breakeven,
                    "breakeven_ci95": [
                        run["breakeven_bundle_tokens"]["ci95_low"],
                        run["breakeven_bundle_tokens"]["ci95_high"],
                    ],
                    "largest_affordable_budget": (
                        affordable["requested_budget"] if affordable else None
                    ),
                    "affordable_bundle_tokens": (
                        affordable["mean_bundle_tokens"] if affordable else None
                    ),
                    "affordable_required_file_recall": (
                        affordable["mean_required_file_recall"] if affordable else None
                    ),
                    "default_8192_bundle_tokens": default_rung["mean_bundle_tokens"],
                    "default_8192_required_file_recall": default_rung["mean_required_file_recall"],
                    "required_file_recall_given_up": (
                        default_rung["mean_required_file_recall"]
                        - affordable["mean_required_file_recall"]
                        if affordable
                        else None
                    ),
                }
            )

    evidence = {
        "analysis": "archex-bundle-budget-curve",
        "question": (
            "How much required-file recall does archex give up at each bundle size, "
            "and what does it deliver at the SWE-bench Pro break-even bundle?"
        ),
        "method": {
            "strategy": "archex_query (product path)",
            "varied": "token_budget only",
            "index_cache": "warm (latency here is not comparable to cold-index runs)",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "clusters": "source repository",
        },
        "tasks": payload["tasks"],
        "saturation": {
            "budget": saturation["requested_budget"],
            "bundle_tokens": saturation["mean_bundle_tokens"],
            "required_file_recall": saturation["mean_required_file_recall"],
            "default_budget": default_rung["requested_budget"],
            "default_bundle_tokens": default_rung["mean_bundle_tokens"],
            "default_required_file_recall": default_rung["mean_required_file_recall"],
            "default_tokens_bought_nothing": (
                default_rung["mean_bundle_tokens"] - saturation["mean_bundle_tokens"]
            ),
        },
        "cheapest_total": {
            "budget": cheapest["requested_budget"],
            "bundle_tokens": cheapest["mean_bundle_tokens"],
            "completion_tokens": cheapest["mean_bundle_completion_tokens"],
            "total_tokens": cheapest["mean_total_tokens"],
            "default_total_tokens": default_rung["mean_total_tokens"],
        },
        "curve": curve,
        "per_intent_presets": _per_intent(records, tasks_dir),
        "agent_loop_constraint": constraints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    print(
        f"{'budget':>8}{'bundle_tok':>12}{'req_recall':>12}{'ci95':>20}"
        f"{'all_present':>13}{'completion_tok':>16}"
    )
    for rung in curve:
        print(
            f"{rung['requested_budget']:8,}{rung['mean_bundle_tokens']:12,.0f}"
            f"{rung['mean_required_file_recall']:12.3f}"
            f"  [{rung['ci95_low']:.3f}, {rung['ci95_high']:.3f}]"
            f"{rung['all_required_present_rate']:13.3f}"
            f"{rung['mean_bundle_completion_tokens']:16,.0f}"
        )
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="sweep the budget ladder")
    collect_parser.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/tasks"))
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--self-only", action="store_true")
    collect_parser.add_argument("--family", default=None)

    analyze_parser = sub.add_parser("analyze", help="derive the curve artifact")
    analyze_parser.add_argument("--records", type=Path, required=True)
    analyze_parser.add_argument("--headroom", type=Path, default=None)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/tasks"))

    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.tasks_dir, args.output, args.self_only, args.family)
    else:
        analyze(args.records, args.headroom, args.output, args.tasks_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
