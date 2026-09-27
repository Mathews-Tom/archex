"""Measure the retrieval-honesty patch against the release it replaces.

The patch removes query-expansion vocabulary that mapped onto individual
benchmark task questions, ranks data files below code, and changes what context
receipts claim. Each of those can move retrieval quality, so this script runs
the same tasks through the same `archex_query` path under each arm and pairs the
cells by task:

- `control` — the previous release. Run this script from the branch, but with
  the previous release's environment (``uv run --project <release-worktree>``).
- `expansion_only` — this branch with the data-file ranking penalty neutralised
  in-process, isolating the expansion removal.
- `patched` — this branch as shipped.

```bash
uv run --project /path/to/release-worktree python scripts/review_findings_ablation.py \
    collect --arm control --output .archex/review-findings/control.json
uv run python scripts/review_findings_ablation.py \
    collect --arm expansion_only --neutralise-data-penalty \
    --output .archex/review-findings/expansion_only.json
uv run python scripts/review_findings_ablation.py \
    collect --arm patched --output .archex/review-findings/patched.json
uv run python scripts/review_findings_ablation.py analyze \
    --records .archex/review-findings/control.json .archex/review-findings/expansion_only.json \
    .archex/review-findings/patched.json \
    --subset-manifest benchmarks/headtohead/results/manifest.yaml \
    --output benchmarks/evidence/review-findings-ablation.json
```

This is measurement only. It registers no strategy and changes no default.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkRetrievalOptions
from archex.benchmark.runner import repo_path_for_task
from archex.benchmark.strategies import (
    reset_benchmark_retrieval_options,
    run_archex_query,
    set_benchmark_retrieval_options,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928
ARMS = ("control", "expansion_only", "patched")
PAIRS = (("control", "expansion_only"), ("control", "patched"), ("expansion_only", "patched"))
METRICS = (
    "required_file_recall",
    "recall",
    "precision",
    "f1_score",
    "token_efficiency",
    "bundle_tokens",
    "bundle_completion_tokens",
    "token_efficiency_with_completion",
)


@contextmanager
def _data_penalty_neutralised() -> Iterator[None]:
    """Stop data files from being treated as support files for the block."""
    from archex.serve import context

    original = context._is_data_file  # pyright: ignore[reportPrivateUsage]

    def _never(file_path: str, query_terms: set[str]) -> bool:  # noqa: ARG001 - signature parity
        return False

    context._is_data_file = _never  # pyright: ignore[reportPrivateUsage]
    try:
        yield
    finally:
        context._is_data_file = original  # pyright: ignore[reportPrivateUsage]


@contextmanager
def _receipt_capture(sink: list[Any]) -> Iterator[None]:
    """Record every finalized bundle's receipt while the block runs."""
    from archex import api

    original = api._finalize_context_bundle  # pyright: ignore[reportPrivateUsage]

    def _capturing(*args: Any, **kwargs: Any) -> Any:
        bundle = original(*args, **kwargs)
        sink.append(bundle.receipt)
        return bundle

    api._finalize_context_bundle = _capturing  # pyright: ignore[reportPrivateUsage]
    try:
        yield
    finally:
        api._finalize_context_bundle = original  # pyright: ignore[reportPrivateUsage]


def _cell(task: Any, repo_path: Path, arm: str) -> dict[str, Any]:
    receipts: list[Any] = []
    with _receipt_capture(receipts):
        result = run_archex_query(task, repo_path)
    receipt = receipts[-1] if receipts else None
    return {
        "task_id": task.task_id,
        "repo": task.repo,
        "arm": arm,
        "required_file_recall": result.required_file_recall,
        "recall": result.recall,
        "precision": result.precision,
        "f1_score": result.f1_score,
        "token_efficiency": result.token_efficiency,
        "all_required_files_present": result.all_required_files_present,
        "bundle_tokens": result.tokens_total,
        "bundle_completion_tokens": result.bundle_completion_tokens,
        "token_efficiency_with_completion": result.token_efficiency_with_completion,
        "receipt_accuracy": result.receipt_accuracy,
        "context_complete": None if receipt is None else str(receipt.context_complete),
        "context_complete_reason": (
            None if receipt is None else str(receipt.context_complete_reason)
        ),
        "recommended_next_action": (
            None if receipt is None else str(receipt.recommended_next_action)
        ),
        "query_terms_unmatched": getattr(receipt, "query_terms_unmatched", None),
        "result_files": result.result_files,
    }


def collect(tasks_dir: Path, output: Path, arm: str, *, neutralise_data_penalty: bool) -> None:
    tasks = load_tasks(tasks_dir)
    records: list[dict[str, Any]] = []
    if output.exists():
        records = json.loads(output.read_text(encoding="utf-8"))["records"]
    done = {r["task_id"] for r in records}

    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup: list[Path] = []
    token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(warm_cache=True))

    def flush() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"arm": arm, "tasks": len(records), "records": records}
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        for position, task in enumerate(tasks, start=1):
            if task.task_id in done:
                continue
            repo_path = repo_path_for_task(task, repo_cache, cleanup)
            print(f"[{position}/{len(tasks)}] {arm} {task.task_id}", flush=True)
            if neutralise_data_penalty:
                with _data_penalty_neutralised():
                    records.append(_cell(task, repo_path, arm))
            else:
                records.append(_cell(task, repo_path, arm))
            flush()
    finally:
        reset_benchmark_retrieval_options(token)
        flush()
        for path in cleanup:
            shutil.rmtree(path, ignore_errors=True)
    print(f"wrote {output} ({len(records)} cells)")


def _cluster_bootstrap(by_repo: dict[str, list[float]]) -> tuple[float, float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    repos = sorted(by_repo)
    observed = statistics.fmean(v for r in repos for v in by_repo[r])
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


def _arm_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cells)
    accuracy = [c["receipt_accuracy"] for c in cells]
    return {
        "tasks": n,
        **{m: statistics.fmean(c[m] for c in cells) for m in METRICS},
        "missed_required_task_rate": sum(not c["all_required_files_present"] for c in cells) / n,
        "zero_recall_tasks": sorted(c["task_id"] for c in cells if c["required_file_recall"] == 0),
        "receipt_accuracy": {
            "true": sum(a is True for a in accuracy),
            "false": sum(a is False for a in accuracy),
            "unknown": sum(a is None for a in accuracy),
        },
        "receipt_status_by_required_files": dict(
            sorted(
                Counter(
                    f"{c['context_complete']}|"
                    f"{'present' if c['all_required_files_present'] else 'missing'}"
                    for c in cells
                ).items()
            )
        ),
        "receipt_reasons": dict(
            sorted(Counter(str(c["context_complete_reason"]) for c in cells).items())
        ),
    }


def _pair_summary(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    shared = sorted(before.keys() & after.keys())
    out: dict[str, Any] = {"paired_tasks": len(shared)}
    for metric in (*METRICS, "all_required_files_present"):
        by_repo: dict[str, list[float]] = defaultdict(list)
        for task_id in shared:
            delta = float(after[task_id][metric]) - float(before[task_id][metric])
            by_repo[before[task_id]["repo"]].append(delta)
        mean, low, high = _cluster_bootstrap(by_repo)
        out[metric] = {"mean_delta": mean, "ci95": [low, high]}
    out["new_zero_recall"] = sorted(
        t
        for t in shared
        if after[t]["required_file_recall"] == 0 and before[t]["required_file_recall"] > 0
    )
    out["recovered_from_zero_recall"] = sorted(
        t
        for t in shared
        if before[t]["required_file_recall"] == 0 and after[t]["required_file_recall"] > 0
    )
    out["changed_result_files"] = sorted(
        t for t in shared if before[t]["result_files"] != after[t]["result_files"]
    )
    out["per_task_required_file_recall"] = {
        t: [before[t]["required_file_recall"], after[t]["required_file_recall"]]
        for t in shared
        if before[t]["required_file_recall"] != after[t]["required_file_recall"]
    }
    return out


def analyze(records_paths: list[Path], subset_manifest: Path | None, output: Path) -> None:
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for path in records_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_arm[payload["arm"]] = {r["task_id"]: r for r in payload["records"]}
    subset: set[str] | None = None
    subset_name: str | None = None
    if subset_manifest is not None:
        manifest = yaml.safe_load(subset_manifest.read_text(encoding="utf-8"))
        subset = set(manifest["task_subset"])
        subset_name = str(manifest["name"])

    def scoped(cells: dict[str, dict[str, Any]], only: set[str] | None) -> dict[str, Any]:
        return {t: c for t, c in cells.items() if only is None or t in only}

    report: dict[str, Any] = {
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "clustering_unit": "source repository",
        },
        "arms": {arm: _arm_summary(list(cells.values())) for arm, cells in by_arm.items()},
        "pairs": {
            f"{a}->{b}": _pair_summary(by_arm[a], by_arm[b])
            for a, b in PAIRS
            if a in by_arm and b in by_arm
        },
    }
    if subset is not None:
        report["subset"] = {
            "manifest": subset_name,
            "arms": {
                arm: _arm_summary(list(scoped(cells, subset).values()))
                for arm, cells in by_arm.items()
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--arm", choices=ARMS, required=True)
    c.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/tasks"))
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--neutralise-data-penalty", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--records", type=Path, nargs="+", required=True)
    a.add_argument("--subset-manifest", type=Path)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(
            args.tasks_dir,
            args.output,
            args.arm,
            neutralise_data_penalty=args.neutralise_data_penalty,
        )
    else:
        analyze(args.records, args.subset_manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
