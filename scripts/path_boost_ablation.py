"""Ablate the filename-alignment boost and measure what it is worth.

`archex.serve.context._path_alignment_boost` multiplies a chunk's weighted score
by up to 3.0 when the file's basename stem matches a query term. On
caller-localization queries that boost was observed promoting files whose names
merely collide with an expansion-injected term over files that actually contain
the queried identifier (see `docs/RETRIEVAL_DEFAULT_DECISIONS.md`).

That single observation does not justify touching a multiplier on the product
path for every query. This script measures the boost's corpus-wide worth instead:
the same tasks, the same `archex_query` path, the same index, run twice with the
boost live and neutralised, so the difference is the boost and nothing else.

```bash
uv run python scripts/path_boost_ablation.py collect \
    --tasks-dir benchmarks/tasks --output benchmarks/swebench_pro/path-boost-records.json
uv run python scripts/path_boost_ablation.py analyze \
    --records benchmarks/swebench_pro/path-boost-records.json \
    --output benchmarks/evidence/path-boost-ablation.json
```

This is measurement only. It registers no strategy, changes no default, and its
neutralisation is applied in-process for the ablation arm alone.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkRetrievalOptions
from archex.benchmark.runner import repo_path_for_task
from archex.benchmark.strategies import (
    reset_benchmark_retrieval_options,
    run_archex_query,
    set_benchmark_retrieval_options,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260909
METRICS = (
    "required_file_recall",
    "recall",
    "precision",
    "f1_score",
    "bundle_tokens",
    "bundle_completion_tokens",
    "token_efficiency_with_completion",
)


@contextmanager
def _boost_neutralised() -> Iterator[None]:
    """Force `_path_alignment_boost` to 1.0 for the duration of the block."""
    from archex.serve import context

    original = context._path_alignment_boost  # pyright: ignore[reportPrivateUsage]

    def _neutral(file_path: str, query_terms: set[str]) -> float:  # noqa: ARG001 - signature parity
        return 1.0

    context._path_alignment_boost = _neutral  # pyright: ignore[reportPrivateUsage]
    try:
        yield
    finally:
        context._path_alignment_boost = original  # pyright: ignore[reportPrivateUsage]


def _cell(task: Any, repo_path: Path, arm: str) -> dict[str, Any]:
    result = run_archex_query(task, repo_path)
    return {
        "task_id": task.task_id,
        "repo": task.repo,
        "family": task.family.value,
        "arm": arm,
        "required_file_recall": result.required_file_recall,
        "recall": result.recall,
        "precision": result.precision,
        "f1_score": result.f1_score,
        "all_required_files_present": result.all_required_files_present,
        "bundle_tokens": result.tokens_total,
        "bundle_completion_tokens": result.bundle_completion_tokens,
        "token_efficiency_with_completion": result.token_efficiency_with_completion,
        "region_recall": result.region_recall,
        "line_recall": result.line_recall,
        "result_files": result.result_files,
    }


def collect(tasks_dir: Path, output: Path) -> None:
    tasks = load_tasks(tasks_dir)
    records: list[dict[str, Any]] = []
    if output.exists():
        records = json.loads(output.read_text(encoding="utf-8"))["records"]
    done = {(r["task_id"], r["arm"]) for r in records}

    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup: list[Path] = []
    token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(warm_cache=True))

    def flush() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {"tasks": len({r["task_id"] for r in records}), "records": records}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        for position, task in enumerate(tasks, start=1):
            if {(task.task_id, "boost_on"), (task.task_id, "boost_off")} <= done:
                continue
            repo_path = repo_path_for_task(task, repo_cache, cleanup)
            print(f"[{position}/{len(tasks)}] {task.task_id}", flush=True)
            if (task.task_id, "boost_on") not in done:
                records.append(_cell(task, repo_path, "boost_on"))
            if (task.task_id, "boost_off") not in done:
                with _boost_neutralised():
                    records.append(_cell(task, repo_path, "boost_off"))
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


def analyze(records_path: Path, output: Path) -> None:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["records"]
    on = {r["task_id"]: r for r in rows if r["arm"] == "boost_on"}
    off = {r["task_id"]: r for r in rows if r["arm"] == "boost_off"}
    paired = sorted(set(on) & set(off))

    deltas: dict[str, Any] = {}
    for metric in METRICS:
        by_repo: dict[str, list[float]] = {}
        for task_id in paired:
            by_repo.setdefault(on[task_id]["repo"], []).append(
                float(on[task_id][metric]) - float(off[task_id][metric])
            )
        mean, low, high = _cluster_bootstrap(by_repo)
        deltas[metric] = {"boost_minus_neutral": mean, "ci95_low": low, "ci95_high": high}

    for metric in ("region_recall", "line_recall"):
        by_repo = {}
        for task_id in paired:
            a, b = on[task_id][metric], off[task_id][metric]
            if a is None or b is None:
                continue
            by_repo.setdefault(on[task_id]["repo"], []).append(float(a) - float(b))
        if by_repo:
            mean, low, high = _cluster_bootstrap(by_repo)
            deltas[metric] = {
                "boost_minus_neutral": mean,
                "ci95_low": low,
                "ci95_high": high,
                "labelled_tasks": sum(len(v) for v in by_repo.values()),
            }

    changed = [t for t in paired if on[t]["result_files"] != off[t]["result_files"]]
    helped = [t for t in paired if on[t]["required_file_recall"] > off[t]["required_file_recall"]]
    hurt = [t for t in paired if on[t]["required_file_recall"] < off[t]["required_file_recall"]]

    evidence = {
        "analysis": "path-alignment-boost-ablation",
        "question": (
            "Is the up-to-3.0x filename-alignment boost in assemble_context worth its "
            "cost on the benchmark corpus?"
        ),
        "method": {
            "arms": "boost_on (shipped) vs boost_off (_path_alignment_boost forced to 1.0)",
            "varied": "the boost only; same tasks, index, budgets, and strategy",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "clusters": "source repository",
        },
        "paired_tasks": len(paired),
        "tasks_with_a_changed_file_set": len(changed),
        "tasks_where_boost_improved_required_file_recall": sorted(helped),
        "tasks_where_boost_hurt_required_file_recall": sorted(hurt),
        "deltas": deltas,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    print(f"paired tasks: {len(paired)}   file set changed by the boost: {len(changed)}")
    print(f"boost helped required-file recall: {len(helped)}   hurt: {len(hurt)}")
    for metric, d in deltas.items():
        print(
            f"   {metric:38s} {d['boost_minus_neutral']:+12.4f} "
            f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]"
        )
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/tasks"))
    c.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--records", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.tasks_dir, args.output)
    else:
        analyze(args.records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
