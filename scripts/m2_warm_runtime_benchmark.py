"""Evidence script: measure warm query() p95 latency, peak RSS, and recall
consistency with vs without QueryRuntime (M2's warm-serving cache).

Run as two separate subprocess invocations (one per mode) so peak-RSS
measurements are never cross-contaminated between the baseline and
runtime-enabled process:

    uv run python scripts/m2_warm_runtime_benchmark.py \\
        <repo_path> baseline <output.json> [tasks_dir] [repeats]
    uv run python scripts/m2_warm_runtime_benchmark.py \\
        <repo_path> runtime <output.json> [tasks_dir] [repeats]

``baseline`` never passes a runtime to query() (byte-equivalent to the
pre-M2 code path, per the QueryRuntime PR's own equivalence tests).
``runtime`` shares one QueryRuntime across every repeated query. Each mode
primes the on-disk index cache once (discarded from timing) before timing
`repeats` repeated calls per task question, then reports p50/p95/p99
latency across all repeated calls, peak process RSS, and per-task recall
against `expected_files` (when a tasks_dir of BenchmarkTask YAML files is
given) using the project's own compute_recall/compute_required_file_metrics
definitions — so a recall difference between modes is measured with the
same yardstick the rest of the benchmark suite uses.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

from archex.api import query
from archex.benchmark.loader import load_task
from archex.benchmark.strategies import compute_recall, compute_required_file_metrics
from archex.models import Config, RepoSource
from archex.serve.runtime import QueryRuntime


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB.

    ru_maxrss is bytes on macOS/BSD and KiB on Linux.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


def main() -> int:
    repo_path = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    if mode not in ("baseline", "runtime"):
        raise SystemExit(f"mode must be 'baseline' or 'runtime', got {mode!r}")
    output_path = Path(sys.argv[3])
    tasks_dir = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 and sys.argv[4] else None
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    source = RepoSource(local_path=str(repo_path))
    config = Config(cache=True, cache_dir=str(repo_path / ".archex_m2_bench_cache"))
    runtime = QueryRuntime() if mode == "runtime" else None

    questions: list[tuple[str, str, list[str]]] = []
    if tasks_dir is not None:
        for task_file in sorted(tasks_dir.glob("archex_*.yaml")):
            task = load_task(task_file)
            questions.append((task.task_id, task.question, task.expected_files))
    if not questions:
        questions = [
            ("q_entry_point", "how does the main entry point work", []),
            ("q_config", "how is configuration loaded and validated", []),
            ("q_error_handling", "how are errors and exceptions handled", []),
            ("q_models", "what are the core data models and classes", []),
            ("q_tests", "how does the test suite set up fixtures", []),
        ]

    # Cold priming pass — builds/opens the on-disk index once; discarded from timing.
    for _task_id, question, _expected in questions:
        query(source, question, config=config, runtime=runtime)

    latencies_ms: dict[str, list[float]] = {}
    recall_by_task: dict[str, float | None] = {}
    required_file_recall_by_task: dict[str, float | None] = {}
    for task_id, question, expected_files in questions:
        latencies_ms[task_id] = []
        last_files: set[str] = set()
        for _ in range(repeats):
            t0 = time.perf_counter()
            bundle = query(source, question, config=config, runtime=runtime)
            latencies_ms[task_id].append((time.perf_counter() - t0) * 1000)
            last_files = {c.chunk.file_path for c in bundle.chunks}
        if expected_files:
            recall_by_task[task_id] = compute_recall(last_files, expected_files)
            required_file_recall_by_task[task_id] = compute_required_file_metrics(
                last_files, expected_files
            )[0]
        else:
            recall_by_task[task_id] = None
            required_file_recall_by_task[task_id] = None

    if runtime is not None:
        runtime.close()

    all_latencies = [v for values in latencies_ms.values() for v in values]
    scored_recalls = [v for v in recall_by_task.values() if v is not None]
    result = {
        "mode": mode,
        "repo": str(repo_path),
        "task_count": len(questions),
        "repeats_per_task": repeats,
        "peak_rss_mb": round(_peak_rss_mb(), 2),
        "p50_latency_ms": round(_percentile(all_latencies, 50), 2),
        "p95_latency_ms": round(_percentile(all_latencies, 95), 2),
        "p99_latency_ms": round(_percentile(all_latencies, 99), 2),
        "zero_recall_task_count": sum(1 for v in scored_recalls if v == 0.0),
        "per_task_latencies_ms": {k: [round(v, 2) for v in vs] for k, vs in latencies_ms.items()},
        "per_task_recall": recall_by_task,
        "per_task_required_file_recall": required_file_recall_by_task,
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
