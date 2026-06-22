"""Benchmark execution engine: runs tasks across strategies and collects results."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
    TaskCompletionResult,
)
from archex.benchmark.strategies import (
    benchmark_index_config,
    benchmark_repo_source,
    default_strategy_registry,
    reset_benchmark_retrieval_options,
    set_benchmark_retrieval_options,
)
from archex.exceptions import ArchexIndexError, BenchmarkCloneError

if TYPE_CHECKING:
    from archex.benchmark.progress import BenchmarkProgress

logger = logging.getLogger(__name__)

DEFAULT_STRATEGIES: list[Strategy] = [
    Strategy.RAW_FILES,
    Strategy.RAW_RIPGREP,
    Strategy.ARCHEX_QUERY,
]

AVAILABLE_STRATEGIES: list[Strategy] = [
    *DEFAULT_STRATEGIES,
    Strategy.ARCHEX_SCOUT_FETCH,
    Strategy.ARCHEX_QUERY_FUSION,
    Strategy.ARCHEX_QUERY_HYBRID,
    Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT,
    Strategy.ARCHEX_QUERY_FUSION_RERANK,
    Strategy.CROSS_LAYER_FUSION,
    Strategy.ARCHEX_QUERY_TASK_AWARE,
    Strategy.ARCHEX_QUERY_COMPRESSED,
    Strategy.ARCHEX_QUERY_EFFICIENCY_PACKED,
    Strategy.ARCHEX_QUERY_DUAL_TRANSFORM,
    Strategy.ARCHEX_QUERY_BOUNDED_RERANK,
    Strategy.ARCHEX_QUERY_SUMMARY_SIDECAR,
    Strategy.ARCHEX_QUERY_GRAPH_MULTIHOP,
    Strategy.ARCHEX_QUERY_CONDITIONAL_RERANK,
    Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK,
    Strategy.ARCHEX_QUERY_DIVERSITY_PACKED,
]

_VECTOR_STRATEGIES: frozenset[Strategy] = frozenset(
    {
        Strategy.ARCHEX_QUERY_VECTOR,
        Strategy.SURROGATE_VECTOR,
        Strategy.ARCHEX_QUERY_FUSION,
        Strategy.ARCHEX_QUERY_FUSION_RERANK,
        Strategy.ARCHEX_QUERY_HYBRID,
        Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT,
        Strategy.CROSS_LAYER_FUSION,
    }
)


def _check_vector_available() -> bool:
    """Check if vector embedding dependencies are available (fastembed or sentence-transformers)."""
    try:
        import fastembed as _fe  # noqa: F401  # pyright: ignore[reportUnusedImport]

        return True
    except ImportError:
        pass
    try:
        import sentence_transformers as _st  # noqa: F401  # pyright: ignore[reportUnusedImport]

        return True
    except ImportError:
        return False


def _warm_repo_index(
    task: BenchmarkTask,
    repo_path: Path,
    retrieval_options: BenchmarkRetrievalOptions | None = None,
    *,
    warm_strategy: Strategy = Strategy.ARCHEX_QUERY_FUSION,
) -> None:
    """Pre-build the shared vector index so every vector strategy runs warm.

    The first vector strategy to execute otherwise absorbs the full embedding
    build (cold) while later ones reuse the cached store and .npz (warm), making
    per-strategy timings incomparable. One discarded query with the same vector
    config the strategies use populates both caches before the timed loop. The
    config must match run_archex_query_fusion so the cache key and vector .npz
    path line up; this warms VectorMode.RAW only (fusion, rerank, query_vector),
    not surrogate-mode strategies.
    """
    from archex.api import query
    from archex.models import Config, IndexConfig

    retrieval_options = retrieval_options or BenchmarkRetrievalOptions()
    warm_rerank = warm_strategy is Strategy.ARCHEX_QUERY_FUSION_RERANK
    if warm_strategy is Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT:
        base_config = IndexConfig(vector=True, quantize_vectors=True, quantize_bits=4)
    elif warm_strategy in {Strategy.ARCHEX_QUERY_HYBRID, Strategy.ARCHEX_QUERY_FUSION}:
        base_config = IndexConfig(vector=True, quantize_vectors=False)
    else:
        base_config = IndexConfig(vector=True, rerank=warm_rerank, quantize_vectors=False)
    token = set_benchmark_retrieval_options(retrieval_options)
    try:
        source = benchmark_repo_source(task, repo_path, strategy=warm_strategy)
        config = Config(cache=True, languages=task.languages)
        index_config = benchmark_index_config(
            base_config,
            strategy=warm_strategy,
        )
    finally:
        reset_benchmark_retrieval_options(token)
    query(
        source,
        task.question,
        token_budget=task.token_budget,
        explicit_token_budget=True,
        config=config,
        index_config=index_config,
    )


def _run_git(
    args: list[str], *, cwd: Path | None = None, timeout: int
) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured output, converting a timeout into BenchmarkCloneError."""
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkCloneError(f"git timed out after {timeout}s: {' '.join(args)}") from exc


def clone_at_commit(repo_slug: str, commit: str) -> tuple[Path, bool]:
    """Clone a GitHub repo and checkout a specific commit/tag. Returns (path, needs_cleanup).

    Raises BenchmarkCloneError (carrying git's stderr) when both the shallow-ref
    clone and the full-clone fallback fail — e.g. network error, rate limit, or
    an unresolvable ref — leaving no temp directory behind.
    """
    url = f"https://github.com/{repo_slug}.git"
    target = Path(tempfile.mkdtemp(prefix="archex-bench-"))

    # Try shallow clone at ref (works for tags and branches, much faster)
    shallow = _run_git(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", commit, url, str(target)],
        timeout=300,
    )
    if shallow.returncode == 0:
        return target, True

    # Fallback: full clone + checkout (needed for bare commit hashes)
    shutil.rmtree(target, ignore_errors=True)
    target = Path(tempfile.mkdtemp(prefix="archex-bench-"))
    full = _run_git(["git", "clone", "--quiet", url, str(target)], timeout=300)
    if full.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        detail = full.stderr.strip() or shallow.stderr.strip() or "unknown git error"
        raise BenchmarkCloneError(f"clone failed for {repo_slug}@{commit}: {detail}")

    checkout = _run_git(["git", "checkout", "--quiet", commit], cwd=target, timeout=30)
    if checkout.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        detail = checkout.stderr.strip() or "unknown git error"
        raise BenchmarkCloneError(f"checkout {commit} failed for {repo_slug}: {detail}")

    return target, True


def run_benchmark(
    task: BenchmarkTask,
    strategies: list[Strategy] | None = None,
    repo_path: Path | None = None,
    progress: BenchmarkProgress | None = None,
    retrieval_options: BenchmarkRetrievalOptions | None = None,
) -> BenchmarkReport:
    """Run a benchmark task across strategies. Clones repo if repo_path not provided."""
    if strategies is None:
        strategies = list(DEFAULT_STRATEGIES)
        if not _check_vector_available():
            strategies = [s for s in strategies if s not in _VECTOR_STRATEGIES]

    needs_cleanup = False
    if repo_path is None:
        if task.repo == ".":
            repo_path = Path.cwd()
        else:
            repo_path, needs_cleanup = clone_at_commit(task.repo, task.commit)

    retrieval_options = retrieval_options or BenchmarkRetrievalOptions()
    token = set_benchmark_retrieval_options(retrieval_options)
    try:
        warm_strategies: list[Strategy] = []
        if _check_vector_available() and any(s in _VECTOR_STRATEGIES for s in strategies):
            warm_rerank = Strategy.ARCHEX_QUERY_FUSION_RERANK in strategies
            warm_strategies.append(
                Strategy.ARCHEX_QUERY_FUSION_RERANK if warm_rerank else Strategy.ARCHEX_QUERY_FUSION
            )
            if Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT in strategies:
                warm_strategies.append(Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT)
        if warm_strategies:
            _log_progress(progress, f"  warming vector index for {task.task_id}...")
            if progress is not None:
                progress.start_warmup()
            for warm_strategy in warm_strategies:
                _warm_repo_index(
                    task,
                    repo_path,
                    retrieval_options,
                    warm_strategy=warm_strategy,
                )
        if progress is not None:
            progress.finish_warmup(strategies)

        results: list[BenchmarkResult] = []
        for strategy in strategies:
            if progress is not None:
                progress.start_strategy(strategy)
            runner = default_strategy_registry.get(strategy)
            if runner is None:
                logger.warning("No runner for strategy %s, skipping", strategy)
                if progress is not None:
                    progress.finish_strategy()
                continue
            try:
                result = runner(task, repo_path)
                results.append(result)
                logger.info(
                    "%s: %d tokens, recall=%.2f, %.0fms",
                    strategy.value,
                    result.tokens_total,
                    result.recall,
                    result.wall_time_ms,
                )
            except (NotImplementedError, ArchexIndexError) as exc:
                logger.info("Skipping %s: %s", strategy.value, exc)
            if progress is not None:
                progress.finish_strategy()

        # Compute baseline and backfill savings_vs_raw
        baseline_tokens = 0
        raw_result = next((r for r in results if r.strategy == Strategy.RAW_FILES), None)
        if raw_result is not None:
            baseline_tokens = raw_result.tokens_total

        if baseline_tokens > 0:
            for result in results:
                if result.strategy != Strategy.RAW_FILES:
                    result.savings_vs_raw = round(
                        (1 - result.tokens_total / baseline_tokens) * 100,
                        1,
                    )
        if raw_result is not None:
            baseline_completion = raw_result.task_completion_result
            for result in results:
                if (
                    baseline_completion != TaskCompletionResult.UNKNOWN
                    and result.task_completion_result != TaskCompletionResult.UNKNOWN
                ):
                    result.completion_preserved = (
                        result.task_completion_result == baseline_completion
                    )

        return BenchmarkReport(
            task_id=task.task_id,
            repo=task.repo,
            question=task.question,
            results=results,
            baseline_tokens=baseline_tokens,
            median_latency_ms=_percentile(
                [result.wall_time_ms for result in results],
                0.50,
            ),
            p95_latency_ms=_percentile(
                [result.wall_time_ms for result in results],
                0.95,
            ),
        )
    finally:
        reset_benchmark_retrieval_options(token)
        if needs_cleanup:
            shutil.rmtree(repo_path, ignore_errors=True)


def run_all(
    tasks_dir: Path,
    output_dir: Path,
    strategies: list[Strategy] | None = None,
    task_filter: str | None = None,
    self_only: bool = False,
    progress: BenchmarkProgress | None = None,
    tasks: list[BenchmarkTask] | None = None,
    retrieval_options: BenchmarkRetrievalOptions | None = None,
) -> list[BenchmarkReport]:
    """Load all tasks, run benchmarks, write results to output_dir."""
    if tasks is None:
        tasks = load_selected_tasks(tasks_dir, task_filter=task_filter, self_only=self_only)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BenchmarkReport] = []
    failures: list[tuple[str, str]] = []
    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup_paths: list[Path] = []

    try:
        for i, task in enumerate(tasks, 1):
            if progress is not None:
                progress.start_task(task)
                if not progress.live_display_enabled:
                    progress.console.log(f"[{i}/{len(tasks)}] {task.task_id} ({task.repo})")
            else:
                print(f"[{i}/{len(tasks)}] {task.task_id} ({task.repo})", file=sys.stderr)
            try:
                task_repo_path = repo_path_for_task(task, repo_cache, cleanup_paths)
                report = run_benchmark(
                    task,
                    strategies=strategies,
                    repo_path=task_repo_path,
                    progress=progress,
                    retrieval_options=retrieval_options,
                )
            except BenchmarkCloneError as exc:
                # Isolate per-task clone failures (network, rate limit, bad ref) so one
                # bad repo does not abort the whole batch.
                logger.warning("Skipping task %s: %s", task.task_id, exc)
                _log_progress(progress, f"  SKIPPED {task.task_id}: {exc}")
                failures.append((task.task_id, str(exc)))
                if progress is not None:
                    progress.finish_task()
                continue
            reports.append(report)

            result_path = output_dir / f"{task.task_id}.json"
            result_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            _log_progress(progress, f"  → Wrote {result_path}")
            if progress is not None:
                progress.finish_task()

        if failures:
            _log_progress(progress, f"{len(failures)} task(s) skipped due to clone failures:")
            for task_id, detail in failures:
                _log_progress(progress, f"  - {task_id}: {detail}")

        return reports
    finally:
        for path in cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)


def load_selected_tasks(
    tasks_dir: Path,
    *,
    task_filter: str | None = None,
    self_only: bool = False,
) -> list[BenchmarkTask]:
    """Load benchmark tasks after applying run filters."""
    tasks = load_tasks(tasks_dir)
    if self_only:
        tasks = [t for t in tasks if t.repo == "."]
        if not tasks:
            raise ValueError(f"No self-only tasks found in {tasks_dir}")
    if task_filter:
        tasks = [t for t in tasks if t.task_id == task_filter]
        if not tasks:
            raise ValueError(f"No task found with id '{task_filter}'")
    return tasks


def repo_path_for_task(
    task: BenchmarkTask,
    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path],
    cleanup_paths: list[Path],
) -> Path:
    """Return a stable repo path for a task within one benchmark run."""
    if task.repo == ".":
        return Path.cwd()

    key = (task.repo, task.commit, tuple(task.include_paths))
    cached = repo_cache.get(key)
    if cached is not None:
        return cached

    clone_path, needs_cleanup = clone_at_commit(task.repo, task.commit)
    if needs_cleanup:
        cleanup_paths.append(clone_path)
    repo_path = clone_path
    if task.include_paths:
        repo_path = _slice_repo_for_task(clone_path, task)
        cleanup_paths.append(repo_path)
    repo_cache[key] = repo_path
    return repo_path


def _slice_repo_for_task(repo_path: Path, task: BenchmarkTask) -> Path:
    """Copy only task-relevant repository paths into a temporary benchmark corpus."""
    target = Path(tempfile.mkdtemp(prefix="archex-bench-slice-"))
    for rel_path in task.include_paths:
        source = repo_path / rel_path
        if not source.exists():
            shutil.rmtree(target, ignore_errors=True)
            raise BenchmarkCloneError(
                f"include path {rel_path!r} does not exist in {task.repo}@{task.commit}"
            )
        _copy_benchmark_path(source, target / rel_path)
    init = _run_git(["git", "init", "--quiet"], cwd=target, timeout=30)
    if init.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        detail = init.stderr.strip() or "unknown git error"
        raise BenchmarkCloneError(f"git init failed for scoped benchmark repo: {detail}")
    return target


def _copy_benchmark_path(source: Path, target: Path) -> None:
    if source.is_dir():
        for child in source.rglob("*"):
            if ".git" in child.parts:
                continue
            relative = child.relative_to(source)
            destination = target / relative
            if child.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif child.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, destination)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _log_progress(progress: BenchmarkProgress | None, message: str) -> None:
    if progress is not None:
        progress.console.log(message)
    else:
        print(message, file=sys.stderr)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
