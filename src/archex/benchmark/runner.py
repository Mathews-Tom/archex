"""Benchmark execution engine: runs tasks across strategies and collects results."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, BenchmarkTask, Strategy
from archex.benchmark.strategies import default_strategy_registry
from archex.exceptions import ArchexIndexError, BenchmarkCloneError

logger = logging.getLogger(__name__)

DEFAULT_STRATEGIES: list[Strategy] = [
    Strategy.RAW_FILES,
    Strategy.RAW_GREPPED,
    Strategy.ARCHEX_QUERY,
]

AVAILABLE_STRATEGIES: list[Strategy] = [
    *DEFAULT_STRATEGIES,
    Strategy.ARCHEX_QUERY_FUSION,
    Strategy.ARCHEX_QUERY_FUSION_RERANK,
    Strategy.CROSS_LAYER_FUSION,
]

_VECTOR_STRATEGIES: frozenset[Strategy] = frozenset(
    {
        Strategy.ARCHEX_QUERY_VECTOR,
        Strategy.SURROGATE_VECTOR,
        Strategy.ARCHEX_QUERY_FUSION,
        Strategy.ARCHEX_QUERY_FUSION_RERANK,
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


def _warm_repo_index(task: BenchmarkTask, repo_path: Path) -> None:
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
    from archex.models import Config, IndexConfig, RepoSource

    source = RepoSource(local_path=str(repo_path))
    config = Config(cache=True, languages=task.languages)
    index_config = IndexConfig(vector=True, embedder="fastembed")
    query(
        source,
        task.question,
        token_budget=task.token_budget,
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

    try:
        if _check_vector_available() and any(s in _VECTOR_STRATEGIES for s in strategies):
            print(f"  warming vector index for {task.task_id}...", file=sys.stderr, flush=True)
            _warm_repo_index(task, repo_path)

        results: list[BenchmarkResult] = []
        with click.progressbar(
            strategies,
            label=f"  {task.task_id}",
            item_show_func=lambda s: s.value if s is not None else "",
            file=sys.stderr,
        ) as bar:
            for strategy in bar:
                runner = default_strategy_registry.get(strategy)
                if runner is None:
                    logger.warning("No runner for strategy %s, skipping", strategy)
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

        return BenchmarkReport(
            task_id=task.task_id,
            repo=task.repo,
            question=task.question,
            results=results,
            baseline_tokens=baseline_tokens,
        )
    finally:
        if needs_cleanup:
            shutil.rmtree(repo_path, ignore_errors=True)


def run_all(
    tasks_dir: Path,
    output_dir: Path,
    strategies: list[Strategy] | None = None,
    task_filter: str | None = None,
) -> list[BenchmarkReport]:
    """Load all tasks, run benchmarks, write results to output_dir."""
    tasks = load_tasks(tasks_dir)
    if task_filter:
        tasks = [t for t in tasks if t.task_id == task_filter]
        if not tasks:
            raise ValueError(f"No task found with id '{task_filter}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BenchmarkReport] = []
    failures: list[tuple[str, str]] = []

    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.task_id} ({task.repo})", file=sys.stderr)
        task_repo_path: Path | None = None
        if task.repo == ".":
            task_repo_path = Path.cwd()
        try:
            report = run_benchmark(task, strategies=strategies, repo_path=task_repo_path)
        except BenchmarkCloneError as exc:
            # Isolate per-task clone failures (network, rate limit, bad ref) so one
            # bad repo does not abort the whole batch.
            logger.warning("Skipping task %s: %s", task.task_id, exc)
            print(f"  SKIPPED {task.task_id}: {exc}", file=sys.stderr)
            failures.append((task.task_id, str(exc)))
            continue
        reports.append(report)

        result_path = output_dir / f"{task.task_id}.json"
        result_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"  → Wrote {result_path}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} task(s) skipped due to clone failures:", file=sys.stderr)
        for task_id, detail in failures:
            print(f"  - {task_id}: {detail}", file=sys.stderr)

    return reports
