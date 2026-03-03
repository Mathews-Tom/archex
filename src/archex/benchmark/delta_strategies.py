"""Delta benchmark strategies: run delta indexing vs full re-index and verify correctness."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from archex.benchmark.models import DeltaBenchmarkResult, DeltaStrategy

if TYPE_CHECKING:
    from archex.benchmark.models import DeltaBenchmarkTask
    from archex.index.store import IndexStore


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _git(repo_path: Path, *args: str) -> str:
    """Run a git command in repo_path and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _checkout(repo_path: Path, commit: str) -> None:
    """Checkout a specific commit in the repo."""
    _git(repo_path, "checkout", "--quiet", commit)


def prepare_repo(repo_slug: str, base_commit: str) -> tuple[Path, bool]:
    """Prepare a repo for benchmarking. Returns (repo_path, needs_cleanup).

    For self-referential repos (repo="."), copies to a temp dir to avoid
    mutating the working tree. For remote repos, clones from GitHub.
    """
    if repo_slug == ".":
        target = Path(tempfile.mkdtemp(prefix="archex-delta-bench-"))
        shutil.copytree(Path.cwd(), target, dirs_exist_ok=True)
        _checkout(target, base_commit)
        return target, True

    url = f"https://github.com/{repo_slug}.git"
    target = Path(tempfile.mkdtemp(prefix="archex-delta-bench-"))
    subprocess.run(
        ["git", "clone", "--quiet", url, str(target)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    _checkout(target, base_commit)
    return target, True


def _collect_store_state(
    store: IndexStore,
) -> tuple[set[tuple[str, str, int, int]], set[tuple[str, str, str]]]:
    """Extract chunk content signatures and edge tuples from an IndexStore.

    Chunk IDs contain UUIDs and differ between indexing runs, so we compare
    by (file_path, symbol_name, start_line, end_line) tuples instead.

    Returns (chunk_signature_set, edge_tuple_set) where each edge tuple is
    (source, target, kind).
    """
    chunks = store.get_chunks()
    chunk_sigs = {(c.file_path, c.symbol_name or "", c.start_line, c.end_line) for c in chunks}

    edges = store.get_edges()
    edge_tuples = {(e.source, e.target, e.kind.value) for e in edges}

    return chunk_sigs, edge_tuples


def run_delta_benchmark(
    task: DeltaBenchmarkTask,
    repo_path: Path,
) -> DeltaBenchmarkResult:
    """Run a delta benchmark: delta index vs full re-index, measure speedup and correctness.

    Args:
        task: The delta benchmark task definition.
        repo_path: Path to a local clone (will be checked out to different commits).

    Returns:
        DeltaBenchmarkResult with timing, correctness, and chunk/edge metrics.
    """
    from archex.api import _full_index  # pyright: ignore[reportPrivateUsage]
    from archex.cache import CacheManager
    from archex.index.delta import apply_delta, compute_delta
    from archex.index.graph import DependencyGraph
    from archex.models import Config, PipelineTiming, RepoSource

    config = Config(cache=False)
    cache = CacheManager(cache_dir=str(Path(tempfile.mkdtemp(prefix="archex-delta-cache-"))))

    # 1. Full index at base_commit
    _checkout(repo_path, task.base_commit)
    source = RepoSource(local_path=str(repo_path))
    cache_key = cache.cache_key(source)
    base_store = _full_index(source, config, cache, cache_key, timing=None)

    # 2. Delta index: checkout delta_commit, compute + apply delta
    _checkout(repo_path, task.delta_commit)
    t_delta_start = time.perf_counter()
    manifest = compute_delta(repo_path, task.base_commit, task.delta_commit)
    graph = DependencyGraph.from_edges(base_store.get_edges())
    delta_meta = apply_delta(base_store, graph, manifest, repo_path, config)
    delta_time_ms = (time.perf_counter() - t_delta_start) * 1000

    # Collect delta state
    delta_chunks, delta_edges = _collect_store_state(base_store)
    base_store.close()

    # 3. Full re-index at delta_commit for ground truth
    t_full_start = time.perf_counter()
    timing_full = PipelineTiming()
    full_store = _full_index(source, config, cache, f"{cache_key}_full", timing=timing_full)
    full_reindex_time_ms = (time.perf_counter() - t_full_start) * 1000

    full_chunks, full_edges = _collect_store_state(full_store)
    total_files = len(full_store.get_file_metadata())
    full_store.close()

    # 4. Correctness: chunk signatures must match (primary retrieval data).
    # Edge equivalence is tracked separately — delta only re-resolves imports for
    # changed files, so edges from unchanged files referencing changed files may differ.
    correctness = delta_chunks == full_chunks

    # 5. Compute metrics
    delta_files = len(manifest.changes)
    delta_pct = (delta_files / total_files * 100) if total_files > 0 else 0.0
    speedup = full_reindex_time_ms / delta_time_ms if delta_time_ms > 0 else 0.0

    chunks_in_common = delta_chunks & full_chunks
    chunks_only_in_full = full_chunks - delta_chunks
    edges_changed = len(delta_edges.symmetric_difference(full_edges))

    return DeltaBenchmarkResult(
        task_id=task.task_id,
        strategy=DeltaStrategy.DELTA_INDEX,
        delta_files=delta_files,
        total_files=total_files,
        delta_pct=round(delta_pct, 1),
        delta_time_ms=round(delta_time_ms, 1),
        full_reindex_time_ms=round(full_reindex_time_ms, 1),
        speedup_factor=round(speedup, 2),
        correctness=correctness,
        chunks_updated=len(chunks_only_in_full) + len(delta_chunks - full_chunks),
        chunks_unchanged=len(chunks_in_common),
        edges_updated=edges_changed,
        timestamp=_now_iso(),
        delta_meta=delta_meta,
    )
