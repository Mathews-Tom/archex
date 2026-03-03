"""Delta indexing: detect file changes between commits and surgically update the index."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from archex.exceptions import DeltaIndexError
from archex.models import (
    ChangeStatus,
    DeltaManifest,
    DeltaMeta,
    FileChange,
    IndexConfig,
)

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph
    from archex.index.store import IndexStore
    from archex.models import CodeChunk, Config, Edge

logger = logging.getLogger(__name__)


def _is_commit_reachable(repo_path: Path, commit: str) -> bool:
    """Check if a commit exists in the local git history."""
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "commit"
    except (subprocess.TimeoutExpired, OSError):
        return False


def compute_delta(
    repo_path: Path,
    base_commit: str,
    current_commit: str,
) -> DeltaManifest:
    """Compute the file-level delta between two commits.

    Args:
        repo_path: Path to the git repository.
        base_commit: The commit hash the cache was built from.
        current_commit: The current HEAD commit hash.

    Returns:
        DeltaManifest with classified file changes.

    Raises:
        DeltaIndexError: If git diff fails (e.g., shallow clone, invalid commit).
    """
    if not _is_commit_reachable(repo_path, base_commit):
        raise DeltaIndexError(
            f"Base commit {base_commit[:12]} not reachable in repository (possible shallow clone)"
        )

    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "-M", f"{base_commit}..{current_commit}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeltaIndexError("git diff timed out after 30s") from exc
    except OSError as exc:
        raise DeltaIndexError(f"git diff failed: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise DeltaIndexError(f"git diff failed: {stderr}")

    changes: list[FileChange] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        status_code = parts[0]
        if status_code == "M":
            changes.append(FileChange(path=parts[1], status=ChangeStatus.MODIFIED))
        elif status_code == "A":
            changes.append(FileChange(path=parts[1], status=ChangeStatus.ADDED))
        elif status_code == "D":
            changes.append(FileChange(path=parts[1], status=ChangeStatus.DELETED))
        elif status_code.startswith("R") and len(parts) >= 3:
            changes.append(
                FileChange(
                    path=parts[2],
                    status=ChangeStatus.RENAMED,
                    old_path=parts[1],
                )
            )

    return DeltaManifest(
        base_commit=base_commit,
        current_commit=current_commit,
        changes=changes,
    )


def apply_delta(
    store: IndexStore,
    graph: DependencyGraph,
    manifest: DeltaManifest,
    repo_path: Path,
    config: Config,
) -> DeltaMeta:
    """Apply a delta manifest to an existing index store and graph.

    Surgically updates only the affected chunks, edges, and FTS entries
    instead of rebuilding the entire index.

    Args:
        store: The cached IndexStore to update in-place.
        graph: The DependencyGraph to update in-place.
        manifest: The computed delta manifest.
        repo_path: Path to the current repo checkout.
        config: Pipeline configuration.

    Returns:
        DeltaMeta with delta operation metrics.
    """
    from archex.acquire import discover_files
    from archex.index.bm25 import BM25Index
    from archex.index.chunker import ASTChunker
    from archex.index.graph import DependencyGraph as _DepGraph
    from archex.parse import (
        TreeSitterEngine,
        build_file_map,
        extract_symbols,
        parse_imports,
        resolve_imports,
    )
    from archex.parse.adapters import default_adapter_registry

    t_start = time.perf_counter()

    # Ensure chunks_fts table exists before any delete operations.
    # BM25Index.__init__ creates the table if absent (idempotent via CREATE IF NOT EXISTS).
    BM25Index(store)

    # 1. Handle renames
    for old_path, new_path in manifest.renamed_files:
        store.update_file_paths(old_path, new_path)
        logger.info("Renamed %s -> %s", old_path, new_path)

    # 2. Handle deletions
    deleted = manifest.deleted_files
    if deleted:
        store.delete_chunks_for_files(deleted)
        store.delete_edges_for_files(deleted)
        logger.info("Deleted %d files from index", len(deleted))

    # 3. Re-parse modified + added files
    reprocess = manifest.modified_files + manifest.added_files
    reprocess_set = set(reprocess)

    new_chunks: list[CodeChunk] = []
    new_edges: list[Edge] = []

    if reprocess:
        all_files = discover_files(
            repo_path,
            languages=config.languages,
            max_file_size=config.max_file_size,
        )
        changed_files = [f for f in all_files if f.path in reprocess_set]

        if changed_files:
            engine = TreeSitterEngine()
            adapters = default_adapter_registry.build_all()

            parsed_files = extract_symbols(changed_files, engine, adapters)
            import_map = parse_imports(changed_files, engine, adapters)
            file_map = build_file_map(changed_files)
            file_languages = {f.path: f.language for f in changed_files}
            resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

            index_config = IndexConfig()
            chunker = ASTChunker(config=index_config)
            sources: dict[str, bytes] = {}
            for f in changed_files:
                try:
                    sources[f.path] = Path(f.absolute_path).read_bytes()
                except OSError:
                    continue
            new_chunks = chunker.chunk_files(parsed_files, sources)

            temp_graph = _DepGraph.from_parsed_files(parsed_files, resolved_map)
            new_edges = temp_graph.file_edges()

            logger.info(
                "Re-parsed %d files: %d chunks, %d edges",
                len(changed_files),
                len(new_chunks),
                len(new_edges),
            )

        remove_paths = list(set(manifest.modified_files))
        if remove_paths or new_chunks:
            store.delete_and_insert_for_files(remove_paths, new_chunks, new_edges)

    # 4. Update dependency graph
    removed_graph_paths = set(manifest.modified_files + manifest.deleted_files)
    for old_path, _ in manifest.renamed_files:
        removed_graph_paths.add(old_path)
    graph.update_files(removed_graph_paths, new_edges)

    # 5. Rebuild BM25 from updated store
    all_chunks = store.get_chunks()
    bm25 = BM25Index(store)
    bm25.build(all_chunks)

    # 6. Update metadata
    store.set_metadata("commit_hash", manifest.current_commit)
    store.set_metadata("delta_applied", "true")
    file_meta = store.get_file_metadata()
    store.set_metadata("file_count", str(len(file_meta)))

    delta_time_ms = (time.perf_counter() - t_start) * 1000

    total_files = len(file_meta)
    changed_count = (
        len(manifest.modified_files)
        + len(manifest.added_files)
        + len(manifest.deleted_files)
        + len(manifest.renamed_files)
    )

    logger.info("Delta applied in %.0fms (%d files changed)", delta_time_ms, changed_count)

    return DeltaMeta(
        base_commit=manifest.base_commit,
        current_commit=manifest.current_commit,
        files_modified=len(manifest.modified_files),
        files_added=len(manifest.added_files),
        files_deleted=len(manifest.deleted_files),
        files_renamed=len(manifest.renamed_files),
        files_unchanged=max(0, total_files - changed_count + len(manifest.deleted_files)),
        delta_time_ms=round(delta_time_ms, 1),
        full_reindex_avoided=True,
    )


def compute_mtime_delta(
    repo_path: Path,
    store: IndexStore,
    last_indexed_at: float,
) -> DeltaManifest:
    """Detect changes using file mtime for non-git repos.

    Args:
        repo_path: Path to the local directory.
        store: Existing IndexStore with previously indexed data.
        last_indexed_at: Unix timestamp of last index operation.

    Returns:
        DeltaManifest with changes classified by mtime comparison.
    """
    from archex.acquire import discover_files

    file_meta = store.get_file_metadata()
    indexed_paths = {str(m["file_path"]) for m in file_meta}

    current_files = discover_files(repo_path)
    current_paths = {f.path for f in current_files}

    changes: list[FileChange] = []

    for path in current_paths - indexed_paths:
        changes.append(FileChange(path=path, status=ChangeStatus.ADDED))

    for path in indexed_paths - current_paths:
        changes.append(FileChange(path=path, status=ChangeStatus.DELETED))

    for f in current_files:
        if f.path in indexed_paths:
            abs_path = Path(f.absolute_path)
            try:
                mtime = abs_path.stat().st_mtime
                if mtime > last_indexed_at:
                    changes.append(FileChange(path=f.path, status=ChangeStatus.MODIFIED))
            except OSError:
                continue

    return DeltaManifest(
        base_commit="mtime",
        current_commit="mtime",
        changes=changes,
    )
