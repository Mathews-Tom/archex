"""Delta indexing: detect file changes between commits and surgically update the index."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from archex.acquire.discovery import EXTENSION_MAP
from archex.exceptions import DeltaIndexError
from archex.models import (
    ChangeStatus,
    DeltaManifest,
    DeltaMeta,
    Edge,
    EdgeKind,
    FileChange,
    IndexConfig,
)
from archex.pipeline.service import build_chunk_surrogates
from archex.reporting import count_tokens

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph
    from archex.index.store import IndexStore
    from archex.models import CodeChunk, Config, DiscoveredFile, ImportStatement

logger = logging.getLogger(__name__)


def _is_source_path(path: str, languages: list[str] | None) -> bool:
    language = EXTENSION_MAP.get(Path(path).suffix.lower())
    if language is None:
        return False
    return languages is None or language in languages


def _parse_porcelain_path(line: str) -> tuple[str, str] | None:
    if len(line) < 4:
        return None
    status = line[:2]
    path = line[3:]
    if " -> " in path:
        path = path.rsplit(" -> ", maxsplit=1)[1]
    return status, path


def compute_working_tree_signature(repo_path: Path, config: Config) -> str:
    """Return a content signature for changed source files in a git working tree."""
    if not (repo_path / ".git").exists():
        return ""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeltaIndexError("git status timed out after 30s") from exc
    except OSError as exc:
        raise DeltaIndexError(f"git status failed: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise DeltaIndexError(f"git status failed: {stderr}")

    changed: list[dict[str, str | int]] = []
    for line in result.stdout.splitlines():
        parsed = _parse_porcelain_path(line)
        if parsed is None:
            continue
        status, path = parsed
        if not _is_source_path(path, config.languages):
            continue

        item: dict[str, str | int] = {"path": path, "status": status}
        file_path = repo_path / path
        if file_path.is_file():
            stat = file_path.stat()
            item["size"] = stat.st_size
            item["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        changed.append(item)

    if not changed:
        return "clean"
    payload = json.dumps(sorted(changed, key=lambda item: str(item["path"])), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_name_status_line(line: str) -> FileChange | None:
    parts = line.split("\t")
    if len(parts) < 2:
        return None

    status_code = parts[0]
    if status_code == "M":
        return FileChange(path=parts[1], status=ChangeStatus.MODIFIED)
    if status_code == "A":
        return FileChange(path=parts[1], status=ChangeStatus.ADDED)
    if status_code == "D":
        return FileChange(path=parts[1], status=ChangeStatus.DELETED)
    if status_code.startswith("R") and len(parts) >= 3:
        return FileChange(
            path=parts[2],
            status=ChangeStatus.RENAMED,
            old_path=parts[1],
        )
    return None


def _build_import_edges(resolved_map: dict[str, list[ImportStatement]]) -> list[Edge]:
    return [
        Edge(
            source=file_path,
            target=imp.resolved_path,
            kind=EdgeKind.IMPORTS,
            location=f"{file_path}:{imp.line}",
        )
        for file_path, imports in resolved_map.items()
        for imp in imports
        if imp.resolved_path is not None
    ]


def _changed_sources(changed_files: list[DiscoveredFile]) -> dict[str, bytes]:
    sources: dict[str, bytes] = {}
    for discovered_file in changed_files:
        try:
            sources[discovered_file.path] = Path(discovered_file.absolute_path).read_bytes()
        except OSError:
            continue
    return sources


def compute_file_states(
    repo_path: Path,
    files: list[DiscoveredFile],
    *,
    previous: dict[str, dict[str, int | str]] | None = None,
) -> dict[str, dict[str, int | str]]:
    """Return file size/mtime/hash state, hashing only paths whose stat changed."""
    previous = previous or {}
    states: dict[str, dict[str, int | str]] = {}
    for discovered_file in files:
        path = Path(discovered_file.absolute_path)
        try:
            stat = path.stat()
        except OSError:
            continue

        prior = previous.get(discovered_file.path)
        size_bytes = stat.st_size
        mtime_ns = stat.st_mtime_ns
        if (
            prior is not None
            and int(prior["size_bytes"]) == size_bytes
            and int(prior["mtime_ns"]) == mtime_ns
        ):
            states[discovered_file.path] = prior
            continue

        try:
            raw_bytes = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(raw_bytes).hexdigest()
        states[discovered_file.path] = {
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "sha256": digest,
            "token_count": count_tokens(raw_bytes.decode("utf-8", errors="replace")),
        }
    return states


def _working_tree_renames(repo_path: Path, languages: list[str] | None) -> list[tuple[str, str]]:
    if not (repo_path / ".git").exists():
        return []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--renames", "--untracked-files=all"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeltaIndexError("git status timed out after 30s") from exc
    except OSError as exc:
        raise DeltaIndexError(f"git status failed: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise DeltaIndexError(f"git status failed: {stderr}")

    renames: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if len(line) < 4 or "R" not in line[:2] or " -> " not in line:
            continue
        old_path, new_path = line[3:].split(" -> ", maxsplit=1)
        if _is_source_path(old_path, languages) or _is_source_path(new_path, languages):
            renames.append((old_path, new_path))
    return renames


def compute_working_tree_delta(
    repo_path: Path,
    store: IndexStore,
    config: Config,
) -> DeltaManifest:
    """Compute a hash-confirmed delta between the persisted index and working tree."""
    from archex.acquire import discover_files

    previous_states = store.get_file_states()
    if config.languages is not None:
        previous_states = {
            path: state
            for path, state in previous_states.items()
            if _is_source_path(path, config.languages)
        }
    if not previous_states:
        indexed_at = 0.0
        try:
            indexed_at = float(store.get_metadata("indexed_at") or "0")
        except ValueError:
            indexed_at = 0.0
        return compute_mtime_delta(repo_path, store, indexed_at)

    current_files = discover_files(repo_path,
    languages=config.languages,
    max_file_size=config.max_file_size,).files
    current_states = compute_file_states(repo_path, current_files, previous=previous_states)
    previous_paths = set(previous_states)
    current_paths = set(current_states)

    deleted_paths = previous_paths - current_paths
    added_paths = current_paths - previous_paths
    changes: list[FileChange] = []

    for old_path, new_path in _working_tree_renames(repo_path, config.languages):
        if old_path not in deleted_paths or new_path not in added_paths:
            continue
        changes.append(FileChange(path=new_path, old_path=old_path, status=ChangeStatus.RENAMED))
        deleted_paths.remove(old_path)
        added_paths.remove(new_path)
        previous_hash = str(previous_states[old_path]["sha256"])
        current_hash = str(current_states[new_path]["sha256"])
        if previous_hash != current_hash:
            changes.append(FileChange(path=new_path, status=ChangeStatus.MODIFIED))

    for path in sorted(current_paths & previous_paths):
        if str(current_states[path]["sha256"]) != str(previous_states[path]["sha256"]):
            changes.append(FileChange(path=path, status=ChangeStatus.MODIFIED))

    changes.extend(FileChange(path=path, status=ChangeStatus.ADDED) for path in sorted(added_paths))
    changes.extend(
        FileChange(path=path, status=ChangeStatus.DELETED) for path in sorted(deleted_paths)
    )
    return DeltaManifest(base_commit="worktree", current_commit="worktree", changes=changes)


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
        change = _parse_name_status_line(line)
        if change is not None:
            changes.append(change)

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
    index_config: IndexConfig | None = None,
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
    from archex.parse import (
        TreeSitterEngine,
        build_file_map,
        extract_symbols_and_imports,
        resolve_imports,
    )
    from archex.parse.adapters import default_adapter_registry
    from archex.pipeline.chunker import chunker_revision, create_chunker

    t_start = time.perf_counter()
    effective_index_config = index_config or IndexConfig()

    # Ensure chunks_fts has the current schema before any delete operations
    # below. A stale-schema migration here drops and recreates the table,
    # emptying it — capture that so step 5 knows a full rebuild (not a
    # targeted update) is required.
    bm25 = BM25Index(
        store,
        identifier_fragment_tokenization=effective_index_config.identifier_fragment_tokenization,
    )
    needs_full_bm25_rebuild = not bm25.has_data

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
        store.delete_file_states(deleted)

    # 3. Re-parse modified, added, and renamed files so chunk IDs stay path-correct.
    renamed_new_paths = [new_path for _, new_path in manifest.renamed_files]
    reprocess = manifest.modified_files + manifest.added_files + renamed_new_paths
    reprocess_set = set(reprocess)

    new_chunks: list[CodeChunk] = []
    new_edges: list[Edge] = []
    new_surrogates = []

    if reprocess:
        all_files = discover_files(repo_path,
        languages=config.languages,
        max_file_size=config.max_file_size,).files
        changed_files = [f for f in all_files if f.path in reprocess_set]

        if changed_files:
            engine = TreeSitterEngine()
            adapters = default_adapter_registry.build_all()

            extraction = extract_symbols_and_imports(changed_files, engine, adapters)
            parsed_files = extraction.parsed_files
            file_map = build_file_map(all_files)
            file_languages = {f.path: f.language for f in all_files}
            resolved_map = resolve_imports(
                extraction.imports_by_path, file_map, adapters, file_languages
            )

            chunker = create_chunker(effective_index_config)
            sources = _changed_sources(changed_files)
            new_chunks = chunker.chunk_files(parsed_files, sources)
            new_surrogates = build_chunk_surrogates(
                new_chunks,
                version=effective_index_config.surrogate_version,
            )

            new_edges = _build_import_edges(resolved_map)
            store.upsert_file_states(compute_file_states(repo_path, changed_files))

            logger.info(
                "Re-parsed %d files: %d chunks, %d edges",
                len(changed_files),
                len(new_chunks),
                len(new_edges),
            )

        remove_paths = list(set(manifest.modified_files + renamed_new_paths))
        if remove_paths or new_chunks:
            store.delete_and_insert_for_files(
                remove_paths,
                new_chunks,
                new_edges,
                new_surrogates,
            )

    # 4. Update dependency graph
    removed_graph_paths = set(manifest.modified_files + manifest.deleted_files)
    for old_path, _ in manifest.renamed_files:
        removed_graph_paths.add(old_path)
    graph.update_files(removed_graph_paths, new_edges)

    # 5. Targeted BM25 update: insert only the newly-parsed chunks. Store
    # deletions in steps 1-3 already scoped stale FTS rows to the delta's
    # changed files, so no full DROP+rebuild is needed for the common case.
    if needs_full_bm25_rebuild:
        bm25.build(store.iter_chunks())
    elif new_chunks:
        bm25.update(new_chunks)

    # 6. Update metadata
    store.set_metadata("chunker", effective_index_config.chunker)
    store.set_metadata("chunker_revision", chunker_revision(effective_index_config.chunker))
    store.set_metadata("repo_total_tokens", str(store.get_chunk_token_total()))
    store.set_metadata("chunk_count", str(store.get_chunk_count()))
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

    current_files = discover_files(repo_path).files
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
