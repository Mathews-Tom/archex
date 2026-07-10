from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from archex.api import index_repository
from archex.cache import CacheManager
from archex.config import load_config, load_index_config, persist_project_index_settings
from archex.index.artifact import ensure_artifact_gitattributes, export_artifact
from archex.models import PipelineTiming, RepoSource
from archex.project import uses_project_cache_layout


def _language_counts(file_metadata: list[dict[str, str | int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in file_metadata:
        language = str(item["language"])
        counts[language] = counts.get(language, 0) + 1
    return dict(sorted(counts.items()))


def run_indexing_and_get_summary(
    source: str,
    splade: bool = False,
    module_prefilter: bool = False,
    allow_remote_code: bool = False,
    quantize_vectors: bool | None = None,
    quantize_bits: str | None = None,
    export_artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run the index pipeline and return a summary dictionary."""
    repo_source = RepoSource(local_path=source)
    repo_root = Path(source).expanduser().resolve()
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)

    if allow_remote_code:
        index_config = index_config.model_copy(update={"allow_remote_code": True})
    if splade:
        index_config = index_config.model_copy(update={"splade": True})
    if module_prefilter:
        index_config = index_config.model_copy(update={"module_prefilter": True})

    index_updates: dict[str, bool | int | str] = {}
    if quantize_vectors is not None:
        index_updates["quantize_vectors"] = quantize_vectors
        if quantize_vectors:
            index_updates["vector"] = True
            if index_config.embedder is None:
                index_updates["embedder"] = "jina-v2"
    if quantize_bits is not None:
        index_updates["quantize_bits"] = int(quantize_bits)

    if index_updates:
        index_config = index_config.model_copy(update=index_updates)
        persist_project_index_settings(repo_source, index_updates)

    timing = PipelineTiming()
    started = time.perf_counter()

    try:
        project_layout = uses_project_cache_layout(source, config.cache_dir)
    except ValueError:
        project_layout = False

    cache = (
        CacheManager(cache_dir=config.cache_dir, project_layout=project_layout)
        if config.cache
        else None
    )
    cache_key = cache.cache_key(repo_source) if cache is not None else None

    store = index_repository(
        repo_source,
        config=config,
        timing=timing,
        index_config=index_config,
    )

    try:
        file_metadata = store.get_file_metadata()
        languages = _language_counts(file_metadata)
        cached_path = cache.get(cache_key) if cache is not None and cache_key is not None else None
        duration_ms = int((time.perf_counter() - started) * 1000)
        # `store.db_path` lives under a scratch directory that `store.close()` removes
        # when the store has no durable cache entry (`ephemeral` is True) — reporting
        # that path here would hand back a location that no longer exists the moment
        # this function returns.
        ephemeral_untracked = cached_path is None and store.ephemeral
        index_path = None if ephemeral_untracked else (cached_path or store.db_path)

        summary = {
            "repo_root": str(repo_root),
            "index_path": str(index_path) if index_path is not None else None,
            "commit_hash": store.get_metadata("commit_hash") or "",
            "strategy": timing.strategy or "full",
            "files_indexed": store.get_file_count(),
            "chunks_indexed": store.get_chunk_count(),
            "languages": languages,
            "duration_ms": duration_ms,
            "embedding_cache_hits": int(store.get_metadata("embedding_cache_hits") or "0"),
            "embedding_cache_misses": int(store.get_metadata("embedding_cache_misses") or "0"),
        }

        if export_artifact_path is not None:
            artifact_header = export_artifact(store, export_artifact_path)
            summary["artifact_path"] = str(export_artifact_path)
            summary["artifact_index_revision"] = artifact_header.index_revision
            summary["artifact_size_bytes"] = export_artifact_path.stat().st_size
            summary["gitattributes_updated"] = ensure_artifact_gitattributes(
                repo_root, export_artifact_path
            )

        return summary
    finally:
        store.close()
