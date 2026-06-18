"""Explicit indexing command for repo-local archex project workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from archex.api import index_repository
from archex.cache import CacheManager
from archex.config import load_config, load_index_config, persist_project_index_settings
from archex.exceptions import ArchexError
from archex.models import PipelineTiming, RepoSource
from archex.project import uses_project_cache_layout


def _language_counts(file_metadata: list[dict[str, str | int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in file_metadata:
        language = str(item["language"])
        counts[language] = counts.get(language, 0) + 1
    return dict(sorted(counts.items()))


@click.command("index")
@click.argument("source", required=False, default=".")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
@click.option("--splade", is_flag=True, default=False, help="Build the opt-in SPLADE index.")
@click.option(
    "--module-prefilter",
    is_flag=True,
    default=False,
    help="Build opt-in module responsibility summaries.",
)
@click.option(
    "--allow-remote-code",
    is_flag=True,
    default=False,
    help="Allow explicitly selected pinned model paths that require Hugging Face remote code.",
)
@click.option(
    "--quantize-vectors/--no-quantize-vectors",
    default=None,
    help="Build vector indexes with TurboQuant compression.",
)
@click.option(
    "--quantize-bits",
    default=None,
    type=click.Choice(["2", "4"]),
    help="TurboQuant bit-width for vector indexes.",
)
def index_cmd(
    source: str,
    output_format: str,
    splade: bool,
    module_prefilter: bool,
    allow_remote_code: bool,
    quantize_vectors: bool | None,
    quantize_bits: str | None,
) -> None:
    """Build or refresh the index for SOURCE without running a query."""
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

    try:
        store = index_repository(
            repo_source,
            config=config,
            timing=timing,
            index_config=index_config,
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        file_metadata = store.get_file_metadata()
        languages = _language_counts(file_metadata)
        cached_path = cache.get(cache_key) if cache is not None and cache_key is not None else None
        duration_ms = int((time.perf_counter() - started) * 1000)
        summary = {
            "repo_root": str(repo_root),
            "index_path": str(cached_path or store.db_path),
            "commit_hash": store.get_metadata("commit_hash") or "",
            "strategy": timing.strategy or "full",
            "files_indexed": store.get_file_count(),
            "chunks_indexed": store.get_chunk_count(),
            "languages": languages,
            "duration_ms": duration_ms,
            "embedding_cache_hits": int(store.get_metadata("embedding_cache_hits") or "0"),
            "embedding_cache_misses": int(store.get_metadata("embedding_cache_misses") or "0"),
        }
    finally:
        store.close()

    if output_format == "json":
        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    click.echo(f"Indexed repository: {summary['repo_root']}")
    click.echo(f"Index path:         {summary['index_path']}")
    click.echo(f"Commit:             {summary['commit_hash'] or 'none'}")
    click.echo(f"Strategy:           {summary['strategy']}")
    click.echo(f"Files indexed:      {summary['files_indexed']}")
    click.echo(f"Chunks indexed:     {summary['chunks_indexed']}")
    if languages:
        language_summary = ", ".join(f"{language}={count}" for language, count in languages.items())
    else:
        language_summary = "none"
    click.echo(f"Languages:          {language_summary}")
    click.echo(
        "Embedding cache:    "
        f"hits={summary['embedding_cache_hits']}, misses={summary['embedding_cache_misses']}"
    )
    click.echo(f"Duration:           {summary['duration_ms']} ms")
