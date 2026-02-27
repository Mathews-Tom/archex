"""Top-level public API: analyze, query, and compare entry points."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from archex.acquire import clone_repo, discover_files, open_local
from archex.analyze.decisions import infer_decisions
from archex.analyze.interfaces import extract_interfaces
from archex.analyze.modules import detect_modules
from archex.analyze.patterns import detect_patterns
from archex.cache import CacheManager
from archex.index.bm25 import BM25Index
from archex.index.chunker import ASTChunker
from archex.index.graph import DependencyGraph
from archex.index.store import IndexStore
from archex.models import ArchProfile, Config, ContextBundle, IndexConfig, RepoMetadata, RepoSource
from archex.parse import (
    LanguageAdapter,
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.parse.adapters.python import PythonAdapter
from archex.parse.adapters.typescript import TypeScriptAdapter
from archex.providers.base import get_provider
from archex.serve.context import assemble_context
from archex.serve.profile import build_profile

if TYPE_CHECKING:
    from archex.models import ComparisonResult


def _acquire(source: RepoSource) -> tuple[Path, str | None, str | None]:
    """Resolve a RepoSource to a local path."""
    if source.url and (source.url.startswith("http://") or source.url.startswith("https://")):
        target_dir = tempfile.mkdtemp()
        return clone_repo(source.url, target_dir), source.url, None
    if source.local_path is not None:
        return open_local(source.local_path), None, source.local_path
    raise ValueError("RepoSource must have a url or local_path")


def _build_adapters() -> dict[str, LanguageAdapter]:
    """Build the registry of language adapters."""
    return {
        "python": PythonAdapter(),
        "typescript": TypeScriptAdapter(),
    }


def analyze(
    source: RepoSource,
    config: Config | None = None,
    index_config: IndexConfig | None = None,
) -> ArchProfile:
    """Acquire, parse, index, and analyze a repository.

    Runs the full pipeline: acquire → parse → graph → modules → patterns → interfaces
    → optional LLM enrichment → profile assembly.
    """
    if config is None:
        config = Config()

    repo_path, url, local_path = _acquire(source)
    files = discover_files(repo_path, languages=config.languages)

    engine = TreeSitterEngine()
    adapters = _build_adapters()

    parsed_files = extract_symbols(files, engine, adapters)
    import_map = parse_imports(files, engine, adapters)
    file_map = build_file_map(files)
    file_languages = {f.path: f.language for f in files}
    resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

    graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

    # Intelligence: module detection, pattern recognition, interface extraction
    modules = detect_modules(graph, parsed_files)
    patterns = detect_patterns(parsed_files, graph, modules)
    interfaces = extract_interfaces(parsed_files, graph)

    # Optional LLM enrichment
    provider = None
    if config.enrich and config.provider:
        provider = get_provider(config.provider, config.provider_config)

    decisions = infer_decisions(patterns, modules, interfaces, provider=provider)

    lang_counts: dict[str, int] = {}
    for f in files:
        lang_counts[f.language] = lang_counts.get(f.language, 0) + 1

    total_lines = sum(pf.lines for pf in parsed_files)

    repo_metadata = RepoMetadata(
        url=url,
        local_path=local_path,
        languages=lang_counts,
        total_files=len(files),
        total_lines=total_lines,
    )

    return build_profile(
        repo_metadata,
        parsed_files,
        graph,
        modules=modules,
        patterns=patterns,
        interfaces=interfaces,
        decisions=decisions,
    )


def query(
    source: RepoSource,
    question: str,
    token_budget: int = 8192,
    config: Config | None = None,
    index_config: IndexConfig | None = None,
) -> ContextBundle:
    """Retrieve a ranked ContextBundle for a natural-language query.

    Runs the full pipeline: acquire → parse → chunk → index → search → assemble.
    Uses cached index if available.
    """
    if config is None:
        config = Config()
    if index_config is None:
        index_config = IndexConfig()

    cache = CacheManager(cache_dir=config.cache_dir)
    cache_key = cache.cache_key(source)

    repo_path, _url, _local_path = _acquire(source)
    files = discover_files(repo_path, languages=config.languages)

    engine = TreeSitterEngine()
    adapters = _build_adapters()

    parsed_files = extract_symbols(files, engine, adapters)
    import_map = parse_imports(files, engine, adapters)
    file_map = build_file_map(files)
    file_languages = {f.path: f.language for f in files}
    resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

    graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

    # Chunk files
    chunker = ASTChunker(config=index_config)
    sources: dict[str, bytes] = {}
    for f in files:
        try:
            sources[f.path] = Path(f.absolute_path).read_bytes()
        except OSError:
            continue
    all_chunks = chunker.chunk_files(parsed_files, sources)

    # Build or load index
    cached_db = cache.get(cache_key) if config.cache else None
    if cached_db is not None:
        store = IndexStore(cached_db)
        bm25 = BM25Index(store)
        # Rebuild FTS from stored chunks (FTS data isn't persisted across copies)
        cached_chunks = store.get_chunks()
        if cached_chunks:
            bm25.build(cached_chunks)
    else:
        db_path = Path(tempfile.mkdtemp()) / "index.db"
        store = IndexStore(db_path)
        store.insert_chunks(all_chunks)
        bm25 = BM25Index(store)
        bm25.build(all_chunks)
        if config.cache:
            # Checkpoint WAL to ensure all data is in the main DB file before copy
            store.conn.execute("PRAGMA wal_checkpoint(FULL)")
            cache.put(cache_key, db_path)

    # Search and assemble
    search_results = bm25.search(question, top_k=50)
    bundle = assemble_context(
        search_results=search_results,
        graph=graph,
        all_chunks=all_chunks,
        question=question,
        token_budget=token_budget,
    )

    store.close()
    return bundle


def compare(
    source_a: RepoSource,
    source_b: RepoSource,
    config: Config | None = None,
) -> ComparisonResult:
    """Analyze two repositories and return a ComparisonResult."""
    # TODO: Implement in Phase 4
    raise NotImplementedError
