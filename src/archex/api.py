"""Top-level public API: analyze, query, and compare entry points."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from archex.acquire import clone_repo, discover_files, open_local
from archex.analyze.decisions import infer_decisions
from archex.analyze.interfaces import extract_interfaces
from archex.analyze.modules import detect_modules
from archex.analyze.patterns import detect_patterns
from archex.cache import CacheManager
from archex.index.bm25 import BM25Index
from archex.index.chunker import ASTChunker, Chunker
from archex.index.graph import DependencyGraph
from archex.index.store import IndexStore
from archex.models import (
    ArchProfile,
    Config,
    ContextBundle,
    IndexConfig,
    RepoMetadata,
    RepoSource,
    ScoringWeights,
)
from archex.parse import (
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.parse.adapters import LanguageAdapter, default_adapter_registry
from archex.providers.base import get_provider
from archex.serve.compare import compare_repos
from archex.serve.context import assemble_context
from archex.serve.profile import build_profile

if TYPE_CHECKING:
    from collections.abc import Callable

    from archex.models import ComparisonResult

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _acquire(source: RepoSource) -> tuple[Path, str | None, str | None, Callable[[], None]]:
    """Resolve a RepoSource to a local path, returning a cleanup callable."""
    if source.url and (source.url.startswith("http://") or source.url.startswith("https://")):
        target_dir = tempfile.mkdtemp()

        def _url_cleanup() -> None:
            shutil.rmtree(target_dir, ignore_errors=True)

        return clone_repo(source.url, target_dir), source.url, None, _url_cleanup

    if source.local_path is not None:

        def _noop() -> None:
            pass

        return open_local(source.local_path), None, source.local_path, _noop
    raise ValueError("RepoSource must have a url or local_path")


def _build_adapters() -> dict[str, LanguageAdapter]:
    """Build the registry of language adapters from the default registry."""
    return default_adapter_registry.build_all()


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

    t0 = time.perf_counter()
    repo_path, url, local_path, cleanup = _acquire(source)
    logger.info("Acquired repo %s in %.0fms", url or local_path, _elapsed_ms(t0))
    try:
        t1 = time.perf_counter()
        files = discover_files(
            repo_path, languages=config.languages, max_file_size=config.max_file_size
        )
        logger.info("Discovered %d files in %.0fms", len(files), _elapsed_ms(t1))

        engine = TreeSitterEngine()
        adapters = _build_adapters()

        t2 = time.perf_counter()
        parsed_files = extract_symbols(files, engine, adapters, parallel=config.parallel)
        import_map = parse_imports(files, engine, adapters, parallel=config.parallel)
        file_map = build_file_map(files)
        file_languages = {f.path: f.language for f in files}
        resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)
        logger.info("Parsed %d files in %.0fms", len(parsed_files), _elapsed_ms(t2))

        graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

        t3 = time.perf_counter()
        modules = detect_modules(graph, parsed_files)
        patterns = detect_patterns(parsed_files, graph, modules)
        interfaces = extract_interfaces(parsed_files, graph)
        logger.info(
            "Analysis: %d modules, %d patterns, %d interfaces in %.0fms",
            len(modules),
            len(patterns),
            len(interfaces),
            _elapsed_ms(t3),
        )

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

        profile = build_profile(
            repo_metadata,
            parsed_files,
            graph,
            modules=modules,
            patterns=patterns,
            interfaces=interfaces,
            decisions=decisions,
        )
        logger.info("analyze() completed in %.0fms", _elapsed_ms(t0))
        return profile
    finally:
        cleanup()


def query(
    source: RepoSource,
    question: str,
    token_budget: int = 8192,
    config: Config | None = None,
    index_config: IndexConfig | None = None,
    scoring_weights: ScoringWeights | None = None,
    chunker: Chunker | None = None,
) -> ContextBundle:
    """Retrieve a ranked ContextBundle for a natural-language query.

    Runs the full pipeline: acquire → parse → chunk → index → search → assemble.
    On cache hit, skips the full parse: loads chunks and graph from the cached store.
    """
    if config is None:
        config = Config()
    if index_config is None:
        index_config = IndexConfig()

    t0 = time.perf_counter()
    cache = CacheManager(cache_dir=config.cache_dir)
    cache_key = cache.cache_key(source)

    # Check cache BEFORE parsing — if cached, skip the expensive parse pipeline
    cached_db = cache.get(cache_key) if config.cache else None
    if cached_db is not None:
        logger.info("Cache hit for %s", cache_key[:12])
        store = IndexStore(cached_db)
        try:
            bm25 = BM25Index(store)
            cached_chunks = store.get_chunks()
            if cached_chunks:
                bm25.build(cached_chunks)
            stored_edges = store.get_edges()
            graph = DependencyGraph.from_edges(stored_edges)

            vector_results: list[tuple[object, float]] | None = None
            if index_config.vector:
                vec_path = cache.vector_path(cache_key)
                if vec_path.exists():
                    from archex.index.vector import VectorIndex

                    vec_idx = VectorIndex()
                    vec_idx.load(vec_path, cached_chunks)
                    embedder = _get_embedder(index_config)
                    if embedder is not None:
                        vector_results = vec_idx.search(question, embedder, top_k=50)  # type: ignore[assignment]

            search_results = bm25.search(question, top_k=50)
            bundle = assemble_context(
                search_results=search_results,
                graph=graph,
                all_chunks=cached_chunks,
                question=question,
                token_budget=token_budget,
                vector_results=vector_results,  # type: ignore[arg-type]
                scoring_weights=scoring_weights,
            )
            logger.info("query() [cached] completed in %.0fms", _elapsed_ms(t0))
            return bundle
        finally:
            store.close()

    # Cache miss — full pipeline
    logger.info("Cache miss — running full pipeline")
    t1 = time.perf_counter()
    repo_path, _url, _local_path, cleanup = _acquire(source)
    logger.info("Acquired repo in %.0fms", _elapsed_ms(t1))
    try:
        t2 = time.perf_counter()
        files = discover_files(
            repo_path, languages=config.languages, max_file_size=config.max_file_size
        )
        logger.info("Discovered %d files in %.0fms", len(files), _elapsed_ms(t2))

        engine = TreeSitterEngine()
        adapters = _build_adapters()

        t3 = time.perf_counter()
        parsed_files = extract_symbols(files, engine, adapters, parallel=config.parallel)
        import_map = parse_imports(files, engine, adapters, parallel=config.parallel)
        file_map = build_file_map(files)
        file_languages = {f.path: f.language for f in files}
        resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)
        logger.info("Parsed %d files in %.0fms", len(parsed_files), _elapsed_ms(t3))

        graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

        t4 = time.perf_counter()
        file_chunker: Chunker = chunker if chunker is not None else ASTChunker(config=index_config)
        sources: dict[str, bytes] = {}
        for f in files:
            try:
                sources[f.path] = Path(f.absolute_path).read_bytes()
            except OSError:
                continue
        all_chunks = file_chunker.chunk_files(parsed_files, sources)
        logger.info("Chunked into %d chunks in %.0fms", len(all_chunks), _elapsed_ms(t4))

        db_path = Path(tempfile.mkdtemp()) / "index.db"
        store = IndexStore(db_path)

        try:
            bm25 = BM25Index(store)
            store.insert_chunks(all_chunks)
            edges = graph.file_edges()
            store.insert_edges(edges)
            bm25.build(all_chunks)

            # Build vector index if configured
            vector_results_miss: list[tuple[object, float]] | None = None
            if index_config.vector:
                embedder = _get_embedder(index_config)
                if embedder is not None:
                    from archex.index.vector import VectorIndex

                    t5 = time.perf_counter()
                    vec_idx = VectorIndex()
                    vec_idx.build(all_chunks, embedder)  # type: ignore[arg-type]
                    vector_results_miss = vec_idx.search(question, embedder, top_k=50)  # type: ignore[assignment]
                    logger.info("Vector index built in %.0fms", _elapsed_ms(t5))

                    if config.cache:
                        vec_idx.save(cache.vector_path(cache_key))

            if config.cache:
                store.conn.execute("PRAGMA wal_checkpoint(FULL)")
                cache.put(cache_key, db_path)

            t6 = time.perf_counter()
            search_results = bm25.search(question, top_k=50)
            bundle = assemble_context(
                search_results=search_results,
                graph=graph,
                all_chunks=all_chunks,
                question=question,
                token_budget=token_budget,
                vector_results=vector_results_miss,  # type: ignore[arg-type]
                scoring_weights=scoring_weights,
            )
            logger.info("Search + assemble in %.0fms", _elapsed_ms(t6))
        finally:
            store.close()

        logger.info("query() completed in %.0fms", _elapsed_ms(t0))
        return bundle
    finally:
        cleanup()


def _get_embedder(index_config: IndexConfig) -> object | None:
    """Create an embedder from index_config, or return None if not configured."""
    if not index_config.embedder:
        return None
    if index_config.embedder == "nomic":
        from archex.index.embeddings.nomic import NomicCodeEmbedder

        return NomicCodeEmbedder()
    if index_config.embedder == "sentence_transformers":
        from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    # API embedder requires additional config — not created here
    return None


def compare(
    source_a: RepoSource,
    source_b: RepoSource,
    dimensions: list[str] | None = None,
    config: Config | None = None,
) -> ComparisonResult:
    """Analyze two repositories and return a ComparisonResult.

    Uses ThreadPoolExecutor to run both analyses concurrently.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(analyze, source_a, config)
        future_b = executor.submit(analyze, source_b, config)
        profile_a = future_a.result()
        profile_b = future_b.result()
    return compare_repos(profile_a, profile_b, dimensions)
