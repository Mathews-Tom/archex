"""Top-level public API: analyze, query, and compare entry points.

Service roles and pipeline phases
==================================

Each public function orchestrates a sequence of distinct service roles:

**Acquisition** — Resolve a RepoSource to a local path on disk.
  Input:  RepoSource (url or local_path)
  Output: (repo_path: Path, cleanup: Callable)
  Module: archex.acquire

**Parsing** — Discover files, extract symbols, resolve imports.
  Input:  repo_path, Config
  Output: ParseArtifacts (files, parsed_files, resolved_imports)
  Module: archex.parse, archex.pipeline.service

**Indexing** — Chunk code, build BM25/vector indices, persist to SQLite.
  Input:  ParseArtifacts, IndexConfig
  Output: IndexStore (open handle to chunk/edge/FTS5 database)
  Module: archex.index

**Retrieval** — Search the index, expand via dependency graph, score, assemble.
  Input:  IndexStore, question, token_budget, ScoringWeights
  Output: ContextBundle (ranked chunks within token budget)
  Module: archex.serve.context

**Analysis** — Detect modules, patterns, interfaces, decisions from parsed code.
  Input:  ParsedFiles, DependencyGraph
  Output: ArchProfile
  Module: archex.analyze, archex.serve.profile

**Observability** — Optional PipelineTrace records step-level timings.
  Input:  PipelineTrace (optional)
  Output: Trace populated with StepTiming records; logged at DEBUG level
  Module: archex.observe
"""

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
from archex.exceptions import ArchexIndexError, DeltaIndexError
from archex.index.bm25 import BM25Index
from archex.index.graph import DependencyGraph
from archex.index.store import IndexStore
from archex.models import (
    ArchProfile,
    ChunkSurrogate,
    CodeChunk,
    Config,
    ContextBundle,
    FileOutline,
    FileTree,
    FileTreeEntry,
    IndexConfig,
    PipelineTiming,
    RepoMetadata,
    RepoSource,
    ScoringWeights,
    SymbolKind,
    SymbolMatch,
    SymbolOutline,
    SymbolSource,
    VectorMode,
    Visibility,
)
from archex.observe import PipelineTrace, StepTiming
from archex.parse import (
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.parse.adapters import LanguageAdapter, default_adapter_registry
from archex.pipeline.chunker import ASTChunker, Chunker
from archex.pipeline.service import build_chunk_surrogates
from archex.providers.base import get_provider
from archex.serve.compare import compare_repos
from archex.serve.context import assemble_context, passthrough_context
from archex.serve.profile import build_profile

if TYPE_CHECKING:
    from collections.abc import Callable

    from archex.index.embeddings.base import Embedder
    from archex.models import ComparisonResult

# ---------------------------------------------------------------------------
# Acquisition + Indexing — shared helpers for all entry points
# ---------------------------------------------------------------------------


def _full_index(
    source: RepoSource,
    config: Config,
    cache: CacheManager,
    cache_key: str,
    timing: PipelineTiming | None,
    index_config: IndexConfig | None = None,
) -> IndexStore:
    """Run the full acquire → parse → chunk → store pipeline.

    Combines acquisition, parsing, and indexing phases into a single
    IndexStore. Used when no cache hit is available and delta indexing
    is not applicable.
    """
    t_acq = time.perf_counter()
    repo_path, _url, _local_path, cleanup, cloned_head = _acquire(source)
    if timing is not None:
        timing.acquire_ms = _elapsed_ms(t_acq)
    try:
        t_parse = time.perf_counter()
        files = discover_files(
            repo_path, languages=config.languages, max_file_size=config.max_file_size
        )
        engine = TreeSitterEngine()
        adapters = _build_adapters()
        parsed_files = extract_symbols(files, engine, adapters, parallel=config.parallel)
        import_map = parse_imports(files, engine, adapters, parallel=config.parallel)
        file_map = build_file_map(files)
        file_languages = {f.path: f.language for f in files}
        resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

        graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

        effective_index_config = index_config or IndexConfig()
        file_chunker: Chunker = ASTChunker(config=effective_index_config)
        sources: dict[str, bytes] = {}
        for f in files:
            try:
                sources[f.path] = Path(f.absolute_path).read_bytes()
            except OSError:
                continue
        all_chunks = file_chunker.chunk_files(parsed_files, sources)
        if timing is not None:
            timing.parse_ms = _elapsed_ms(t_parse)

        t_idx = time.perf_counter()
        db_path = Path(tempfile.mkdtemp()) / "index.db"
        store = IndexStore(db_path)
        store.insert_chunks(all_chunks)
        store.insert_chunk_surrogates(
            build_chunk_surrogates(
                all_chunks,
                version=effective_index_config.surrogate_version,
            )
        )
        edges = graph.file_edges()
        store.insert_edges(edges)

        if config.cache:
            commit = cloned_head or cache.git_head(source.local_path) or source.commit or ""
            identity = source.url or source.local_path or ""
            store.set_metadata("commit_hash", commit)
            store.set_metadata("source_identity", identity)
            store.set_metadata("indexed_at", str(time.time()))
            store.conn.execute("PRAGMA wal_checkpoint(FULL)")
            cache.put(
                cache_key,
                db_path,
                resolved_commit=commit,
                source_identity=identity,
            )
        if timing is not None:
            timing.index_ms = _elapsed_ms(t_idx)
            timing.strategy = "full"

        return store
    finally:
        cleanup()


def _ensure_index(
    source: RepoSource,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
    index_config: IndexConfig | None = None,
) -> IndexStore:
    """Ensure the repo is indexed and return an open IndexStore.

    On exact cache hit (same commit), returns the cached store directly.
    On same-repo different-commit, applies delta if within threshold.
    On cache miss, runs the full acquire → parse → chunk → store pipeline.
    The caller is responsible for closing the returned store.
    """
    if config is None:
        config = Config()

    t_start = time.perf_counter()
    cache = CacheManager(cache_dir=config.cache_dir)
    cache_key = cache.cache_key(source)

    # Path 1: Exact cache hit (same commit) — fast path
    cached_db = cache.get(cache_key) if config.cache else None
    if cached_db is not None:
        store = IndexStore(cached_db)
        if not store.needs_reindex():
            if timing is not None:
                timing.cached = True
                timing.strategy = "cached"
                timing.index_ms = _elapsed_ms(t_start)
            return store
        store.close()
        cache.invalidate(cache_key)

    # Path 2: Delta path — same repo, different commit (local repos only)
    if config.cache and source.local_path:
        existing = cache.find_store_for_source(source)
        if existing is not None:
            db_path, cached_commit = existing
            current_commit = CacheManager.git_head(source.local_path)
            if current_commit and cached_commit != current_commit:
                try:
                    from archex.index.delta import apply_delta, compute_delta

                    repo_path = (
                        Path(source.local_path).resolve() if source.local_path else Path(".")
                    )
                    manifest = compute_delta(repo_path, cached_commit, current_commit)
                    total_files = len(
                        discover_files(
                            repo_path,
                            languages=config.languages,
                            max_file_size=config.max_file_size,
                        )
                    )
                    change_ratio = len(manifest.changes) / total_files if total_files > 0 else 1.0
                    if change_ratio < config.delta_threshold:
                        if timing is not None:
                            timing.delta_attempted = True
                        store = IndexStore(db_path)
                        graph = DependencyGraph.from_edges(store.get_edges())
                        delta_meta = apply_delta(
                            store,
                            graph,
                            manifest,
                            repo_path,
                            config,
                            index_config=index_config,
                        )
                        identity = source.url or source.local_path or ""
                        store.set_metadata("commit_hash", current_commit)
                        store.set_metadata("source_identity", identity)
                        store.set_metadata("indexed_at", str(time.time()))
                        store.conn.execute("PRAGMA wal_checkpoint(FULL)")
                        cache.put(
                            cache_key,
                            db_path,
                            resolved_commit=current_commit,
                            source_identity=identity,
                        )
                        if timing is not None:
                            timing.delta_ms = delta_meta.delta_time_ms
                            timing.delta_meta = delta_meta
                            timing.index_ms = _elapsed_ms(t_start)
                            timing.delta_succeeded = True
                            timing.strategy = "delta"
                        logger.info("Delta index applied in %.0fms", delta_meta.delta_time_ms)
                        return store
                except DeltaIndexError:
                    if timing is not None:
                        timing.delta_attempted = True
                    logger.info("Delta indexing failed, falling back to full re-index")

    # Path 3: Full re-index
    return _full_index(source, config, cache, cache_key, timing, index_config=index_config)


def _chunk_to_symbol_source(chunk: CodeChunk) -> SymbolSource:
    """Convert a CodeChunk to a SymbolSource model."""
    return SymbolSource(
        symbol_id=chunk.symbol_id or chunk.id,
        name=chunk.symbol_name or "",
        kind=chunk.symbol_kind or SymbolKind.VARIABLE,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        signature=chunk.signature,
        visibility=Visibility(chunk.visibility) if chunk.visibility else Visibility.PUBLIC,
        docstring=chunk.docstring,
        source=chunk.content,
        imports_context=chunk.imports_context,
        token_count=chunk.token_count,
    )


def _chunk_to_symbol_outline(chunk: CodeChunk) -> SymbolOutline:
    """Convert a CodeChunk to a SymbolOutline (no source code)."""
    return SymbolOutline(
        symbol_id=chunk.symbol_id or chunk.id,
        name=chunk.symbol_name or "",
        kind=chunk.symbol_kind or SymbolKind.VARIABLE,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        signature=chunk.signature,
        visibility=Visibility(chunk.visibility) if chunk.visibility else Visibility.PUBLIC,
        docstring=chunk.docstring,
    )


def _chunk_to_symbol_match(chunk: CodeChunk, score: float = 0.0) -> SymbolMatch:
    """Convert a CodeChunk to a SymbolMatch (search result)."""
    return SymbolMatch(
        symbol_id=chunk.symbol_id or chunk.id,
        name=chunk.symbol_name or "",
        kind=chunk.symbol_kind or SymbolKind.VARIABLE,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        signature=chunk.signature,
        visibility=Visibility(chunk.visibility) if chunk.visibility else Visibility.PUBLIC,
        relevance_score=score,
    )


logger = logging.getLogger(__name__)
_plugin_bootstrap_strict: bool | None = None


def _bootstrap_plugins(strict: bool = False) -> None:
    """Load adapter, pattern, and embedder plugins once per process."""
    global _plugin_bootstrap_strict
    if _plugin_bootstrap_strict is not None and (not strict or _plugin_bootstrap_strict):
        return  # already loaded at equal or higher strictness
    default_adapter_registry.load_entry_points(strict=strict)
    from archex.analyze.patterns import default_registry
    from archex.index.embeddings import default_embedder_registry

    default_registry.load_entry_points(strict=strict)
    default_embedder_registry.load_entry_points(strict=strict)
    _plugin_bootstrap_strict = strict


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _acquire(
    source: RepoSource,
) -> tuple[Path, str | None, str | None, Callable[[], None], str | None]:
    """Acquisition phase: resolve a RepoSource to a local filesystem path.

    Returns:
        (repo_path, url, local_path, cleanup_fn, cloned_head_commit)

    The cleanup_fn removes any temporary directory created for URL clones.
    For local paths, cleanup_fn is a no-op.
    """
    if source.url and (source.url.startswith("http://") or source.url.startswith("https://")):
        target_dir = tempfile.mkdtemp()

        def _url_cleanup() -> None:
            shutil.rmtree(target_dir, ignore_errors=True)

        repo_path = clone_repo(source.url, target_dir)
        cloned_head = CacheManager.git_head(str(repo_path))
        return repo_path, source.url, None, _url_cleanup, cloned_head

    if source.local_path is not None:

        def _noop() -> None:
            pass

        return open_local(source.local_path), None, source.local_path, _noop, None
    raise ValueError("RepoSource must have a url or local_path")


def _build_adapters() -> dict[str, LanguageAdapter]:
    """Build the registry of language adapters from the default registry."""
    return default_adapter_registry.build_all()


# ---------------------------------------------------------------------------
# Retrieval — search augmentation helpers (path boost, symbol seeds, reranking)
# ---------------------------------------------------------------------------


def _compute_top_k(total_chunks: int) -> int:
    """Scale BM25 candidate pool with repo size."""
    if total_chunks <= 100:
        return 30
    if total_chunks <= 500:
        return 50
    if total_chunks <= 2000:
        return 100
    return 150


_PATH_NOISE = frozenset(
    {
        "how",
        "does",
        "implement",
        "what",
        "handle",
        "manage",
        "function",
        "method",
        "class",
        "module",
        "file",
        "code",
        "work",
        "used",
        "using",
        "create",
        "make",
        "define",
        "call",
        "return",
        "type",
        "data",
        "value",
    }
)


_STEM_SUFFIXES = ("ors", "ers", "ing", "tion", "ment", "ness", "ity", "ies", "ous")

_SYMBOL_NOISE = _PATH_NOISE | frozenset(
    {
        "implement",
        "show",
        "find",
        "look",
        "search",
        "describe",
        "explain",
        "this",
        "that",
        "with",
        "from",
        "into",
        "have",
        "been",
        "does",
        "where",
        "which",
        "when",
        "will",
        "also",
        "each",
        "some",
    }
)


def _extract_path_terms(question: str) -> list[str]:
    """Extract terms from a query that might match file/directory names.

    Returns terms sorted longest-first so more specific terms (e.g. "validators")
    get priority over generic ones (e.g. "pydantic") when the boost limit is hit.
    Also generates stem variants by stripping common suffixes (e.g. "validators"
    → "validat") to match related file names like "_validate_call.py".
    """
    import re

    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", question)
    raw = [w.lower() for w in words if w.lower() not in _PATH_NOISE and len(w) >= 4]
    seen: set[str] = set()
    terms: list[str] = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            terms.append(t)
        for suffix in _STEM_SUFFIXES:
            if t.endswith(suffix) and len(t) - len(suffix) >= 4:
                stem = t[: -len(suffix)]
                if stem not in seen:
                    seen.add(stem)
                    terms.append(stem)
    terms.sort(key=len, reverse=True)
    return terms


def _file_path_boost(
    store: IndexStore,
    question: str,
    existing_ids: set[str],
    max_bm25_score: float = 1.0,
    max_boost_chunks: int = 30,
) -> list[tuple[CodeChunk, float]]:
    """Find chunks whose file_path contains query terms as exact substrings.

    Terms are searched longest-first (from _extract_path_terms) so specific terms
    like "validators" get priority over generic ones like "pydantic". Boost score
    is 0.5× max BM25 so path-matched chunks compete without dominating.
    """
    terms = _extract_path_terms(question)
    boosted: list[tuple[CodeChunk, float]] = []
    seen: set[str] = set(existing_ids)
    boost_score = max_bm25_score * 0.5

    for term in terms:
        for chunk in store.search_chunks_by_path_keyword(term, limit=20):
            if chunk.id not in seen:
                seen.add(chunk.id)
                boosted.append((chunk, boost_score))
            if len(boosted) >= max_boost_chunks:
                return boosted

    return boosted


_SYMBOL_EXACT_BOOST = 0.60
_SYMBOL_PARTIAL_BOOST = 0.15


def _symbol_search_seeds(
    store: IndexStore,
    question: str,
    max_bm25_score: float = 1.0,
    max_symbol_chunks: int = 20,
) -> list[tuple[CodeChunk, float]]:
    """Find chunks by matching query terms against symbol names via FTS5.

    Differentiates between two confidence levels:
    - Exact symbol_name match (case-insensitive equality): high-confidence seed,
      boosted at 0.60× max BM25.  Handles queries like "dependency injection"
      finding symbols literally named "depend" or "inject".
    - Partial / qualified-name match: low-confidence seed, boosted at 0.15×
      max BM25.  Gets the file into seed_files for import expansion without
      competing directly for top file slots.

    File_path-only matches from FTS5 are discarded as noise.
    """
    import re

    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", question)
    filtered = [w.lower() for w in words if w.lower() not in _SYMBOL_NOISE and len(w) >= 3]
    # Generate snake_case combinations of adjacent terms
    # e.g. "field validators" → "field_validator", "field_validators"
    bigrams: list[str] = []
    for i in range(len(filtered) - 1):
        bigrams.append(f"{filtered[i]}_{filtered[i + 1]}")
    # Only use bigrams (specific) and long single terms (>= 6 chars, less generic)
    terms = bigrams + [t for t in filtered if len(t) >= 6]

    seen: set[str] = set()
    seeds: list[tuple[CodeChunk, float]] = []

    for term in terms:
        chunks = store.search_symbols(term, limit=20)
        for chunk in chunks:
            # Only keep if symbol_name or qualified_name actually matches the term
            # (file_path-only matches from FTS5 are noise)
            name_lower = (chunk.symbol_name or "").lower()
            qname_lower = (chunk.qualified_name or "").lower()
            if term not in name_lower and term not in qname_lower:
                continue
            # Exact symbol_name equality → high-confidence seed
            exact_match = name_lower == term
            score = max_bm25_score * (_SYMBOL_EXACT_BOOST if exact_match else _SYMBOL_PARTIAL_BOOST)
            if chunk.id not in seen:
                seen.add(chunk.id)
                seeds.append((chunk, score))
            if len(seeds) >= max_symbol_chunks:
                return seeds

    return seeds


def _bm25_search_with_boosts(
    bm25: BM25Index,
    store: IndexStore,
    question: str,
    top_k: int,
) -> tuple[
    list[tuple[CodeChunk, float]],
    list[tuple[CodeChunk, float]],
    list[tuple[CodeChunk, float]],
]:
    """Run BM25 search plus file-path and symbol-seed boosts.

    Returns (search_results, path_boost, symbol_seeds) as separate lists so
    callers can record individual counts for observability, then combine them.
    """
    results = bm25.search(question, top_k=top_k)
    bm25_ids = {c.id for c, _ in results}
    max_bm25 = max((s for _, s in results), default=1.0)
    path_boost = _file_path_boost(store, question, bm25_ids, max_bm25_score=max_bm25)
    all_existing = bm25_ids | {c.id for c, _ in path_boost}
    symbol_seeds = [
        (c, s)
        for c, s in _symbol_search_seeds(store, question, max_bm25_score=max_bm25)
        if c.id not in all_existing
    ]
    return results, path_boost, symbol_seeds


def _vector_search_precomputed(
    npz_path: Path,
    all_chunks: list[CodeChunk],
    question: str,
    index_config: IndexConfig,
    top_k: int,
) -> list[tuple[CodeChunk, float]] | None:
    """Run independent vector search against a pre-computed .npz index.

    Returns None when vector is disabled, the embedder is unavailable, or the
    .npz file does not exist.
    """
    if not index_config.vector:
        return None
    if not npz_path.exists():
        return None
    embedder = _get_embedder(index_config)
    if embedder is None:
        return None
    from archex.index.vector import VectorIndex

    vec_idx = VectorIndex()
    try:
        vec_idx.load(
            npz_path,
            all_chunks,
            embedder_name=index_config.embedder or "",
            vector_dim=embedder.dimension,
            vector_mode=index_config.vector_mode,
            surrogate_version=index_config.surrogate_version,
        )
    except ArchexIndexError:
        return None
    return vec_idx.search(question, embedder, top_k=top_k)


def _surrogate_lookup(
    store: IndexStore | None,
    chunks: list[CodeChunk],
    index_config: IndexConfig,
) -> dict[str, ChunkSurrogate] | None:
    """Load or synthesize surrogate text for surrogate-mode vector retrieval."""
    if index_config.vector_mode != VectorMode.SURROGATE:
        return None
    stored_by_id: dict[str, ChunkSurrogate] = {}
    if store is not None:
        stored_by_id = {
            surrogate.chunk_id: surrogate
            for surrogate in store.get_chunk_surrogates([chunk.id for chunk in chunks])
        }
    missing = [chunk for chunk in chunks if chunk.id not in stored_by_id]
    if missing:
        for surrogate in build_chunk_surrogates(
            missing,
            version=index_config.surrogate_version,
        ):
            stored_by_id[surrogate.chunk_id] = surrogate
    return stored_by_id


def _compute_dynamic_budget(total_repo_tokens: int, user_budget: int) -> int:
    """Scale token budget proportional to repo size.

    - total ≤ budget: return total (passthrough — everything fits)
    - budget < total ≤ 3× budget: linear ramp from total down to budget
    - total > 3× budget: return user_budget (full retrieval mode)

    This prevents noise inflation on small repos while preserving full
    retrieval capacity for large ones.
    """
    if total_repo_tokens <= user_budget:
        return total_repo_tokens
    cap = user_budget * 3
    if total_repo_tokens >= cap:
        return user_budget
    # Linear interpolation: at 1× → total_repo_tokens, at 3× → user_budget
    t = (total_repo_tokens - user_budget) / (cap - user_budget)
    return int(total_repo_tokens * (1.0 - t) + user_budget * t)


def _total_chunk_tokens(chunks: list[CodeChunk]) -> int:
    """Sum estimated tokens across all chunks."""
    from archex.serve.context import estimate_tokens

    return sum(estimate_tokens(c) for c in chunks)


# ---------------------------------------------------------------------------
# Public entry points — orchestrate acquisition → indexing → retrieval/analysis
# ---------------------------------------------------------------------------


def analyze(
    source: RepoSource,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> ArchProfile:
    """Acquire, parse, and analyze a repository, returning an ArchProfile.

    Pipeline phases:
      1. Acquisition — clone or open local repo
      2. Parsing — discover files, extract symbols, resolve imports
      3. Analysis — detect modules, patterns, interfaces, infer decisions
      4. (Optional) LLM enrichment via configured provider

    The optional ``timing`` parameter records per-phase millisecond breakdowns.
    """
    if config is None:
        config = Config()
    _bootstrap_plugins()

    t0 = time.perf_counter()
    repo_path, url, local_path, cleanup, _cloned_head = _acquire(source)
    acquire_ms = _elapsed_ms(t0)
    logger.info("Acquired repo %s in %.0fms", url or local_path, acquire_ms)
    if timing is not None:
        timing.acquire_ms = acquire_ms
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
        parse_ms = _elapsed_ms(t1)  # discover + parse combined
        logger.info("Parsed %d files in %.0fms", len(parsed_files), _elapsed_ms(t2))
        if timing is not None:
            timing.parse_ms = parse_ms

        graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

        t3 = time.perf_counter()
        modules = detect_modules(graph, parsed_files)
        patterns = detect_patterns(parsed_files, graph)
        interfaces = extract_interfaces(parsed_files, graph)
        analysis_ms = _elapsed_ms(t3)
        logger.info(
            "Analysis: %d modules, %d patterns, %d interfaces in %.0fms",
            len(modules),
            len(patterns),
            len(interfaces),
            analysis_ms,
        )
        if timing is not None:
            timing.index_ms = analysis_ms

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
        if timing is not None:
            timing.total_ms = _elapsed_ms(t0)
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
    timing: PipelineTiming | None = None,
    trace: PipelineTrace | None = None,
) -> ContextBundle:
    """Retrieve a ranked ContextBundle for a natural-language query.

    Runs the full pipeline: acquire → parse → chunk → index → search → assemble.
    On cache hit, skips the full parse: loads chunks and graph from the cached store.

    When trace is provided, records step-level timings (acquire, parse, index,
    search, assemble) into the PipelineTrace for structured observability.
    """
    if config is None:
        config = Config()
    if index_config is None:
        index_config = IndexConfig()
    _bootstrap_plugins()

    t0 = time.perf_counter()
    if trace is not None:
        trace.operation = "query"
        trace.start_ns = time.perf_counter_ns()
    cache = CacheManager(cache_dir=config.cache_dir)
    cache_key = cache.cache_key(source)

    # Check cache BEFORE parsing — if cached, skip the expensive parse pipeline
    cached_db = cache.get(cache_key) if config.cache else None
    if cached_db is not None:
        logger.info("Cache hit for %s", cache_key[:12])
        store = IndexStore(cached_db)
        if store.needs_reindex():
            logger.info("Stale cache (missing symbol_ids) — forcing full re-index")
            store.close()
            cache.invalidate(cache_key)
            cached_db = None
        else:
            try:
                if timing is not None:
                    timing.cached = True
                if trace is not None:
                    trace.metadata["cache_hit"] = True

                # Use persisted corpus stats — avoids full-chunk hydration
                stored_tokens = store.get_metadata("repo_total_tokens")
                total_repo_tokens = (
                    int(stored_tokens) if stored_tokens is not None else store.get_total_tokens()
                )
                stored_count = store.get_metadata("chunk_count")
                chunk_count = (
                    int(stored_count) if stored_count is not None else len(store.get_chunks())
                )
                effective_budget = _compute_dynamic_budget(total_repo_tokens, token_budget)

                # Passthrough: entire repo fits within budget
                if effective_budget >= total_repo_tokens:
                    cached_chunks = store.get_chunks()
                    pt = passthrough_context(cached_chunks, question, effective_budget)
                    if timing is not None:
                        timing.strategy = "passthrough"
                    if trace is not None:
                        trace.metadata["strategy"] = "passthrough"
                    pt.retrieval_metadata.retrieval_time_ms = _elapsed_ms(t0)
                    logger.info("query() [passthrough] completed in %.0fms", _elapsed_ms(t0))
                    if timing is not None:
                        timing.total_ms = _elapsed_ms(t0)
                    if trace is not None:
                        trace.end_ns = time.perf_counter_ns()
                        trace.log_summary()
                    return pt

                bm25 = BM25Index(store)
                stored_edges = store.get_edges()
                graph = DependencyGraph.from_edges(stored_edges)

                top_k = _compute_top_k(chunk_count)
                # Pre-load all chunks into memory before parallel search so the
                # vector thread has no dependency on the SQLite store connection.
                all_chunks_cached = store.get_chunks()
                surrogate_lookup = _surrogate_lookup(store, all_chunks_cached, index_config)

                t_search = time.perf_counter()
                cached_npz = cache.vector_path(
                    cache_key,
                    vector_mode=index_config.vector_mode,
                    surrogate_version=index_config.surrogate_version,
                )
                # Run vector search in a background thread while BM25 runs on the
                # calling thread.  BM25 must stay on the creating thread because
                # SQLite connections are not thread-safe; the vector path uses only
                # pre-loaded numpy arrays and requires no store access.
                with ThreadPoolExecutor(max_workers=1) as _pool:
                    _vec_future = _pool.submit(
                        _vector_search_precomputed,
                        cached_npz,
                        all_chunks_cached,
                        question,
                        index_config,
                        top_k,
                    )
                    if index_config.bm25:
                        _bm25_raw, path_boost, symbol_seeds = _bm25_search_with_boosts(
                            bm25, store, question, top_k
                        )
                        search_results = _bm25_raw + path_boost + symbol_seeds
                    else:
                        search_results = []
                        path_boost = []
                        symbol_seeds = []
                    vector_results: list[tuple[CodeChunk, float]] | None = _vec_future.result()

                # Fall back to rerank when pre-computed .npz is absent
                if vector_results is None and index_config.vector:
                    if search_results:
                        vector_results = _two_stage_rerank(
                            question,
                            search_results,
                            index_config,
                            timing,
                            surrogates_by_chunk_id=surrogate_lookup,
                        )
                    else:
                        vector_results = _vector_search_in_memory(
                            question,
                            all_chunks_cached,
                            index_config,
                            timing,
                            top_k=top_k,
                            surrogates_by_chunk_id=surrogate_lookup,
                        )

                t_vec_cached = t_search  # timing reference for logging below
                if vector_results is not None and timing is not None:
                    timing.vector_used = True
                    timing.vector_build_ms = _elapsed_ms(t_vec_cached)
                if vector_results is not None:
                    logger.info(
                        "Vector search (pre-computed cached, parallel) in %.0fms",
                        _elapsed_ms(t_search),
                    )

                if timing is not None:
                    timing.search_ms = _elapsed_ms(t_search)
                if trace is not None:
                    trace.add_step(
                        StepTiming(
                            name="search",
                            start_ns=int(t_search * 1_000_000_000),
                            end_ns=time.perf_counter_ns(),
                            metadata={
                                "bm25_results": len(search_results),
                                "path_boost": len(path_boost),
                                "symbol_seeds": len(symbol_seeds),
                                "top_k": top_k,
                            },
                        )
                    )
                bundle = assemble_context(
                    search_results=search_results,
                    graph=graph,
                    all_chunks=all_chunks_cached,
                    question=question,
                    token_budget=effective_budget,
                    vector_results=vector_results,  # type: ignore[arg-type]
                    scoring_weights=scoring_weights,
                    trace=trace,
                )
                bundle.retrieval_metadata.vector_mode = index_config.vector_mode
                bundle.retrieval_metadata.surrogate_version = (
                    index_config.surrogate_version
                    if index_config.vector_mode == VectorMode.SURROGATE
                    else None
                )
                if timing is not None:
                    timing.assemble_ms = bundle.retrieval_metadata.assembly_time_ms
                bundle.retrieval_metadata.retrieval_time_ms = _elapsed_ms(t0)
                logger.info("query() [cached] completed in %.0fms", _elapsed_ms(t0))
                if timing is not None:
                    timing.total_ms = _elapsed_ms(t0)
                if trace is not None:
                    trace.end_ns = time.perf_counter_ns()
                    trace.log_summary()
                return bundle
            finally:
                store.close()

    # Cache miss — full pipeline
    logger.info("Cache miss — running full pipeline")
    if trace is not None:
        trace.metadata["cache_hit"] = False

    t1 = time.perf_counter()
    repo_path, _url, _local_path, cleanup, cloned_head = _acquire(source)
    acquire_ms = _elapsed_ms(t1)
    logger.info("Acquired repo in %.0fms", acquire_ms)
    if timing is not None:
        timing.acquire_ms = acquire_ms
    if trace is not None:
        trace.add_step(
            StepTiming(
                name="acquire",
                start_ns=int(t1 * 1_000_000_000),
                end_ns=time.perf_counter_ns(),
            )
        )
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
        parse_ms = _elapsed_ms(t3)
        logger.info("Parsed %d files in %.0fms", len(parsed_files), parse_ms)
        if timing is not None:
            timing.parse_ms = parse_ms
        if trace is not None:
            trace.add_step(
                StepTiming(
                    name="parse",
                    start_ns=int(t3 * 1_000_000_000),
                    end_ns=time.perf_counter_ns(),
                    metadata={
                        "files_discovered": len(files),
                        "files_parsed": len(parsed_files),
                    },
                )
            )

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

        total_repo_tokens = _total_chunk_tokens(all_chunks)
        effective_budget = _compute_dynamic_budget(total_repo_tokens, token_budget)

        db_path = Path(tempfile.mkdtemp()) / "index.db"
        store = IndexStore(db_path)
        chunk_surrogates = build_chunk_surrogates(
            all_chunks,
            version=index_config.surrogate_version,
        )
        surrogate_lookup = {surrogate.chunk_id: surrogate for surrogate in chunk_surrogates}

        try:
            bm25 = BM25Index(store)
            store.insert_chunks(all_chunks)
            store.insert_chunk_surrogates(chunk_surrogates)
            edges = graph.file_edges()
            store.insert_edges(edges)
            bm25.build(all_chunks)

            # Pre-compute vector embeddings at index time when vector mode is enabled
            _precomputed_vector_path: Path | None = None
            if index_config.vector:
                embedder = _get_embedder(index_config)
                if embedder is not None:
                    from archex.index.vector import VectorIndex

                    t_vec_build = time.perf_counter()
                    vec_idx = VectorIndex()
                    vec_idx.build(
                        all_chunks,
                        embedder,
                        surrogates_by_chunk_id=surrogate_lookup,
                        vector_mode=index_config.vector_mode,
                    )
                    npz_path = (
                        cache.vector_path(
                            cache_key,
                            vector_mode=index_config.vector_mode,
                            surrogate_version=index_config.surrogate_version,
                        )
                        if config.cache
                        else store.vector_index_path_for(
                            vector_mode=index_config.vector_mode,
                            surrogate_version=index_config.surrogate_version,
                        )
                    )
                    vec_idx.save(
                        npz_path,
                        embedder_name=index_config.embedder or "",
                        vector_dim=embedder.dimension,
                        vector_mode=index_config.vector_mode,
                        surrogate_version=index_config.surrogate_version,
                    )
                    _precomputed_vector_path = npz_path
                    vec_build_ms = _elapsed_ms(t_vec_build)
                    if timing is not None:
                        timing.vector_index_ms = vec_build_ms
                    logger.info(
                        "Pre-computed %d chunk embeddings at index time in %.0fms",
                        len(all_chunks),
                        vec_build_ms,
                    )

            if timing is not None:
                timing.index_ms = _elapsed_ms(t4)
            if trace is not None:
                trace.add_step(
                    StepTiming(
                        name="index",
                        start_ns=int(t4 * 1_000_000_000),
                        end_ns=time.perf_counter_ns(),
                        metadata={
                            "chunks": len(all_chunks),
                            "edges": len(edges),
                        },
                    )
                )

            top_k = _compute_top_k(len(all_chunks))

            # Persist corpus stats for fast warm-query access
            store.set_metadata("repo_total_tokens", str(total_repo_tokens))
            store.set_metadata("chunk_count", str(len(all_chunks)))
            file_count = len({c.file_path for c in all_chunks})
            store.set_metadata("file_count", str(file_count))

            if config.cache:
                commit = cloned_head or cache.git_head(source.local_path) or source.commit or ""
                identity = source.url or source.local_path or ""
                store.set_metadata("commit_hash", commit)
                store.set_metadata("source_identity", identity)
                store.set_metadata("indexed_at", str(time.time()))
                store.conn.execute("PRAGMA wal_checkpoint(FULL)")
                cache.put(
                    cache_key,
                    db_path,
                    resolved_commit=commit,
                    source_identity=identity,
                )

            # Passthrough: entire repo fits within budget
            if effective_budget >= total_repo_tokens:
                pt = passthrough_context(all_chunks, question, effective_budget)
                if timing is not None:
                    timing.strategy = "passthrough"
                    timing.total_ms = _elapsed_ms(t0)
                if trace is not None:
                    trace.metadata["strategy"] = "passthrough"
                    trace.end_ns = time.perf_counter_ns()
                    trace.log_summary()
                pt.retrieval_metadata.retrieval_time_ms = _elapsed_ms(t0)
                logger.info("query() [passthrough] completed in %.0fms", _elapsed_ms(t0))
                return pt

            t6 = time.perf_counter()
            # all_chunks already loaded above for graph building; pass it to the
            # vector thread so the vector path has no SQLite store dependency.
            # BM25 must stay on the calling thread — SQLite connections are not
            # thread-safe.  Vector search runs in a background thread (pure numpy).
            _miss_npz: Path = (
                _precomputed_vector_path
                if _precomputed_vector_path is not None
                else Path("/nonexistent/__no_npz__")
            )
            with ThreadPoolExecutor(max_workers=1) as _pool:
                _vec_future = _pool.submit(
                    _vector_search_precomputed,
                    _miss_npz,
                    all_chunks,
                    question,
                    index_config,
                    top_k,
                )
                if index_config.bm25:
                    _bm25_raw, path_boost, symbol_seeds_miss = _bm25_search_with_boosts(
                        bm25, store, question, top_k
                    )
                    search_results = _bm25_raw + path_boost + symbol_seeds_miss
                else:
                    search_results = []
                    path_boost = []
                    symbol_seeds_miss = []
                vector_results_miss: list[tuple[CodeChunk, float]] | None = _vec_future.result()

            # Fall back to rerank when pre-computed .npz is absent
            if vector_results_miss is None and index_config.vector:
                if search_results:
                    vector_results_miss = _two_stage_rerank(
                        question,
                        search_results,
                        index_config,
                        timing,
                        surrogates_by_chunk_id=surrogate_lookup,
                    )
                else:
                    vector_results_miss = _vector_search_in_memory(
                        question,
                        all_chunks,
                        index_config,
                        timing,
                        top_k=top_k,
                        surrogates_by_chunk_id=surrogate_lookup,
                    )

            if vector_results_miss is not None and timing is not None:
                timing.vector_used = True
                timing.vector_build_ms = _elapsed_ms(t6)
            if vector_results_miss is not None:
                logger.info(
                    "Vector search (pre-computed, parallel) in %.0fms",
                    _elapsed_ms(t6),
                )

            if timing is not None:
                timing.search_ms = _elapsed_ms(t6)
            if trace is not None:
                trace.add_step(
                    StepTiming(
                        name="search",
                        start_ns=int(t6 * 1_000_000_000),
                        end_ns=time.perf_counter_ns(),
                        metadata={
                            "bm25_results": len(search_results),
                            "path_boost": len(path_boost),
                            "symbol_seeds": len(symbol_seeds_miss),
                            "top_k": top_k,
                        },
                    )
                )
            bundle = assemble_context(
                search_results=search_results,
                graph=graph,
                all_chunks=all_chunks,
                question=question,
                token_budget=effective_budget,
                vector_results=vector_results_miss,  # type: ignore[arg-type]
                scoring_weights=scoring_weights,
                trace=trace,
            )
            bundle.retrieval_metadata.vector_mode = index_config.vector_mode
            bundle.retrieval_metadata.surrogate_version = (
                index_config.surrogate_version
                if index_config.vector_mode == VectorMode.SURROGATE
                else None
            )
            if timing is not None:
                timing.assemble_ms = bundle.retrieval_metadata.assembly_time_ms
            bundle.retrieval_metadata.retrieval_time_ms = _elapsed_ms(t0)
            logger.info("Search + assemble in %.0fms", _elapsed_ms(t6))
        finally:
            store.close()

        logger.info("query() completed in %.0fms", _elapsed_ms(t0))
        if timing is not None:
            timing.total_ms = _elapsed_ms(t0)
        if trace is not None:
            trace.end_ns = time.perf_counter_ns()
            trace.log_summary()
        return bundle
    finally:
        cleanup()


_RERANK_MAX_CANDIDATES = 50


def _two_stage_rerank(
    question: str,
    bm25_results: list[tuple[CodeChunk, float]],
    index_config: IndexConfig,
    timing: PipelineTiming | None,
    *,
    surrogates_by_chunk_id: dict[str, ChunkSurrogate] | None = None,
) -> list[tuple[CodeChunk, float]] | None:
    """Rerank BM25 candidates using vector similarity (two-stage retrieval).

    Embeds only the top BM25 candidate chunks + query instead of the full corpus.
    Caps at _RERANK_MAX_CANDIDATES to bound embedding latency.
    """
    embedder = _get_embedder(index_config)
    if embedder is None:
        return None

    from archex.index.vector import VectorIndex

    candidates = [chunk for chunk, _ in bm25_results[:_RERANK_MAX_CANDIDATES]]
    t_vec = time.perf_counter()
    vec_idx = VectorIndex()
    vector_results = vec_idx.rerank(
        question,
        candidates,
        embedder,  # type: ignore[arg-type]
        surrogates_by_chunk_id=surrogates_by_chunk_id,
        vector_mode=index_config.vector_mode,
    )
    rerank_ms = _elapsed_ms(t_vec)
    if timing is not None:
        timing.vector_used = True
        timing.vector_build_ms = rerank_ms
    logger.info("Two-stage rerank (%d candidates) in %.0fms", len(candidates), rerank_ms)
    return vector_results  # type: ignore[return-value]


def _vector_search_in_memory(
    question: str,
    all_chunks: list[CodeChunk],
    index_config: IndexConfig,
    timing: PipelineTiming | None,
    *,
    top_k: int,
    surrogates_by_chunk_id: dict[str, ChunkSurrogate] | None = None,
) -> list[tuple[CodeChunk, float]] | None:
    """Build an in-memory vector index when no persisted artifact is available."""
    embedder = _get_embedder(index_config)
    if embedder is None:
        return None

    from archex.index.vector import VectorIndex

    t_vec = time.perf_counter()
    vec_idx = VectorIndex()
    vec_idx.build(
        all_chunks,
        embedder,  # type: ignore[arg-type]
        surrogates_by_chunk_id=surrogates_by_chunk_id,
        vector_mode=index_config.vector_mode,
    )
    vector_results = vec_idx.search(question, embedder, top_k=top_k)  # type: ignore[arg-type]
    build_ms = _elapsed_ms(t_vec)
    if timing is not None:
        timing.vector_used = True
        timing.vector_build_ms = build_ms
    logger.info("In-memory vector search (%d chunks) in %.0fms", len(all_chunks), build_ms)
    return vector_results  # type: ignore[return-value]


def _get_embedder(index_config: IndexConfig) -> Embedder | None:
    """Create an embedder from index_config via the EmbedderRegistry."""
    from archex.index.embeddings import default_embedder_registry

    return default_embedder_registry.create(index_config)


def compare(
    source_a: RepoSource,
    source_b: RepoSource,
    dimensions: list[str] | None = None,
    config: Config | None = None,
) -> ComparisonResult:
    """Analyze two repositories and return a ComparisonResult.

    Runs two parallel ``analyze()`` pipelines (acquisition → parsing → analysis)
    then delegates to ``compare_repos()`` for dimension-by-dimension comparison.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(analyze, source_a, config)
        future_b = executor.submit(analyze, source_b, config)
        profile_a = future_a.result()
        profile_b = future_b.result()
    return compare_repos(profile_a, profile_b, dimensions)


# ---------------------------------------------------------------------------
# Tier 1 — Precision Symbol Tools
# ---------------------------------------------------------------------------


def file_tree(
    source: RepoSource,
    max_depth: int = 5,
    language: str | None = None,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> FileTree:
    """Return the annotated file structure of an indexed repository."""
    t0 = time.perf_counter()
    store = _ensure_index(source, config, timing=timing)
    try:
        t_op = time.perf_counter()
        file_meta = store.get_file_metadata()
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    # Filter by language if requested
    if language:
        file_meta = [m for m in file_meta if m["language"] == language]

    # Build hierarchical tree from flat file paths
    lang_counts: dict[str, int] = {}
    root_entries: dict[str, FileTreeEntry] = {}

    for meta in file_meta:
        fp = str(meta["file_path"])
        lang = str(meta["language"])
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        parts = fp.split("/")
        # Walk/create directory entries
        current_level = root_entries
        for i, part in enumerate(parts[:-1]):
            if i >= max_depth:
                break
            if part not in current_level:
                dir_path = "/".join(parts[: i + 1])
                current_level[part] = FileTreeEntry(path=dir_path, is_directory=True)
            entry = current_level[part]
            # Build a child dict from the children list for lookup
            child_map = {c.path.split("/")[-1]: c for c in entry.children}
            current_level = child_map  # type: ignore[assignment]
            # Ensure the next level exists in children
            if i + 1 < len(parts) - 1:
                next_part = parts[i + 1]
                if next_part not in current_level:
                    next_path = "/".join(parts[: i + 2])
                    new_child = FileTreeEntry(path=next_path, is_directory=True)
                    entry.children.append(new_child)
                    current_level[next_part] = new_child  # type: ignore[assignment]

        # Add the file entry
        if len(parts) - 1 < max_depth:
            file_entry = FileTreeEntry(
                path=fp,
                language=lang,
                lines=int(meta["lines"]),
                symbol_count=int(meta["symbol_count"]),
                is_directory=False,
            )
            if len(parts) == 1:
                root_entries[parts[0]] = file_entry
            else:
                _add_file_to_tree(root_entries, parts, file_entry)

    entries = sorted(root_entries.values(), key=lambda e: (not e.is_directory, e.path))

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return FileTree(
        root=source.local_path or source.url or "",
        entries=entries,
        total_files=len(file_meta),
        languages=lang_counts,
    )


def _add_file_to_tree(
    root_entries: dict[str, FileTreeEntry],
    parts: list[str],
    file_entry: FileTreeEntry,
) -> None:
    """Walk the tree and add a file entry under its parent directory."""
    if parts[0] not in root_entries:
        dir_path = parts[0]
        root_entries[parts[0]] = FileTreeEntry(path=dir_path, is_directory=True)

    current = root_entries[parts[0]]
    for part in parts[1:-1]:
        found = False
        for child in current.children:
            if child.path.split("/")[-1] == part:
                current = child
                found = True
                break
        if not found:
            return

    # Avoid duplicate file entries
    existing_paths = {c.path for c in current.children}
    if file_entry.path not in existing_paths:
        current.children.append(file_entry)


def file_outline(
    source: RepoSource,
    file_path: str,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> FileOutline:
    """Return the symbol hierarchy for a single file — no source code."""
    t0 = time.perf_counter()
    store = _ensure_index(source, config, timing=timing)
    try:
        t_op = time.perf_counter()
        chunks = store.get_chunks_for_file(file_path)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if not chunks:
        return FileOutline(
            file_path=file_path,
            language="unknown",
            lines=0,
            symbols=[],
            token_count_raw=0,
        )

    language = chunks[0].language
    max_line = max(c.end_line for c in chunks)
    token_count_raw = sum(c.token_count for c in chunks)

    # Build flat outlines
    outlines = [_chunk_to_symbol_outline(c) for c in chunks]

    # Reconstruct parent-child hierarchy from qualified_name
    top_level: list[SymbolOutline] = []
    by_qname: dict[str, SymbolOutline] = {}

    for outline in outlines:
        qname = outline.name
        chunk = next((c for c in chunks if (c.symbol_id or c.id) == outline.symbol_id), None)
        if chunk and chunk.qualified_name:
            qname = chunk.qualified_name
        by_qname[qname] = outline

    for outline in outlines:
        chunk = next((c for c in chunks if (c.symbol_id or c.id) == outline.symbol_id), None)
        qname = chunk.qualified_name if chunk and chunk.qualified_name else outline.name
        # Check if this is a child (has a dot separator indicating parent.child)
        parent_name = _get_parent_qname(qname)
        if parent_name and parent_name in by_qname:
            by_qname[parent_name].children.append(outline)
        else:
            top_level.append(outline)

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return FileOutline(
        file_path=file_path,
        language=language,
        lines=max_line,
        symbols=top_level,
        token_count_raw=token_count_raw,
    )


def _get_parent_qname(qualified_name: str) -> str | None:
    """Extract the parent's qualified name from a dotted or :: separated name."""
    if "::" in qualified_name:
        parts = qualified_name.rsplit("::", 1)
        return parts[0] if len(parts) > 1 else None
    if "." in qualified_name:
        parts = qualified_name.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else None
    return None


def search_symbols(
    source: RepoSource,
    query: str,
    kind: str | None = None,
    language: str | None = None,
    limit: int = 20,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> list[SymbolMatch]:
    """Search symbols by name across the indexed repository."""
    t0 = time.perf_counter()
    store = _ensure_index(source, config, timing=timing)
    try:
        t_op = time.perf_counter()
        sym_kind = SymbolKind(kind) if kind else None
        chunks = store.search_symbols(query, kind=sym_kind, limit=limit)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if language:
        chunks = [c for c in chunks if c.language == language]

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return [_chunk_to_symbol_match(c) for c in chunks[:limit]]


def get_symbol(
    source: RepoSource,
    symbol_id: str,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> SymbolSource | None:
    """Retrieve the full source code of a single symbol by its stable ID."""
    t0 = time.perf_counter()
    store = _ensure_index(source, config, timing=timing)
    try:
        t_op = time.perf_counter()
        chunk = store.get_chunk_by_symbol_id(symbol_id)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    if chunk is None:
        return None
    return _chunk_to_symbol_source(chunk)


def get_symbols_batch(
    source: RepoSource,
    symbol_ids: list[str],
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> list[SymbolSource | None]:
    """Batch retrieve N symbols by their stable IDs. Preserves input order."""
    if len(symbol_ids) > 50:
        raise ValueError(f"Maximum 50 symbol IDs per batch, got {len(symbol_ids)}")

    t0 = time.perf_counter()
    store = _ensure_index(source, config, timing=timing)
    try:
        t_op = time.perf_counter()
        chunks = store.get_chunks_by_symbol_ids(symbol_ids)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    # Preserve input order: build lookup by symbol_id, map back
    by_sid: dict[str, CodeChunk] = {c.symbol_id: c for c in chunks if c.symbol_id}
    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return [_chunk_to_symbol_source(by_sid[sid]) if sid in by_sid else None for sid in symbol_ids]


# ---------------------------------------------------------------------------
# Token efficiency utilities
# ---------------------------------------------------------------------------


def get_repo_total_tokens(
    source: RepoSource,
    config: Config | None = None,
) -> int:
    """Return the total token count across all indexed chunks for a repository."""
    store = _ensure_index(source, config)
    try:
        return store.get_total_tokens()
    finally:
        store.close()


def get_file_token_count(
    source: RepoSource,
    file_path: str,
    config: Config | None = None,
) -> int:
    """Return the total token count for a single file in an indexed repository."""
    store = _ensure_index(source, config)
    try:
        return store.get_file_tokens(file_path)
    finally:
        store.close()


def get_files_token_count(
    source: RepoSource,
    file_paths: list[str],
    config: Config | None = None,
) -> int:
    """Return the total token count across unique files in an indexed repository."""
    store = _ensure_index(source, config)
    try:
        return store.get_files_tokens(file_paths)
    finally:
        store.close()
