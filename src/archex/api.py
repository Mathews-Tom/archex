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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from archex import precision as precision_api
from archex.acquire import clone_repo, discover_files, open_local
from archex.analyze.decisions import infer_decisions
from archex.analyze.interfaces import extract_interfaces
from archex.analyze.modules import detect_modules
from archex.analyze.patterns import detect_patterns
from archex.cache import CacheManager
from archex.config import load_config, load_index_config
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
    IndexConfig,
    Module,
    ParsedFile,
    PipelineTiming,
    RepoMetadata,
    RepoSource,
    ScoringWeights,
    SymbolMatch,
    SymbolSource,
    VectorMode,
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
from archex.project import uses_project_cache_layout
from archex.serve.compare import compare_repos
from archex.serve.context import assemble_context, passthrough_context
from archex.serve.profile import build_profile

if TYPE_CHECKING:
    from collections.abc import Callable

    from archex.index.embeddings.base import Embedder
    from archex.index.rerank import CrossEncoderReranker
    from archex.models import ComparisonResult

# ---------------------------------------------------------------------------
# Acquisition + Indexing — shared helpers for all entry points
# ---------------------------------------------------------------------------


def _cache_manager_for_source(source: RepoSource, config: Config) -> CacheManager:
    project_layout = False
    if source.local_path:
        try:
            project_layout = uses_project_cache_layout(source.local_path, config.cache_dir)
        except ValueError:
            project_layout = False
    return CacheManager(cache_dir=config.cache_dir, project_layout=project_layout)


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
        _build_splade_index(store, all_chunks, effective_index_config)
        _build_module_summaries(store, graph, parsed_files, effective_index_config)

        if config.cache:
            commit = cloned_head or cache.git_head(source.local_path) or source.commit or ""
            identity = source.url or source.local_path or ""
            store.set_metadata("commit_hash", commit)
            store.set_metadata("source_identity", identity)
            store.set_metadata("indexed_at", str(time.time()))
            _set_working_tree_signature(store, repo_path, config)
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


@dataclass(frozen=True)
class _DeltaIndexAttempt:
    source: RepoSource
    config: Config
    cache: CacheManager
    cache_key: str
    working_tree_signature: str | None
    timing: PipelineTiming | None
    index_config: IndexConfig | None
    t_start: float


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
        config = load_config(source)

    t_start = time.perf_counter()
    cache = _cache_manager_for_source(source, config)
    cache_key = cache.cache_key(source)
    working_tree_signature = _working_tree_signature(source, config)

    # Path 1: Exact cache hit (same commit) — fast path
    cached_db = cache.get(cache_key) if config.cache else None
    if cached_db is not None:
        store = IndexStore(cached_db)
        store_signature = store.get_metadata("working_tree_signature")
        signature_matches = (
            working_tree_signature is None or store_signature == working_tree_signature
        )
        needs_reindex = store.needs_reindex()
        if not needs_reindex and signature_matches:
            if timing is not None:
                timing.cached = True
                timing.strategy = "cached"
                timing.index_ms = _elapsed_ms(t_start)
            return store
        store.close()
        if needs_reindex:
            cache.invalidate(cache_key)

    # Path 2: Delta path — same repo, different commit (local repos only)
    delta_store = _try_delta_index(
        _DeltaIndexAttempt(
            source=source,
            config=config,
            cache=cache,
            cache_key=cache_key,
            working_tree_signature=working_tree_signature,
            timing=timing,
            index_config=index_config,
            t_start=t_start,
        )
    )
    if delta_store is not None:
        return delta_store

    # Path 3: Full re-index
    return _full_index(source, config, cache, cache_key, timing, index_config=index_config)


def index_repository(
    source: RepoSource,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
    index_config: IndexConfig | None = None,
) -> IndexStore:
    """Build or load the index for a repository without running analysis or retrieval."""
    return _ensure_index(
        source,
        config=config,
        timing=timing,
        index_config=index_config,
    )


def _try_delta_index(attempt: _DeltaIndexAttempt) -> IndexStore | None:
    source = attempt.source
    config = attempt.config
    if not config.cache or not source.local_path:
        return None

    existing = attempt.cache.find_store_for_source(source)
    if existing is None:
        return None

    db_path, cached_commit = existing
    current_commit = CacheManager.git_head(source.local_path)
    if not current_commit:
        return None
    same_commit = cached_commit == current_commit
    if same_commit and attempt.working_tree_signature is None:
        return None

    try:
        from archex.index.delta import apply_delta, compute_delta, compute_mtime_delta

        repo_path = Path(source.local_path).resolve()
        store = IndexStore(db_path) if same_commit else None
        try:
            if same_commit:
                if store is None:
                    return None
                indexed_signature = store.get_metadata("working_tree_signature")
                if indexed_signature == attempt.working_tree_signature:
                    return None
                indexed_at = _metadata_float(store.get_metadata("indexed_at"))
                manifest = compute_mtime_delta(repo_path, store, indexed_at)
                manifest.base_commit = cached_commit
                manifest.current_commit = current_commit
            else:
                manifest = compute_delta(repo_path, cached_commit, current_commit)
        finally:
            if store is not None:
                store.close()

        if not manifest.changes and same_commit:
            return None

        total_files = len(
            discover_files(
                repo_path,
                languages=config.languages,
                max_file_size=config.max_file_size,
            )
        )
        change_ratio = len(manifest.changes) / total_files if total_files > 0 else 1.0
        if change_ratio >= config.delta_threshold:
            return None

        if attempt.timing is not None:
            attempt.timing.delta_attempted = True
        store = IndexStore(db_path)
        graph = DependencyGraph.from_edges(store.get_edges())
        delta_meta = apply_delta(
            store,
            graph,
            manifest,
            repo_path,
            config,
            index_config=attempt.index_config,
        )
        identity = source.url or source.local_path or ""
        store.set_metadata("commit_hash", current_commit)
        store.set_metadata("source_identity", identity)
        store.set_metadata("indexed_at", str(time.time()))
        if attempt.working_tree_signature is not None:
            store.set_metadata("working_tree_signature", attempt.working_tree_signature)
        store.conn.execute("PRAGMA wal_checkpoint(FULL)")
        attempt.cache.put(
            attempt.cache_key,
            db_path,
            resolved_commit=current_commit,
            source_identity=identity,
        )
        if attempt.timing is not None:
            attempt.timing.delta_ms = delta_meta.delta_time_ms
            attempt.timing.delta_meta = delta_meta
            attempt.timing.index_ms = _elapsed_ms(attempt.t_start)
            attempt.timing.delta_succeeded = True
            attempt.timing.strategy = "delta"
        logger.info("Delta index applied in %.0fms", delta_meta.delta_time_ms)
        return store
    except DeltaIndexError:
        if attempt.timing is not None:
            attempt.timing.delta_attempted = True
        logger.info("Delta indexing failed, falling back to full re-index")
        return None


def _working_tree_signature(source: RepoSource, config: Config) -> str | None:
    if not source.local_path:
        return None
    repo_path = Path(source.local_path).expanduser().resolve()
    if not (repo_path / ".git").exists():
        return None
    from archex.index.delta import compute_working_tree_signature

    return compute_working_tree_signature(repo_path, config)


def _set_working_tree_signature(store: IndexStore, repo_path: Path, config: Config) -> None:
    if not (repo_path / ".git").exists():
        return
    from archex.index.delta import compute_working_tree_signature

    store.set_metadata("working_tree_signature", compute_working_tree_signature(repo_path, config))


def _metadata_float(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


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


def _maybe_reranker(index_config: IndexConfig) -> CrossEncoderReranker | None:
    """Return a CrossEncoderReranker only when reranking is explicitly enabled.

    Reranking activates solely on index_config.rerank. Callers that leave it
    unset (the default) opt out regardless of whether sentence-transformers is
    installed, so the flag remains observable as on/off.
    """
    from archex.index.rerank import DEFAULT_MODEL, CrossEncoderReranker

    if index_config.rerank:
        return CrossEncoderReranker(
            model_name=index_config.rerank_model or DEFAULT_MODEL,
        )

    return None


def _compute_top_k(total_chunks: int) -> int:
    """Scale BM25 candidate pool with repo size."""
    if total_chunks <= 100:
        return 30
    if total_chunks <= 500:
        return 50
    if total_chunks <= 2000:
        return 100
    return 150


def _compute_vector_top_k(*, bm25_top_k: int, total_chunks: int) -> int:
    """Use a deeper independent vector pool for fusion candidate union."""
    return min(total_chunks, bm25_top_k * 2)


def _compute_splade_top_k(*, bm25_top_k: int, total_chunks: int) -> int:
    """Use SPLADE as a recall leg with the same depth as vector fusion."""
    return min(total_chunks, bm25_top_k * 2)


def _build_splade_index(
    store: IndexStore,
    chunks: list[CodeChunk],
    index_config: IndexConfig,
) -> None:
    if not index_config.splade:
        return
    from archex.index.splade import SPLADEIndex

    splade = SPLADEIndex(store)
    splade.build(chunks)


def _build_module_summaries(
    store: IndexStore,
    graph: DependencyGraph,
    parsed_files: list[ParsedFile],
    index_config: IndexConfig,
) -> list[Module]:
    if not index_config.module_prefilter:
        return []
    modules = detect_modules(graph, parsed_files)
    store.insert_modules(modules)
    return modules


def _splade_search_or_raise(
    store: IndexStore,
    question: str,
    index_config: IndexConfig,
    top_k: int,
) -> list[tuple[CodeChunk, float]] | None:
    if not index_config.splade:
        return None
    from archex.index.splade import SPLADEIndex

    splade = SPLADEIndex(store)
    if not splade.has_data:
        raise ArchexIndexError(
            "SPLADE requested but the cached index has no SPLADE rows; "
            "refresh it with `archex index --splade`."
        )
    return splade.search(question, top_k=top_k)


def _modules_or_raise(store: IndexStore, index_config: IndexConfig) -> list[Module]:
    if not index_config.module_prefilter:
        return []
    modules = store.get_modules()
    if not modules:
        raise ArchexIndexError(
            "Module prefilter requested but the cached index has no module summaries; "
            "refresh it with `archex index --module-prefilter`."
        )
    return modules


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


_STEM_SUFFIXES = ("ors", "ers", "ing", "tion", "ment", "ness", "ity", "ies", "ous", "s")

_PATH_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "dispatch": ("dispatcher", "strategy", "worker"),
    "execute": ("worker", "runner"),
    "executing": ("worker", "runner"),
    "task": ("worker", "strategy"),
    "tasks": ("task", "worker", "strategy"),
    "runtime": ("scheduler",),
}

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
        for expansion in _PATH_TERM_EXPANSIONS.get(t, ()):
            if expansion not in seen:
                seen.add(expansion)
                terms.append(expansion)
        for suffix in _STEM_SUFFIXES:
            if t.endswith(suffix) and len(t) - len(suffix) >= 4:
                stem = t[: -len(suffix)]
                if stem not in seen:
                    seen.add(stem)
                    terms.append(stem)
    terms.sort(key=len, reverse=True)
    return terms


def _path_match_multiplier(file_path: str, term: str) -> float:
    """Score a path keyword by specificity of the file-path match."""
    import re

    lowered = file_path.lower()
    term_lower = term.lower()
    basename = lowered.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    segments = {segment for segment in re.split(r"[/_.-]+", lowered) if segment}
    if term_lower in (stem, basename):
        return 0.85
    if term_lower in segments:
        return 0.70
    return 0.45


_RETRIEVAL_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "query": ("search", "retrieve", "retrieval", "context"),
    "pipeline": ("workflow", "stage", "assembly", "context"),
    "retrieval": ("search", "rank", "score", "bm25", "context"),
}

_QUERY_PIPELINE_EXPANSIONS = ("bm25", "BM25Index", "assemble_context")


def _expand_retrieval_question(question: str) -> str:
    """Add code-level terms for retrieval architecture queries.

    Natural-language architecture questions often use product terms such as
    "query pipeline" while the implementation files use algorithm and assembly
    names. Expand only from known retrieval terms so ordinary symbol lookups do
    not inherit broad search vocabulary.
    """
    import re

    raw_terms = {w.lower() for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", question)}
    expansions: list[str] = []
    for term, candidates in _RETRIEVAL_QUERY_EXPANSIONS.items():
        if term not in raw_terms:
            continue
        expansions.extend(candidates)

    if {"query", "pipeline"} <= raw_terms:
        expansions.extend(_QUERY_PIPELINE_EXPANSIONS)

    if not expansions:
        return question

    existing = {w.lower() for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", question)}
    ordered_unique: list[str] = []
    for term in expansions:
        lowered = term.lower()
        if lowered in existing:
            continue
        existing.add(lowered)
        ordered_unique.append(term)

    if not ordered_unique:
        return question
    return f"{question} {' '.join(ordered_unique)}"


def _file_path_boost(
    store: IndexStore,
    question: str,
    existing_ids: set[str],
    max_bm25_score: float = 1.0,
    max_boost_chunks: int = 30,
    max_chunks_per_file: int = 3,
) -> list[tuple[CodeChunk, float]]:
    """Find chunks whose file_path contains query terms as exact substrings.

    Terms are searched longest-first (from _extract_path_terms) so specific terms
    like "validators" get priority over generic ones like "pydantic". Boost score
    is max BM25 multiplied by _path_match_multiplier: 0.85 for stem/basename
    matches, 0.70 for path segment matches, and 0.45 for substring matches.
    Per-file caps prevent one matching directory from flooding file-level aggregation.
    """
    terms = _extract_path_terms(question)
    boosted: list[tuple[CodeChunk, float]] = []
    seen: set[str] = set(existing_ids)
    boosted_by_file: dict[str, int] = {}

    for term in terms:
        for chunk in store.search_chunks_by_path_keyword(term, limit=20):
            if boosted_by_file.get(chunk.file_path, 0) >= max_chunks_per_file:
                continue
            if chunk.id not in seen:
                seen.add(chunk.id)
                boost_score = max_bm25_score * _path_match_multiplier(chunk.file_path, term)
                boosted.append((chunk, boost_score))
                boosted_by_file[chunk.file_path] = boosted_by_file.get(chunk.file_path, 0) + 1
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


def _module_prefilter_boosts(
    modules: list[Module],
    all_chunks: list[CodeChunk],
    question: str,
    existing_ids: set[str],
    max_bm25_score: float,
    *,
    max_modules: int = 3,
    max_chunks_per_module: int = 8,
) -> list[tuple[CodeChunk, float]]:
    if not modules:
        return []

    import sqlite3

    from archex.index.bm25 import escape_fts_query

    escaped = escape_fts_query(question)
    if not escaped:
        return []

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE module_fts USING fts5("
            "ordinal UNINDEXED, name, responsibility, tokenize='porter unicode61')"
        )
        conn.executemany(
            "INSERT INTO module_fts (ordinal, name, responsibility) VALUES (?, ?, ?)",
            [
                (idx, module.name, module.responsibility or "")
                for idx, module in enumerate(modules)
                if module.responsibility
            ],
        )
        rows = conn.execute(
            "SELECT ordinal, bm25(module_fts, 2.0, 8.0) AS score "
            "FROM module_fts WHERE module_fts MATCH ? ORDER BY score LIMIT ?",
            (escaped, max_modules),
        ).fetchall()
        if not rows and " AND " in escaped:
            rows = conn.execute(
                "SELECT ordinal, bm25(module_fts, 2.0, 8.0) AS score "
                "FROM module_fts WHERE module_fts MATCH ? ORDER BY score LIMIT ?",
                (escaped.replace(" AND ", " OR "), max_modules),
            ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    chunks_by_file: dict[str, list[CodeChunk]] = {}
    for chunk in all_chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    seen = set(existing_ids)
    boosts: list[tuple[CodeChunk, float]] = []
    for rank, (ordinal, raw_score) in enumerate(rows):
        module = modules[int(ordinal)]
        score = max_bm25_score * (0.45 / (rank + 1))
        if float(raw_score) == 0.0:
            score *= 0.5
        added_for_module = 0
        for file_path in module.files:
            for chunk in chunks_by_file.get(file_path, []):
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                boosts.append((chunk, score))
                added_for_module += 1
                if added_for_module >= max_chunks_per_module:
                    break
            if added_for_module >= max_chunks_per_module:
                break

    return boosts


def _bm25_search_with_boosts(
    bm25: BM25Index,
    store: IndexStore,
    question: str,
    top_k: int,
    all_chunks: list[CodeChunk],
    modules: list[Module],
    index_config: IndexConfig,
) -> tuple[
    list[tuple[CodeChunk, float]],
    list[tuple[CodeChunk, float]],
    list[tuple[CodeChunk, float]],
    list[tuple[CodeChunk, float]],
]:
    """Run BM25 search plus candidate-pool boosts.

    Returns retrieval legs as separate lists so
    callers can record individual counts for observability, then combine them.
    """
    expanded_question = _expand_retrieval_question(question)
    results = bm25.search(expanded_question, top_k=top_k)
    bm25_ids = {c.id for c, _ in results}
    max_bm25 = max((s for _, s in results), default=1.0)
    path_boost = _file_path_boost(
        store,
        expanded_question,
        bm25_ids,
        max_bm25_score=max_bm25,
    )
    all_existing = bm25_ids | {c.id for c, _ in path_boost}
    symbol_seeds = [
        (c, s)
        for c, s in _symbol_search_seeds(store, expanded_question, max_bm25_score=max_bm25)
        if c.id not in all_existing
    ]
    module_boost = _module_prefilter_boosts(
        modules if index_config.module_prefilter else [],
        all_chunks,
        expanded_question,
        all_existing | {c.id for c, _ in symbol_seeds},
        max_bm25_score=max_bm25,
    )
    return results, path_boost, symbol_seeds, module_boost


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
        config = load_config(source)
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

        decisions = infer_decisions(patterns)

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
        config = load_config(source)
    if index_config is None:
        index_config = load_index_config(source)
    _bootstrap_plugins()

    t0 = time.perf_counter()
    if trace is not None:
        trace.operation = "query"
        trace.start_ns = time.perf_counter_ns()
    cache = _cache_manager_for_source(source, config)
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

                if graph.file_edge_count == 0 and graph.file_count > 1:
                    co_dir_added = graph.add_co_directory_edges()
                    logger.info(
                        "Added %d co-directory edges (cached index had 0 edges)", co_dir_added
                    )

                top_k = _compute_top_k(chunk_count)
                vector_top_k = _compute_vector_top_k(
                    bm25_top_k=top_k,
                    total_chunks=chunk_count,
                )
                splade_top_k = _compute_splade_top_k(
                    bm25_top_k=top_k,
                    total_chunks=chunk_count,
                )
                # Pre-load all chunks into memory before parallel search so the
                # vector thread has no dependency on the SQLite store connection.
                all_chunks_cached = store.get_chunks()
                surrogate_lookup = _surrogate_lookup(store, all_chunks_cached, index_config)
                modules_cached = _modules_or_raise(store, index_config)

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
                        vector_top_k,
                    )
                    if index_config.bm25:
                        (
                            _bm25_raw,
                            path_boost,
                            symbol_seeds,
                            module_boost,
                        ) = _bm25_search_with_boosts(
                            bm25,
                            store,
                            question,
                            top_k,
                            all_chunks_cached,
                            modules_cached,
                            index_config,
                        )
                        search_results = _bm25_raw + path_boost + symbol_seeds + module_boost
                    else:
                        search_results = []
                        path_boost = []
                        symbol_seeds = []
                        module_boost = []
                    splade_results = _splade_search_or_raise(
                        store,
                        question,
                        index_config,
                        splade_top_k,
                    )
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
                            top_k=vector_top_k,
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
                                "module_boost": len(module_boost),
                                "splade_results": len(splade_results or []),
                                "top_k": top_k,
                                "vector_top_k": vector_top_k,
                                "splade_top_k": splade_top_k,
                            },
                        )
                    )
                query_avg_idf = (
                    bm25.avg_idf(question)
                    if vector_results is not None or splade_results is not None
                    else None
                )
                # Cross-encoder reranker: enabled only when index_config.rerank is set.
                _reranker = _maybe_reranker(index_config)
                bundle = assemble_context(
                    search_results=search_results,
                    graph=graph,
                    all_chunks=all_chunks_cached,
                    question=question,
                    token_budget=effective_budget,
                    vector_results=vector_results,  # type: ignore[arg-type]
                    splade_results=splade_results,
                    scoring_weights=scoring_weights,
                    trace=trace,
                    avg_idf=query_avg_idf,
                    reranker=_reranker,
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

        # Fallback for languages where import resolution fails (Go, Rust):
        # add co-directory edges so graph expansion has something to work with.
        if graph.file_edge_count == 0 and graph.file_count > 1:
            co_dir_added = graph.add_co_directory_edges()
            logger.info(
                "Added %d co-directory edges (import resolution produced 0 edges)",
                co_dir_added,
            )

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
            _build_splade_index(store, all_chunks, index_config)
            modules_miss = _build_module_summaries(store, graph, parsed_files, index_config)

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
            vector_top_k = _compute_vector_top_k(
                bm25_top_k=top_k,
                total_chunks=len(all_chunks),
            )
            splade_top_k = _compute_splade_top_k(
                bm25_top_k=top_k,
                total_chunks=len(all_chunks),
            )

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
                _set_working_tree_signature(store, repo_path, config)
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
                    vector_top_k,
                )
                if index_config.bm25:
                    (
                        _bm25_raw,
                        path_boost,
                        symbol_seeds_miss,
                        module_boost_miss,
                    ) = _bm25_search_with_boosts(
                        bm25,
                        store,
                        question,
                        top_k,
                        all_chunks,
                        modules_miss,
                        index_config,
                    )
                    search_results = _bm25_raw + path_boost + symbol_seeds_miss + module_boost_miss
                else:
                    search_results = []
                    path_boost = []
                    symbol_seeds_miss = []
                    module_boost_miss = []
                splade_results_miss = _splade_search_or_raise(
                    store,
                    question,
                    index_config,
                    splade_top_k,
                )
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
                        top_k=vector_top_k,
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
                            "module_boost": len(module_boost_miss),
                            "splade_results": len(splade_results_miss or []),
                            "top_k": top_k,
                            "vector_top_k": vector_top_k,
                            "splade_top_k": splade_top_k,
                        },
                    )
                )
            query_avg_idf_miss = (
                bm25.avg_idf(question)
                if vector_results_miss is not None or splade_results_miss is not None
                else None
            )
            _reranker_miss = _maybe_reranker(index_config)
            bundle = assemble_context(
                search_results=search_results,
                graph=graph,
                all_chunks=all_chunks,
                question=question,
                token_budget=effective_budget,
                vector_results=vector_results_miss,  # type: ignore[arg-type]
                splade_results=splade_results_miss,
                scoring_weights=scoring_weights,
                trace=trace,
                avg_idf=query_avg_idf_miss,
                reranker=_reranker_miss,
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
    return precision_api.file_tree(
        source,
        max_depth=max_depth,
        language=language,
        config=config,
        timing=timing,
        ensure_index=_ensure_index,
    )


def file_outline(
    source: RepoSource,
    file_path: str,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> FileOutline:
    """Return the symbol hierarchy for a single file — no source code."""
    return precision_api.file_outline(
        source,
        file_path=file_path,
        config=config,
        timing=timing,
        ensure_index=_ensure_index,
    )


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
    return precision_api.search_symbols(
        source,
        query=query,
        kind=kind,
        language=language,
        limit=limit,
        config=config,
        timing=timing,
        ensure_index=_ensure_index,
    )


def get_symbol(
    source: RepoSource,
    symbol_id: str,
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> SymbolSource | None:
    """Retrieve the full source code of a single symbol by its stable ID."""
    return precision_api.get_symbol(
        source,
        symbol_id=symbol_id,
        config=config,
        timing=timing,
        ensure_index=_ensure_index,
    )


def get_symbols_batch(
    source: RepoSource,
    symbol_ids: list[str],
    config: Config | None = None,
    timing: PipelineTiming | None = None,
) -> list[SymbolSource | None]:
    """Batch retrieve N symbols by their stable IDs. Preserves input order."""
    return precision_api.get_symbols_batch(
        source,
        symbol_ids=symbol_ids,
        config=config,
        timing=timing,
        ensure_index=_ensure_index,
    )


# ---------------------------------------------------------------------------
# Token efficiency utilities
# ---------------------------------------------------------------------------


def get_repo_total_tokens(
    source: RepoSource,
    config: Config | None = None,
) -> int:
    """Return the total token count across all indexed chunks for a repository."""
    return precision_api.get_repo_total_tokens(source, config=config, ensure_index=_ensure_index)


def get_file_token_count(
    source: RepoSource,
    file_path: str,
    config: Config | None = None,
) -> int:
    """Return the total token count for a single file in an indexed repository."""
    return precision_api.get_file_token_count(
        source,
        file_path=file_path,
        config=config,
        ensure_index=_ensure_index,
    )


def get_files_token_count(
    source: RepoSource,
    file_paths: list[str],
    config: Config | None = None,
) -> int:
    """Return the total token count across unique files in an indexed repository."""
    return precision_api.get_files_token_count(
        source,
        file_paths=file_paths,
        config=config,
        ensure_index=_ensure_index,
    )
