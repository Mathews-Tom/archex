"""ContextBundle assembly: retrieve, rank, and assemble chunks into a ContextBundle."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from archex.models import (
    CodeChunk,
    ContextBundle,
    Module,
    RankedChunk,
    RetrievalMetadata,
    ScoringWeights,
    StructuralContext,
    SymbolKind,
    TypeDefinition,
)
from archex.observe import PipelineTrace, StepTiming

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph

logger = logging.getLogger(__name__)

_TYPE_LIKE = {SymbolKind.CLASS, SymbolKind.TYPE, SymbolKind.INTERFACE}

# Direct imports (files the seed depends on) get a strong boost — same call chain.
IMPORT_TARGET_DECAY = 0.65

# Importers (files that depend on the seed) get a weaker boost — consumers, not deps.
IMPORTER_DECAY = 0.20

MIN_SCORE_RATIO = 0.30

MAX_EXPANSION_FILES = 5

MAX_FILES = 8

# Seeds below this fraction of max normalized seed score do not trigger expansion.
# Lowered from 0.10 — large repos have flatter BM25 distributions where
# 10% excluded valid seeds, leaving graph expansion permanently inert.
SEED_EXPANSION_MIN = 0.05

# Files below this fraction of the top file's aggregate score are excluded.
FILE_SCORE_CUTOFF = 0.15


def estimate_tokens(chunk: CodeChunk) -> int:
    if chunk.token_count > 0:
        return chunk.token_count
    return int(len(chunk.content.split()) * 1.3)


def passthrough_context(
    all_chunks: list[CodeChunk],
    question: str,
    token_budget: int,
) -> ContextBundle:
    """Return all chunks directly when total tokens fit within budget.

    Skips BM25/scoring overhead for small repos where retrieval adds no value.
    """
    assembly_start = time.perf_counter()
    total_tokens = sum(estimate_tokens(c) for c in all_chunks)
    included = [
        RankedChunk(chunk=chunk, relevance_score=1.0, final_score=1.0) for chunk in all_chunks
    ]
    included_files = sorted({c.file_path for c in all_chunks})
    file_tree = "\n".join(included_files)
    structural_context = StructuralContext(file_tree=file_tree)

    type_defs: list[TypeDefinition] = []
    for rc in included:
        if rc.chunk.symbol_kind in _TYPE_LIKE:
            type_defs.append(
                TypeDefinition(
                    symbol=rc.chunk.symbol_name or rc.chunk.id,
                    file_path=rc.chunk.file_path,
                    start_line=rc.chunk.start_line,
                    end_line=rc.chunk.end_line,
                    content=rc.chunk.content,
                )
            )

    assembly_ms = (time.perf_counter() - assembly_start) * 1000

    meta = RetrievalMetadata(
        candidates_found=len(all_chunks),
        candidates_after_expansion=len(all_chunks),
        chunks_included=len(included),
        chunks_dropped=0,
        strategy="passthrough",
        assembly_time_ms=assembly_ms,
    )

    return ContextBundle(
        query=question,
        chunks=included,
        structural_context=structural_context,
        type_definitions=type_defs,
        token_count=total_tokens,
        token_budget=token_budget,
        truncated=False,
        retrieval_metadata=meta,
    )


_QUERY_STOP = frozenset(
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
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "show",
        "find",
    }
)

# Architecture-intent synonyms: map each architectural keyword to code-level equivalents.
# These expand BM25 misses caused by vocabulary gaps between natural-language queries
# and the actual identifiers/comments in source files.
_ARCH_SYNONYMS: dict[str, list[str]] = {
    "pipeline": ["workflow", "chain", "process", "pipe", "stage"],
    "middleware": ["handler", "interceptor", "filter", "hook"],
    "registry": ["register", "catalog", "factory", "provider"],
    "adapter": ["plugin", "connector", "driver", "bridge"],
    "injection": ["inject", "resolve", "depend", "wire"],
    "routing": ["route", "router", "dispatch", "endpoint", "path"],
    "indexing": ["index", "reindex", "delta", "catalog"],
    "dependency": ["depend", "resolve", "inject", "require"],
    "query": ["search", "retrieve", "lookup", "find", "fetch"],
    "session": ["connection", "pool", "client", "transport"],
    "hook": ["callback", "listener", "subscriber", "event"],
    "orm": ["model", "schema", "mapper", "table", "entity"],
    "task": ["job", "worker", "celery", "dispatch", "execute"],
    "runtime": ["scheduler", "executor", "loop", "spawn"],
}

# Architecture keywords that trigger 2-hop expansion.
_ARCH_KEYWORDS = frozenset(_ARCH_SYNONYMS.keys())


def _split_compound_token(token: str) -> list[str]:
    """Split a camelCase or snake_case token into component words.

    Pass the original mixed-case token; the caller lowercases all returned
    parts before use.  Returns the original token plus split components.
    Examples:
      "queryPipeline"  → ["queryPipeline", "query", "Pipeline"]
      "next_function"  → ["next_function", "next", "function"]
      "BM25Index"      → ["BM25Index", "BM25", "index"]
    """
    import re

    # Split snake_case by underscore
    if "_" in token:
        parts = [p for p in token.split("_") if p]
        return [token] + parts if len(parts) > 1 else [token]

    # Split camelCase / PascalCase by uppercase boundaries
    camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)", token)
    if len(camel_parts) > 1:
        return [token] + [p.lower() for p in camel_parts]

    return [token]


def _query_terms(question: str) -> set[str]:
    """Extract lowercased content words from a query for expansion prioritization.

    Enhancements over the basic word-extraction:
    - Splits camelCase and snake_case tokens into components so path matching
      works against both joined and split forms.
    - Keeps compound phrase forms (e.g. "dependency_injection") alongside
      individual words so expansion scoring can match on both.
    - Expands architectural-intent keywords to code-level synonyms to close
      vocabulary gaps between natural-language queries and source identifiers.
    """
    import re

    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", question)

    expanded: set[str] = set()
    normalized_tokens: list[str] = []
    for tok in raw_tokens:
        low = tok.lower()
        if low in _QUERY_STOP:
            continue
        # Pass the original (mixed-case) token so camelCase boundaries are visible,
        # then lowercase all returned parts before adding to the term set.
        parts = _split_compound_token(tok)
        for p in parts:
            p_low = p.lower()
            if p_low not in _QUERY_STOP and len(p_low) >= 3:
                expanded.add(p_low)
                normalized_tokens.append(p_low)

    # Add bigram compound forms for adjacent non-stop pairs (e.g. dependency + injection
    # → "dependency_injection") so they match identifiers that use this combined form.
    clean = [t for t in normalized_tokens if t not in _QUERY_STOP and len(t) >= 3]
    for i in range(len(clean) - 1):
        compound = f"{clean[i]}_{clean[i + 1]}"
        expanded.add(compound)

    # Architecture-intent synonym expansion
    for term in list(expanded):
        if term in _ARCH_SYNONYMS:
            expanded.update(_ARCH_SYNONYMS[term])

    return expanded


def _adaptive_max_files(
    file_scores: list[tuple[str, float]],
    default: int = MAX_FILES,
) -> int:
    """Reduce MAX_FILES when BM25 has clear score separation.

    When the top file score is >3x the median score, reduce to 5.
    When >5x, reduce to 4.
    Otherwise keep the default of 8.
    """
    if len(file_scores) < 3:
        return min(default, len(file_scores))
    scores = [s for _, s in file_scores]
    top = scores[0]
    median = scores[len(scores) // 2]
    if median <= 0:
        return default
    ratio = top / median
    if ratio > 5.0:
        return 4
    if ratio > 3.0:
        return 5
    return default


def _is_architecture_query(question: str) -> bool:
    """Return True if the query contains any architecture keyword."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _ARCH_KEYWORDS)


def assemble_context(
    search_results: list[tuple[CodeChunk, float]],
    graph: DependencyGraph,
    all_chunks: list[CodeChunk],
    question: str,
    token_budget: int = 8192,
    vector_results: list[tuple[CodeChunk, float]] | None = None,
    scoring_weights: ScoringWeights | None = None,
    modules: list[Module] | None = None,
    trace: PipelineTrace | None = None,
    expansion_min_override: float | None = None,
) -> ContextBundle:
    """Assemble a token-budgeted ContextBundle from search results and a dependency graph.

    When vector_results is provided, uses Reciprocal Rank Fusion to merge BM25 and
    vector results before scoring.
    When modules is provided, computes cohesion signal per chunk.
    When trace is provided, records step-level timings for graph_expansion, scoring,
    and assembly phases.
    When expansion_min_override is provided, uses it instead of SEED_EXPANSION_MIN
    for expansion gating (useful for architecture-broad queries with flat BM25 scores).
    """
    assembly_start = time.perf_counter()
    weights = scoring_weights or ScoringWeights()

    strategy = "hybrid+graph" if vector_results else "bm25+graph"

    if not search_results and not vector_results:
        return ContextBundle(
            query=question,
            token_budget=token_budget,
            retrieval_metadata=RetrievalMetadata(strategy=strategy),
        )

    # Merge BM25 + vector via confidence-weighted RRF when both are available
    fusion_bm25_weight: float | None = None
    fusion_vector_weight: float | None = None
    fusion_skipped = False
    fusion_skip_reason = ""
    bm25_cv_val: float | None = None
    if vector_results:
        from archex.index.vector import bm25_score_cv, confidence_weighted_rrf, should_fuse

        # Gate fusion: skip when BM25 is confident and signals agree.
        # Always fuse when BM25 is empty — vector is the only signal.
        fuse, fuse_reason = should_fuse(search_results, vector_results)
        bm25_cv_val = bm25_score_cv(search_results)

        if fuse or not search_results:
            # Compute signal_agreement (Jaccard of BM25 top-20 and vector top-20 file paths)
            _k_agree = 20
            _bm25_top_k = {chunk.file_path for chunk, _ in search_results[:_k_agree]}
            _vec_top_k = {chunk.file_path for chunk, _ in vector_results[:_k_agree]}
            _union = _bm25_top_k | _vec_top_k
            signal_agreement_pre: float = (
                len(_bm25_top_k & _vec_top_k) / len(_union) if _union else 0.0
            )

            merged, fusion_bm25_weight, fusion_vector_weight = confidence_weighted_rrf(
                search_results, vector_results, signal_agreement_pre, bm25_cv_val, k=60
            )
            max_score = max(score for _, score in merged) or 1.0
            bm25_by_id: dict[str, float] = {chunk.id: score / max_score for chunk, score in merged}
            logger.debug("Fusion applied: %s", fuse_reason)
        else:
            # BM25 is confident — skip fusion, use BM25 results only
            signal_agreement_pre = 0.0
            fusion_skipped = True
            fusion_skip_reason = fuse_reason
            strategy = "bm25+graph"  # downgrade strategy label
            max_score = max((score for _, score in search_results), default=1.0) or 1.0
            bm25_by_id = {chunk.id: score / max_score for chunk, score in search_results}
            logger.debug("Fusion skipped: %s", fuse_reason)
    else:
        signal_agreement_pre = 0.0
        # Normalize BM25 scores to [0, 1]
        max_score = max(score for _, score in search_results) or 1.0
        bm25_by_id = {chunk.id: score / max_score for chunk, score in search_results}
    # When fusion is skipped, exclude vector results from seeds to avoid noise
    effective_vector = vector_results if (vector_results and not fusion_skipped) else []
    all_results = search_results + effective_vector
    seed_files: set[str] = {chunk.file_path for chunk, _ in all_results}

    candidates_found = len(search_results)

    # Effective expansion threshold: caller override takes precedence over constant.
    effective_expansion_min = (
        expansion_min_override if expansion_min_override is not None else SEED_EXPANSION_MIN
    )

    # --- Graph expansion phase ---
    _expansion_start = time.perf_counter_ns()

    # Expand: follow directed imports from seed files, prioritized by seed score.
    # imports_of(file) = files this file depends on (high relevance — same call chain)
    # imported_by(file) = files that depend on this file (moderate relevance — consumers)
    seed_file_scores: dict[str, float] = {}
    for chunk, score in search_results:
        fp = chunk.file_path
        is_test = fp.startswith("test") or "/test" in fp
        effective = score * (0.6 if is_test else 1.0)
        seed_file_scores[fp] = max(seed_file_scores.get(fp, 0.0), effective)
    # Include vector-only seeds so they can gate expansion independently of BM25.
    if vector_results:
        for chunk, score in vector_results:
            fp = chunk.file_path
            if fp not in seed_file_scores:
                is_test = fp.startswith("test") or "/test" in fp
                effective = score * (0.6 if is_test else 1.0)
                seed_file_scores[fp] = effective

    # Normalize seed file scores to [0, 1] for expansion gating
    max_seed_score = max(seed_file_scores.values()) if seed_file_scores else 1.0
    norm_seed_scores = {fp: s / max_seed_score for fp, s in seed_file_scores.items()}

    # Extract query terms for path-aware import prioritization
    q_terms = _query_terms(question)

    # Determine whether this is an architecture query (enables 2-hop expansion)
    is_arch_query = _is_architecture_query(question)

    # Expansion diagnostic logging
    qualifying_seeds = [
        fp for fp in seed_files if norm_seed_scores.get(fp, 0.0) >= effective_expansion_min
    ]
    expansion_eligible_seeds = len(qualifying_seeds)
    logger.debug(
        "graph_expansion: %d/%d seed files qualify (threshold=%.3f)",
        len(qualifying_seeds),
        len(seed_files),
        effective_expansion_min,
    )
    for fp in qualifying_seeds:
        imports_of_count = len(list(graph.imports_of(fp)))
        imported_by_count = len(list(graph.imported_by(fp)))
        logger.debug(
            "  seed %s (norm_score=%.3f): imports_of=%d imported_by=%d",
            fp,
            norm_seed_scores[fp],
            imports_of_count,
            imported_by_count,
        )

    expansion_priority: dict[str, float] = {}
    for file_path in seed_files:
        # Only expand from seeds above the confidence threshold
        if norm_seed_scores.get(file_path, 0.0) < effective_expansion_min:
            continue
        seed_score = seed_file_scores.get(file_path, 0.0)
        # Direct imports get full seed score — they're in the same call chain
        for dep in graph.imports_of(file_path):
            if dep not in seed_files:
                # Boost imports whose file path matches a query term
                path_lower = dep.lower()
                path_match = any(t in path_lower for t in q_terms)
                priority = seed_score * (1.5 if path_match else 1.0)
                expansion_priority[dep] = max(
                    expansion_priority.get(dep, 0.0),
                    priority,
                )
        # Importers get half seed score — they're consumers, not dependencies
        for imp in graph.imported_by(file_path):
            if imp not in seed_files:
                path_lower = imp.lower()
                path_match = any(t in path_lower for t in q_terms)
                priority = seed_score * (0.75 if path_match else 0.5)
                expansion_priority[imp] = max(
                    expansion_priority.get(imp, 0.0),
                    priority,
                )

    total_expansion_candidates = len(expansion_priority)
    if total_expansion_candidates == 0:
        logger.warning(
            "graph_expansion: zero candidates found. seed_files=%s norm_scores=%s",
            list(seed_files),
            {fp: f"{norm_seed_scores.get(fp, 0.0):.3f}" for fp in seed_files},
        )
    else:
        logger.debug("graph_expansion: %d total expansion candidates", total_expansion_candidates)

    sorted_expansion = sorted(
        expansion_priority.keys(),
        key=lambda f: -expansion_priority[f],
    )

    # Build chunk lookup by file
    chunks_by_file: dict[str, list[CodeChunk]] = {}
    for chunk in all_chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    # Collect candidate chunks (seed + file-capped expansion), dedup by id
    # Cap per-file to prevent one large file from monopolizing the expansion budget.
    # Skip test files in expansion — they add noise without improving relevance.
    max_per_file = 3
    candidate_map: dict[str, CodeChunk] = {}
    for chunk, _ in search_results:
        candidate_map[chunk.id] = chunk
    # Vector-only seeds: add their chunks so they participate in scoring even when
    # BM25 returned nothing for that file.
    if vector_results:
        for chunk, _ in vector_results:
            if chunk.id not in candidate_map:
                candidate_map[chunk.id] = chunk
    expansion_files_added = 0
    hop1_files_added: list[str] = []
    for file_path in sorted_expansion:
        if file_path.startswith("test") or "/test" in file_path:
            continue
        added = 0
        for chunk in chunks_by_file.get(file_path, []):
            if chunk.id not in candidate_map:
                candidate_map[chunk.id] = chunk
                added += 1
                if added >= max_per_file:
                    break
        if added > 0:
            expansion_files_added += 1
            hop1_files_added.append(file_path)
        if expansion_files_added >= MAX_EXPANSION_FILES:
            break

    # 2-hop expansion for architecture queries: follow imports_of for top hop-1 candidates
    if is_arch_query and hop1_files_added:
        remaining_budget = MAX_EXPANSION_FILES - expansion_files_added
        if remaining_budget > 0:
            hop2_priority: dict[str, float] = {}
            for hop1_fp in hop1_files_added:
                hop1_score = expansion_priority.get(hop1_fp, 0.0)
                for dep in graph.imports_of(hop1_fp):
                    if dep not in seed_files and dep not in set(hop1_files_added):
                        path_lower = dep.lower()
                        path_match = any(t in path_lower for t in q_terms)
                        priority = hop1_score * (1.5 if path_match else 1.0)
                        hop2_priority[dep] = max(hop2_priority.get(dep, 0.0), priority)

            sorted_hop2 = sorted(hop2_priority.keys(), key=lambda f: -hop2_priority[f])
            hop2_added = 0
            for file_path in sorted_hop2:
                if file_path.startswith("test") or "/test" in file_path:
                    continue
                if hop2_added >= remaining_budget:
                    break
                added = 0
                for chunk in chunks_by_file.get(file_path, []):
                    if chunk.id not in candidate_map:
                        candidate_map[chunk.id] = chunk
                        added += 1
                        if added >= max_per_file:
                            break
                if added > 0:
                    hop2_added += 1
                    expansion_files_added += 1

            logger.debug(
                "graph_expansion 2-hop: arch_query=True, hop2_candidates=%d, hop2_added=%d",
                len(hop2_priority),
                hop2_added,
            )

    candidates_after_expansion = len(candidate_map)

    if trace is not None:
        trace.add_step(
            StepTiming(
                name="graph_expansion",
                start_ns=_expansion_start,
                end_ns=time.perf_counter_ns(),
                metadata={
                    "seed_files": len(seed_files),
                    "expansion_files_added": expansion_files_added,
                    "candidates_after_expansion": candidates_after_expansion,
                },
            )
        )

    # --- Scoring phase ---
    _scoring_start = time.perf_counter_ns()

    # Get structural centrality scores
    centrality = graph.structural_centrality()

    # Build file-to-module mapping for cohesion signal
    file_to_module: dict[str, Module] = {}
    if modules:
        for mod in modules:
            for fp in mod.files:
                file_to_module[fp] = mod

    # Signal agreement was computed pre-fusion; carry it forward for metadata
    signal_agreement: float | None = signal_agreement_pre if vector_results else None

    # Candidate file set for cohesion computation
    candidate_files = {c.file_path for c in candidate_map.values()}

    # Propagate BM25 relevance to import-expanded neighbors (directed).
    # Only propagate from seeds above the expansion confidence threshold.
    neighbor_boost: dict[str, float] = {}
    for file_path in seed_files:
        if norm_seed_scores.get(file_path, 0.0) < effective_expansion_min:
            continue
        seed_score = max(
            (bm25_by_id.get(c.id, 0.0) for c in candidate_map.values() if c.file_path == file_path),
            default=0.0,
        )
        # Direct imports get strong boost — same call chain
        for dep in graph.imports_of(file_path):
            if dep not in seed_files:
                path_lower = dep.lower()
                path_match = any(t in path_lower for t in q_terms)
                decay = IMPORT_TARGET_DECAY * (1.3 if path_match else 1.0)
                neighbor_boost[dep] = max(
                    neighbor_boost.get(dep, 0.0),
                    seed_score * decay,
                )
        # Importers get weaker boost — consumers, not dependencies
        for imp in graph.imported_by(file_path):
            if imp not in seed_files:
                neighbor_boost[imp] = max(
                    neighbor_boost.get(imp, 0.0),
                    seed_score * IMPORTER_DECAY,
                )

    # Build RankedChunks
    ranked: list[RankedChunk] = []
    for chunk in candidate_map.values():
        relevance = bm25_by_id.get(chunk.id, 0.0) or neighbor_boost.get(chunk.file_path, 0.0)
        structural = centrality.get(chunk.file_path, 0.0)
        type_coverage = 0.5 if chunk.symbol_kind in _TYPE_LIKE else 0.0

        # Cohesion signal: proportion of co-module files present * module cohesion
        cohesion = 0.0
        mod = file_to_module.get(chunk.file_path)
        if mod and mod.files:
            co_present = sum(1 for f in mod.files if f in candidate_files)
            cohesion = (co_present / len(mod.files)) * mod.cohesion_score

        # Test files are less likely to be the answer — mild penalty
        is_test = chunk.file_path.startswith("test") or "/test" in chunk.file_path
        test_penalty = 0.6 if is_test else 1.0

        final = (
            weights.relevance * relevance
            + weights.structural * structural
            + weights.type_coverage * type_coverage
            + weights.cohesion * cohesion
        ) * test_penalty
        ranked.append(
            RankedChunk(
                chunk=chunk,
                relevance_score=relevance,
                structural_score=structural,
                type_coverage_score=type_coverage,
                cohesion_score=cohesion,
                final_score=final,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)

    # File-level ranking: aggregate per-file scores, apply score-relative cutoff,
    # then hard-cap at adaptive MAX_FILES to limit tail noise.
    file_agg: dict[str, float] = {}
    for rc in ranked:
        fp = rc.chunk.file_path
        file_agg[fp] = file_agg.get(fp, 0.0) + rc.final_score
    sorted_files = sorted(file_agg.items(), key=lambda x: -x[1])
    top_file_score = sorted_files[0][1] if sorted_files else 0.0
    score_cutoff = top_file_score * FILE_SCORE_CUTOFF
    top_files: set[str] = set()
    adaptive_max = _adaptive_max_files(sorted_files)
    for fp, score in sorted_files[:adaptive_max]:
        if score < score_cutoff:
            break
        top_files.add(fp)
    ranked = [rc for rc in ranked if rc.chunk.file_path in top_files]

    # Greedy bin-packing within token budget with score cutoff
    included: list[RankedChunk] = []
    total_tokens = 0
    score_floor = ranked[0].final_score * MIN_SCORE_RATIO if ranked else 0.0
    for rc in ranked:
        if rc.final_score < score_floor:
            break
        tokens = estimate_tokens(rc.chunk)
        if total_tokens + tokens > token_budget:
            continue
        included.append(rc)
        total_tokens += tokens

    chunks_dropped = len(ranked) - len(included)
    truncated = chunks_dropped > 0

    if trace is not None:
        trace.add_step(
            StepTiming(
                name="scoring",
                start_ns=_scoring_start,
                end_ns=time.perf_counter_ns(),
                metadata={
                    "candidates_scored": candidates_after_expansion,
                    "files_selected": len(top_files),
                    "chunks_included": len(included),
                    "chunks_dropped": chunks_dropped,
                },
            )
        )

    # --- Assembly phase ---
    _assembly_start = time.perf_counter_ns()

    # Build StructuralContext
    included_files = sorted({rc.chunk.file_path for rc in included})
    file_tree = "\n".join(included_files)

    all_file_edges = graph.file_edges()
    included_file_set = set(included_files)
    dep_subgraph: dict[str, list[str]] = {}
    for edge in all_file_edges:
        if edge.source in included_file_set and edge.target in included_file_set:
            dep_subgraph.setdefault(edge.source, []).append(edge.target)

    structural_context = StructuralContext(
        file_tree=file_tree,
        file_dependency_subgraph=dep_subgraph,
    )

    # Collect TypeDefinitions from included chunks
    type_defs: list[TypeDefinition] = []
    for rc in included:
        if rc.chunk.symbol_kind in _TYPE_LIKE:
            type_defs.append(
                TypeDefinition(
                    symbol=rc.chunk.symbol_name or rc.chunk.id,
                    file_path=rc.chunk.file_path,
                    start_line=rc.chunk.start_line,
                    end_line=rc.chunk.end_line,
                    content=rc.chunk.content,
                )
            )

    if trace is not None:
        trace.add_step(
            StepTiming(
                name="assembly",
                start_ns=_assembly_start,
                end_ns=time.perf_counter_ns(),
                metadata={
                    "included_files": len(included_files),
                    "type_definitions": len(type_defs),
                    "total_tokens": total_tokens,
                },
            )
        )

    assembly_ms = (time.perf_counter() - assembly_start) * 1000

    meta = RetrievalMetadata(
        candidates_found=candidates_found,
        candidates_after_expansion=candidates_after_expansion,
        chunks_included=len(included),
        chunks_dropped=chunks_dropped,
        strategy=strategy,
        assembly_time_ms=assembly_ms,
        signal_agreement=signal_agreement,
        fusion_bm25_weight=fusion_bm25_weight,
        fusion_vector_weight=fusion_vector_weight,
        expansion_eligible_seeds=expansion_eligible_seeds,
        expansion_candidates_found=len(expansion_priority),
        expansion_files_added=expansion_files_added,
        fusion_skipped=fusion_skipped,
        fusion_skip_reason=fusion_skip_reason,
        bm25_cv=bm25_cv_val,
    )

    return ContextBundle(
        query=question,
        chunks=included,
        structural_context=structural_context,
        type_definitions=type_defs,
        token_count=total_tokens,
        token_budget=token_budget,
        truncated=truncated,
        retrieval_metadata=meta,
    )
