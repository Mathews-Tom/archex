"""ContextBundle assembly: retrieve, rank, and assemble chunks into a ContextBundle."""

from __future__ import annotations

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

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph

_TYPE_LIKE = {SymbolKind.CLASS, SymbolKind.TYPE, SymbolKind.INTERFACE}

NEIGHBOR_DECAY = 0.35

MIN_SCORE_RATIO = 0.30

MAX_EXPANSION_FILES = 8

MAX_FILES = 8


def _estimate_tokens(chunk: CodeChunk) -> int:
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
    total_tokens = sum(_estimate_tokens(c) for c in all_chunks)
    included = [
        RankedChunk(chunk=chunk, relevance_score=1.0, final_score=1.0)
        for chunk in all_chunks
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


def assemble_context(
    search_results: list[tuple[CodeChunk, float]],
    graph: DependencyGraph,
    all_chunks: list[CodeChunk],
    question: str,
    token_budget: int = 8192,
    vector_results: list[tuple[CodeChunk, float]] | None = None,
    scoring_weights: ScoringWeights | None = None,
    modules: list[Module] | None = None,
) -> ContextBundle:
    """Assemble a token-budgeted ContextBundle from search results and a dependency graph.

    When vector_results is provided, uses Reciprocal Rank Fusion to merge BM25 and
    vector results before scoring.
    When modules is provided, computes cohesion signal per chunk.
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

    # Merge BM25 + vector via RRF when both are available
    if vector_results:
        from archex.index.vector import reciprocal_rank_fusion

        merged = reciprocal_rank_fusion(search_results, vector_results, k=60)
        max_score = max(score for _, score in merged) or 1.0
        bm25_by_id: dict[str, float] = {chunk.id: score / max_score for chunk, score in merged}
    else:
        # Normalize BM25 scores to [0, 1]
        max_score = max(score for _, score in search_results) or 1.0
        bm25_by_id = {chunk.id: score / max_score for chunk, score in search_results}
    all_results = search_results + (vector_results or [])
    seed_files: set[str] = {chunk.file_path for chunk, _ in all_results}

    candidates_found = len(search_results)

    # Expand: find neighbor files via graph, prioritized by penalized seed score
    seed_file_scores: dict[str, float] = {}
    for chunk, score in search_results:
        fp = chunk.file_path
        is_test = fp.startswith("test") or "/test" in fp
        effective = score * (0.6 if is_test else 1.0)
        seed_file_scores[fp] = max(seed_file_scores.get(fp, 0.0), effective)

    neighbor_files: set[str] = set()
    expansion_priority: dict[str, float] = {}
    for file_path in seed_files:
        seed_score = seed_file_scores.get(file_path, 0.0)
        for nb in graph.neighborhood(file_path, hops=1):
            neighbor_files.add(nb)
            expansion_priority[nb] = max(expansion_priority.get(nb, 0.0), seed_score)
    expansion_files = neighbor_files - seed_files
    sorted_expansion = sorted(expansion_files, key=lambda f: -expansion_priority.get(f, 0.0))

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
    expansion_files_added = 0
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
        if expansion_files_added >= MAX_EXPANSION_FILES:
            break

    candidates_after_expansion = len(candidate_map)

    # Get structural centrality scores
    centrality = graph.structural_centrality()

    # Build file-to-module mapping for cohesion signal
    file_to_module: dict[str, Module] = {}
    if modules:
        for mod in modules:
            for fp in mod.files:
                file_to_module[fp] = mod

    # Compute signal agreement (Jaccard of BM25 top-K and vector top-K)
    signal_agreement: float | None = None
    if vector_results:
        k = 20
        bm25_top_k = {chunk.file_path for chunk, _ in search_results[:k]}
        vec_top_k = {chunk.file_path for chunk, _ in vector_results[:k]}
        union = bm25_top_k | vec_top_k
        if union:
            signal_agreement = len(bm25_top_k & vec_top_k) / len(union)

    # Candidate file set for cohesion computation
    candidate_files = {c.file_path for c in candidate_map.values()}

    # Propagate BM25 relevance to graph-expansion neighbors
    neighbor_boost: dict[str, float] = {}
    for file_path in seed_files:
        seed_score = max(
            (bm25_by_id.get(c.id, 0.0) for c in candidate_map.values() if c.file_path == file_path),
            default=0.0,
        )
        for nb in graph.neighborhood(file_path, hops=1) - seed_files:
            neighbor_boost[nb] = max(neighbor_boost.get(nb, 0.0), seed_score * NEIGHBOR_DECAY)

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

    # File-level cap: keep only top files by aggregated score
    file_agg: dict[str, float] = {}
    for rc in ranked:
        fp = rc.chunk.file_path
        file_agg[fp] = file_agg.get(fp, 0.0) + rc.final_score
    top_files = {
        fp for fp, _ in sorted(file_agg.items(), key=lambda x: -x[1])[:MAX_FILES]
    }
    ranked = [rc for rc in ranked if rc.chunk.file_path in top_files]

    # Greedy bin-packing within token budget with score cutoff
    included: list[RankedChunk] = []
    total_tokens = 0
    score_floor = ranked[0].final_score * MIN_SCORE_RATIO if ranked else 0.0
    for rc in ranked:
        if rc.final_score < score_floor:
            break
        tokens = _estimate_tokens(rc.chunk)
        if total_tokens + tokens > token_budget:
            continue
        included.append(rc)
        total_tokens += tokens

    chunks_dropped = len(ranked) - len(included)
    truncated = chunks_dropped > 0

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

    assembly_ms = (time.perf_counter() - assembly_start) * 1000

    meta = RetrievalMetadata(
        candidates_found=candidates_found,
        candidates_after_expansion=candidates_after_expansion,
        chunks_included=len(included),
        chunks_dropped=chunks_dropped,
        strategy=strategy,
        assembly_time_ms=assembly_ms,
        signal_agreement=signal_agreement,
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
