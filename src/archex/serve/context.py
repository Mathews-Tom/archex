"""ContextBundle assembly: retrieve, rank, and assemble chunks into a ContextBundle."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from archex.models import (
    CodeChunk,
    ContextBundle,
    RankedChunk,
    RetrievalMetadata,
    StructuralContext,
    SymbolKind,
    TypeDefinition,
)

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph

_TYPE_LIKE = {SymbolKind.CLASS, SymbolKind.TYPE, SymbolKind.INTERFACE}


def _estimate_tokens(chunk: CodeChunk) -> int:
    if chunk.token_count > 0:
        return chunk.token_count
    return int(len(chunk.content.split()) * 1.3)


def assemble_context(
    search_results: list[tuple[CodeChunk, float]],
    graph: DependencyGraph,
    all_chunks: list[CodeChunk],
    question: str,
    token_budget: int = 8192,
) -> ContextBundle:
    """Assemble a token-budgeted ContextBundle from BM25 results and a dependency graph."""
    assembly_start = time.perf_counter()

    if not search_results:
        return ContextBundle(
            query=question,
            token_budget=token_budget,
            retrieval_metadata=RetrievalMetadata(strategy="bm25+graph"),
        )

    # Normalize BM25 scores to [0, 1]
    max_score = max(score for _, score in search_results) or 1.0
    bm25_by_id: dict[str, float] = {chunk.id: score / max_score for chunk, score in search_results}
    seed_files: set[str] = {chunk.file_path for chunk, _ in search_results}

    candidates_found = len(search_results)

    # Expand: find neighbor files via graph
    neighbor_files: set[str] = set()
    for file_path in seed_files:
        neighbor_files |= graph.neighborhood(file_path, hops=1)
    expansion_files = neighbor_files - seed_files

    # Build chunk lookup by file
    chunks_by_file: dict[str, list[CodeChunk]] = {}
    for chunk in all_chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    # Collect candidate chunks (seed + expansion), dedup by id
    candidate_map: dict[str, CodeChunk] = {}
    for chunk, _ in search_results:
        candidate_map[chunk.id] = chunk
    for file_path in expansion_files:
        for chunk in chunks_by_file.get(file_path, []):
            if chunk.id not in candidate_map:
                candidate_map[chunk.id] = chunk

    candidates_after_expansion = len(candidate_map)

    # Get structural centrality scores
    centrality = graph.structural_centrality()

    # Build RankedChunks
    ranked: list[RankedChunk] = []
    for chunk in candidate_map.values():
        relevance = bm25_by_id.get(chunk.id, 0.0)
        structural = centrality.get(chunk.file_path, 0.0)
        type_coverage = 0.5 if chunk.symbol_kind in _TYPE_LIKE else 0.0
        final = 0.6 * relevance + 0.3 * structural + 0.1 * type_coverage
        ranked.append(
            RankedChunk(
                chunk=chunk,
                relevance_score=relevance,
                structural_score=structural,
                type_coverage_score=type_coverage,
                final_score=final,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)

    # Greedy bin-packing within token budget
    included: list[RankedChunk] = []
    total_tokens = 0
    for rc in ranked:
        tokens = _estimate_tokens(rc.chunk)
        if total_tokens + tokens > token_budget:
            break
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
        strategy="bm25+graph",
        assembly_time_ms=assembly_ms,
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
