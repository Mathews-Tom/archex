"""Token-capped structural scout maps for two-step repository context."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from archex.graph_query import GraphQuery, GraphQueryError
from archex.models import CodeChunk, ContextReceipt, Module, RankedChunk, SymbolKind
from archex.reporting import count_tokens
from archex.serve.intent import QueryIntent, classify_intent

if TYPE_CHECKING:
    from archex.index.store import IndexStore

ScoutFormat = Literal["json", "markdown"]
ScoutHandleKind = Literal["file", "symbol", "chunk"]
ScoutFetchStrategy = Literal["chunk_first", "hybrid_fetch", "direct_query"]

DEFAULT_SCOUT_TOKEN_BUDGET = 1000
MIN_SCOUT_TOKEN_BUDGET = 64
DEFAULT_SCOUT_FILE_LIMIT = 12
DEFAULT_SCOUT_SYMBOLS_PER_FILE = 3
DEFAULT_SCOUT_MODULE_LIMIT = 6
DEFAULT_SCOUT_GRAPH_EDGE_LIMIT = 12
FILE_HANDLE_PREFIX = "file:"
SYMBOL_HANDLE_PREFIX = "symbol:"
CHUNK_HANDLE_PREFIX = "chunk:"

INTENT_FETCH_HANDLE_LIMITS: dict[QueryIntent, int] = {
    QueryIntent.DEFINITION_LOOKUP: 1,
    QueryIntent.ARCHITECTURE_BROAD: 4,
    QueryIntent.USAGE_SEARCH: 3,
    QueryIntent.DEBUGGING: 2,
    QueryIntent.CLI: 2,
    QueryIntent.GENERAL: 3,
}

INTENT_DIRECT_QUERY_FILE_LIMITS: dict[QueryIntent, int] = {
    QueryIntent.DEFINITION_LOOKUP: 2,
    QueryIntent.ARCHITECTURE_BROAD: 4,
    QueryIntent.USAGE_SEARCH: 4,
    QueryIntent.DEBUGGING: 3,
    QueryIntent.CLI: 3,
    QueryIntent.GENERAL: 4,
}

INTENT_FETCH_MAX_HANDLE_LIMITS: dict[QueryIntent, int] = {
    QueryIntent.DEFINITION_LOOKUP: 2,
    QueryIntent.ARCHITECTURE_BROAD: 9,
    QueryIntent.USAGE_SEARCH: 7,
    QueryIntent.DEBUGGING: 4,
    QueryIntent.CLI: 4,
    QueryIntent.GENERAL: 7,
}

INTENT_FETCH_SCORE_MASS_TARGETS: dict[QueryIntent, float] = {
    QueryIntent.DEFINITION_LOOKUP: 0.55,
    QueryIntent.ARCHITECTURE_BROAD: 0.92,
    QueryIntent.USAGE_SEARCH: 0.84,
    QueryIntent.DEBUGGING: 0.75,
    QueryIntent.CLI: 0.75,
    QueryIntent.GENERAL: 0.82,
}

INTENT_DIRECT_QUERY_FALLBACK_FILE_LIMITS: dict[QueryIntent, int] = {
    QueryIntent.DEFINITION_LOOKUP: 4,
    QueryIntent.ARCHITECTURE_BROAD: 12,
    QueryIntent.USAGE_SEARCH: 10,
    QueryIntent.DEBUGGING: 6,
    QueryIntent.CLI: 6,
    QueryIntent.GENERAL: 10,
}

INTENT_HYBRID_FILE_LIMITS: dict[QueryIntent, int] = {
    QueryIntent.DEFINITION_LOOKUP: 0,
    QueryIntent.ARCHITECTURE_BROAD: 2,
    QueryIntent.USAGE_SEARCH: 2,
    QueryIntent.DEBUGGING: 1,
    QueryIntent.CLI: 1,
    QueryIntent.GENERAL: 2,
}

INTENT_WEAK_COVERAGE_MULTIPLIERS: dict[QueryIntent, float] = {
    QueryIntent.DEFINITION_LOOKUP: 0.85,
    QueryIntent.ARCHITECTURE_BROAD: 0.95,
    QueryIntent.USAGE_SEARCH: 0.92,
    QueryIntent.DEBUGGING: 0.90,
    QueryIntent.CLI: 0.90,
    QueryIntent.GENERAL: 0.92,
}


class ScoutHandle(BaseModel):
    kind: ScoutHandleKind
    value: str


class ScoutBudget(BaseModel):
    token_budget: int
    token_count: int = 0
    truncated: bool = False
    omitted_files: int = 0
    omitted_symbols: int = 0
    omitted_modules: int = 0
    omitted_graph_edges: int = 0


class ScoutFile(BaseModel):
    path: str
    language: str
    lines: int
    symbol_count: int
    handle: str
    score: float = 0.0
    reason: str = "ranked"
    primary_chunk_handle: str | None = None
    primary_symbol_handle: str | None = None


class ScoutSymbol(BaseModel):
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    chunk_id: str
    file_handle: str
    chunk_handle: str
    signature: str | None = None
    visibility: str | None = None
    symbol_id: str | None = None
    symbol_handle: str | None = None
    score: float = 0.0


class ScoutModule(BaseModel):
    name: str
    root_path: str
    responsibility: str | None = None
    cohesion_score: float = 0.0
    file_count: int = 0
    relevant_files: list[str] = []
    file_handles: list[str] = []
    exports: list[str] = []
    score: float = 0.0


class ScoutGraphEdge(BaseModel):
    source: str
    target: str
    kind: str
    source_path: str | None = None
    target_path: str | None = None
    source_handle: str | None = None
    target_handle: str | None = None
    confidence: str
    confidence_score: float
    evidence: list[str] = []


class ScoutFetchPlan(BaseModel):
    handles: list[str] = []
    file_reasons: dict[str, str] = {}
    estimated_fetch_tokens: int = 0
    estimated_fetch_files: int = 0
    direct_query_tokens: int = 0
    direct_query_files: int = 0
    estimated_total_tokens: int = 0
    coverage_score_mass: float = 0.0
    target_score_mass: float = 0.0
    projected_precision: float = 0.0
    direct_query_precision: float = 0.0
    recommended_strategy: ScoutFetchStrategy = "chunk_first"
    guardrail_reason: str = ""


class ScoutResult(BaseModel):
    query: str
    ranked_files: list[ScoutFile] = []
    modules: list[ScoutModule] = []
    symbols: list[ScoutSymbol] = []
    graph: list[ScoutGraphEdge] = []
    budget: ScoutBudget
    fetch_plan: ScoutFetchPlan = ScoutFetchPlan()
    receipt: ContextReceipt | None = None


def file_handle(file_path: str) -> str:
    return f"{FILE_HANDLE_PREFIX}{file_path}"


def symbol_handle(symbol_id: str) -> str:
    return f"{SYMBOL_HANDLE_PREFIX}{symbol_id}"


def chunk_handle(chunk_id: str) -> str:
    return f"{CHUNK_HANDLE_PREFIX}{chunk_id}"


def parse_scout_handle(value: str) -> ScoutHandle | None:
    if value.startswith(FILE_HANDLE_PREFIX):
        return ScoutHandle(kind="file", value=value.removeprefix(FILE_HANDLE_PREFIX))
    if value.startswith(SYMBOL_HANDLE_PREFIX):
        return ScoutHandle(kind="symbol", value=value.removeprefix(SYMBOL_HANDLE_PREFIX))
    if value.startswith(CHUNK_HANDLE_PREFIX):
        return ScoutHandle(kind="chunk", value=value.removeprefix(CHUNK_HANDLE_PREFIX))
    return None


def normalize_symbol_lookup_handle(value: str) -> str:
    handle = parse_scout_handle(value)
    if handle is None:
        return value
    if handle.kind == "symbol":
        return handle.value
    return value


def assemble_scout_from_store(
    store: IndexStore,
    question: str,
    *,
    ranked_chunks: list[RankedChunk] | None = None,
    token_budget: int = DEFAULT_SCOUT_TOKEN_BUDGET,
    output_format: ScoutFormat = "markdown",
    file_limit: int = DEFAULT_SCOUT_FILE_LIMIT,
    symbols_per_file: int = DEFAULT_SCOUT_SYMBOLS_PER_FILE,
    module_limit: int = DEFAULT_SCOUT_MODULE_LIMIT,
    modules_override: list[Module] | None = None,
    graph_edge_limit: int = DEFAULT_SCOUT_GRAPH_EDGE_LIMIT,
    bundle_file_paths: list[str] | None = None,
    seed_file_paths: list[str] | None = None,
    expanded_file_paths: list[str] | None = None,
    direct_query_tokens: int = 0,
    direct_query_file_paths: list[str] | None = None,
) -> ScoutResult:
    """Build a deterministic no-body structural map from an indexed repository."""
    _validate_token_budget(token_budget)
    ranked = ranked_chunks or []
    score_by_chunk = _score_by_chunk(ranked)
    files, omitted_files = _rank_files(
        store,
        ranked,
        file_limit=file_limit,
        bundle_file_paths=bundle_file_paths or [],
        seed_file_paths=seed_file_paths or [],
        expanded_file_paths=expanded_file_paths or [],
    )
    chunks_by_file = _chunks_by_file(store, [item.path for item in files])
    _attach_primary_handles(files, chunks_by_file, score_by_chunk)
    symbols, omitted_symbols = _top_symbols(
        chunks_by_file, score_by_chunk, symbols_per_file=symbols_per_file
    )
    modules_source = modules_override if modules_override is not None else store.get_modules()
    modules, omitted_modules = _rank_modules(modules_source, files, module_limit=module_limit)
    graph_edges, omitted_graph_edges = _graph_sketch(store, files, edge_limit=graph_edge_limit)
    result = ScoutResult(
        query=question,
        ranked_files=files,
        modules=modules,
        symbols=symbols,
        graph=graph_edges,
        budget=ScoutBudget(
            token_budget=token_budget,
            omitted_files=omitted_files,
            omitted_symbols=omitted_symbols,
            omitted_modules=omitted_modules,
            omitted_graph_edges=omitted_graph_edges,
        ),
    )
    _finalize_scout_result(
        result,
        output_format=output_format,
        chunks_by_file=chunks_by_file,
        score_by_chunk=score_by_chunk,
        question=question,
        direct_query_tokens=direct_query_tokens,
        direct_query_file_paths=direct_query_file_paths or [],
    )
    return result


def enforce_scout_token_budget(result: ScoutResult, *, output_format: ScoutFormat) -> ScoutResult:
    """Trim the least-specific scout items until the rendered result fits its cap."""
    budget = result.budget.token_budget
    _validate_token_budget(budget)
    while True:
        token_count = _stable_rendered_token_count(result, output_format=output_format)
        if token_count <= budget:
            return result
        result.budget.truncated = True
        if result.graph:
            result.graph.pop()
            result.budget.omitted_graph_edges += 1
            continue
        if result.symbols:
            result.symbols.pop()
            result.budget.omitted_symbols += 1
            continue
        if result.modules:
            result.modules.pop()
            result.budget.omitted_modules += 1
            continue
        if result.ranked_files:
            result.ranked_files.pop()
            result.budget.omitted_files += 1
            continue
        if result.fetch_plan != ScoutFetchPlan():
            result.fetch_plan = ScoutFetchPlan()
            continue
        msg = f"Scout metadata exceeds token budget {budget}; minimum practical budget is higher"
        raise ValueError(msg)


def render_scout(result: ScoutResult, *, output_format: ScoutFormat = "markdown") -> str:
    if output_format == "json":
        return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if output_format == "markdown":
        return _render_markdown(result)
    raise ValueError(f"Unsupported scout format {output_format!r}")


def _finalize_scout_result(
    result: ScoutResult,
    *,
    output_format: ScoutFormat,
    chunks_by_file: dict[str, list[CodeChunk]],
    score_by_chunk: dict[str, float],
    question: str,
    direct_query_tokens: int,
    direct_query_file_paths: list[str],
) -> None:
    while True:
        previous = _scout_shape(result)
        result.fetch_plan = _build_fetch_plan(
            result.ranked_files,
            chunks_by_file,
            score_by_chunk,
            question=question,
            direct_query_tokens=direct_query_tokens,
            direct_query_file_paths=direct_query_file_paths,
            scout_tokens=result.budget.token_count,
        )
        enforce_scout_token_budget(result, output_format=output_format)
        result.fetch_plan = _build_fetch_plan(
            result.ranked_files,
            chunks_by_file,
            score_by_chunk,
            question=question,
            direct_query_tokens=direct_query_tokens,
            direct_query_file_paths=direct_query_file_paths,
            scout_tokens=result.budget.token_count,
        )
        if not (result.ranked_files or result.modules or result.symbols or result.graph):
            result.fetch_plan = ScoutFetchPlan()
            _stable_rendered_token_count(result, output_format=output_format)
            return
        if previous == _scout_shape(result):
            return


def _scout_shape(
    result: ScoutResult,
) -> tuple[int, int, int, int, tuple[tuple[str, str], ...]]:
    return (
        len(result.ranked_files),
        len(result.modules),
        len(result.symbols),
        len(result.graph),
        tuple(sorted(result.fetch_plan.file_reasons.items())),
    )


def _stable_rendered_token_count(result: ScoutResult, *, output_format: ScoutFormat) -> int:
    for _ in range(4):
        token_count = count_tokens(render_scout(result, output_format=output_format))
        if token_count == result.budget.token_count:
            return token_count
        result.budget.token_count = token_count
    return count_tokens(render_scout(result, output_format=output_format))


def _rank_files(
    store: IndexStore,
    ranked_chunks: list[RankedChunk],
    *,
    file_limit: int,
    bundle_file_paths: list[str],
    seed_file_paths: list[str],
    expanded_file_paths: list[str],
) -> tuple[list[ScoutFile], int]:
    metadata = {str(row["file_path"]): row for row in store.get_file_metadata()}
    aggregate_scores: dict[str, float] = defaultdict(float)
    max_scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for idx, ranked in enumerate(ranked_chunks):
        path = ranked.chunk.file_path
        score = ranked.final_score or ranked.relevance_score or ranked.structural_score
        aggregate_scores[path] += score
        max_scores[path] = max(score, max_scores.get(path, 0.0))
        first_rank.setdefault(path, idx)
    candidate_paths: list[str] = []
    seen: set[str] = set()
    for path in [
        *bundle_file_paths,
        *seed_file_paths,
        *expanded_file_paths,
        *aggregate_scores.keys(),
    ]:
        if path in metadata and path not in seen:
            candidate_paths.append(path)
            seen.add(path)
    if not candidate_paths:
        candidate_paths = sorted(metadata)
    bundle_rank = {path: idx for idx, path in enumerate(candidate_paths)}
    ordered_paths = sorted(
        candidate_paths,
        key=lambda path: (
            -_file_rank_score(path, aggregate_scores, max_scores, bundle_rank),
            bundle_rank.get(path, len(bundle_rank)),
            first_rank.get(path, len(first_rank)),
            path,
        ),
    )
    selected_paths = ordered_paths[:file_limit]
    files = [
        ScoutFile(
            path=path,
            language=str(metadata.get(path, {}).get("language", "unknown")),
            lines=int(metadata.get(path, {}).get("lines", 0)),
            symbol_count=int(metadata.get(path, {}).get("symbol_count", 0)),
            handle=file_handle(path),
            score=round(_file_rank_score(path, aggregate_scores, max_scores, bundle_rank), 6),
            reason=_file_reason(path, bundle_file_paths, seed_file_paths, expanded_file_paths),
        )
        for path in selected_paths
    ]
    return files, max(0, len(ordered_paths) - len(selected_paths))


def _file_rank_score(
    path: str,
    aggregate_scores: dict[str, float],
    max_scores: dict[str, float],
    bundle_rank: dict[str, int],
) -> float:
    aggregate = aggregate_scores.get(path, 0.0)
    maximum = max_scores.get(path, 0.0)
    rank_bonus = 1.0 / (bundle_rank[path] + 1) if path in bundle_rank else 0.0
    return aggregate + (maximum * 0.25) + rank_bonus


def _file_reason(
    path: str,
    bundle_file_paths: list[str],
    seed_file_paths: list[str],
    expanded_file_paths: list[str],
) -> str:
    if path in bundle_file_paths:
        return "query_bundle"
    if path in seed_file_paths:
        return "retrieval_seed"
    if path in expanded_file_paths:
        return "graph_expansion"
    return "file_tree"


def _chunks_by_file(store: IndexStore, file_paths: list[str]) -> dict[str, list[CodeChunk]]:
    chunks = store.get_chunks_for_files(file_paths)
    result: dict[str, list[CodeChunk]] = {path: [] for path in file_paths}
    for chunk in sorted(chunks, key=lambda item: (item.file_path, item.start_line, item.id)):
        result.setdefault(chunk.file_path, []).append(chunk)
    return result


def _score_by_chunk(ranked_chunks: list[RankedChunk]) -> dict[str, float]:
    return {
        ranked.chunk.id: ranked.final_score or ranked.relevance_score or ranked.structural_score
        for ranked in ranked_chunks
    }


def _sorted_chunks_for_scout(
    chunks: list[CodeChunk],
    score_by_chunk: dict[str, float],
) -> list[CodeChunk]:
    return sorted(
        chunks,
        key=lambda chunk: (
            -(score_by_chunk.get(chunk.id, 0.0)),
            0 if chunk.symbol_name and chunk.symbol_kind else 1,
            _visibility_rank(chunk.visibility),
            chunk.start_line,
            chunk.id,
        ),
    )


def _attach_primary_handles(
    files: list[ScoutFile],
    chunks_by_file: dict[str, list[CodeChunk]],
    score_by_chunk: dict[str, float],
) -> None:
    for item in files:
        chunks = _sorted_chunks_for_scout(chunks_by_file.get(item.path, []), score_by_chunk)
        if not chunks:
            continue
        chunk = chunks[0]
        item.primary_chunk_handle = chunk_handle(chunk.id)
        if chunk.symbol_id is not None:
            item.primary_symbol_handle = symbol_handle(str(chunk.symbol_id))


def _top_symbols(
    chunks_by_file: dict[str, list[CodeChunk]],
    score_by_chunk: dict[str, float],
    *,
    symbols_per_file: int,
) -> tuple[list[ScoutSymbol], int]:
    symbols: list[ScoutSymbol] = []
    omitted = 0
    for file_path in sorted(chunks_by_file):
        chunks = [
            chunk
            for chunk in _sorted_chunks_for_scout(chunks_by_file[file_path], score_by_chunk)
            if chunk.symbol_name and chunk.symbol_kind
        ]
        for chunk in chunks[:symbols_per_file]:
            symbol_id = str(chunk.symbol_id) if chunk.symbol_id else None
            symbols.append(
                ScoutSymbol(
                    name=chunk.qualified_name or chunk.symbol_name or chunk.id,
                    kind=chunk.symbol_kind or SymbolKind.FUNCTION,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_id=chunk.id,
                    file_handle=file_handle(chunk.file_path),
                    chunk_handle=chunk_handle(chunk.id),
                    signature=chunk.signature,
                    visibility=chunk.visibility,
                    symbol_id=symbol_id,
                    symbol_handle=symbol_handle(symbol_id) if symbol_id else None,
                    score=round(score_by_chunk.get(chunk.id, 0.0), 6),
                )
            )
        omitted += max(0, len(chunks) - symbols_per_file)
    return symbols, omitted


def _rank_modules(
    modules: list[Module],
    files: list[ScoutFile],
    *,
    module_limit: int,
) -> tuple[list[ScoutModule], int]:
    file_scores = {item.path: item.score for item in files}
    ranked: list[ScoutModule] = []
    for module in modules:
        relevant = sorted(path for path in module.files if path in file_scores)
        score = sum(file_scores[path] for path in relevant) + len(relevant) if relevant else 0.0
        relevant_files = relevant[:5]
        ranked.append(
            ScoutModule(
                name=module.name,
                root_path=module.root_path,
                responsibility=module.responsibility,
                cohesion_score=module.cohesion_score,
                file_count=module.file_count or len(module.files),
                relevant_files=relevant_files,
                file_handles=[file_handle(path) for path in relevant_files],
                exports=sorted(ref.qualified_name or ref.name for ref in module.exports)[:5],
                score=round(score, 6),
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.root_path, item.name))
    selected = ranked[:module_limit]
    return selected, max(0, len(ranked) - len(selected))


def _graph_sketch(
    store: IndexStore,
    files: list[ScoutFile],
    *,
    edge_limit: int,
) -> tuple[list[ScoutGraphEdge], int]:
    query = GraphQuery.from_store(store)
    edges: list[ScoutGraphEdge] = []
    seen: set[tuple[str, str, str, str]] = set()
    omitted = 0
    for file_item in files:
        if len(edges) >= edge_limit:
            omitted += 1
            continue
        try:
            neighborhood = query.neighbors(
                file_item.path, direction="both", depth=1, limit=edge_limit
            )
        except GraphQueryError:
            continue
        for edge in neighborhood.edges:
            key = (edge.source.id, edge.target.id, edge.type, edge.source.path or "")
            if key in seen:
                continue
            seen.add(key)
            if len(edges) >= edge_limit:
                omitted += 1
                continue
            edges.append(
                ScoutGraphEdge(
                    source=edge.source.id,
                    target=edge.target.id,
                    kind=edge.type,
                    source_path=edge.source.path,
                    target_path=edge.target.path,
                    source_handle=file_handle(edge.source.path) if edge.source.path else None,
                    target_handle=file_handle(edge.target.path) if edge.target.path else None,
                    confidence=edge.confidence,
                    confidence_score=edge.confidence_score,
                    evidence=edge.evidence[:2],
                )
            )
        omitted += neighborhood.omitted_edges
    edges.sort(key=lambda item: (item.source_path or "", item.target_path or "", item.kind))
    return edges[:edge_limit], omitted + max(0, len(edges) - edge_limit)


def _build_fetch_plan(
    files: list[ScoutFile],
    chunks_by_file: dict[str, list[CodeChunk]],
    score_by_chunk: dict[str, float],
    *,
    question: str,
    direct_query_tokens: int,
    direct_query_file_paths: list[str],
    scout_tokens: int,
) -> ScoutFetchPlan:
    intent = classify_intent(question)
    base_limit = INTENT_FETCH_HANDLE_LIMITS[intent]
    max_limit = INTENT_FETCH_MAX_HANDLE_LIMITS[intent]
    target_score_mass = INTENT_FETCH_SCORE_MASS_TARGETS[intent]
    direct_query_fallback_limit = INTENT_DIRECT_QUERY_FALLBACK_FILE_LIMITS[intent]
    hybrid_file_limit = INTENT_HYBRID_FILE_LIMITS[intent]
    weak_coverage_multiplier = INTENT_WEAK_COVERAGE_MULTIPLIERS[intent]
    eligible = [item for item in files if item.primary_symbol_handle or item.primary_chunk_handle]
    total_score = sum(max(item.score, 0.0) for item in eligible)
    selected: list[tuple[ScoutFile, str]] = []
    hybrid_handles: list[str] = []
    file_reasons: dict[str, str] = {}
    estimated_fetch_tokens = 0
    selected_scores: list[float] = []
    selected_score_mass = 0.0
    seen_handles: set[str] = set()
    selected_paths: set[str] = set()
    direct_query_rank = {path: idx for idx, path in enumerate(direct_query_file_paths)}

    for rank, item in enumerate(eligible, start=1):
        handle = item.primary_symbol_handle or item.primary_chunk_handle
        if handle is None:
            file_reasons[item.path] = f"no_precise_handle rank={rank} score={item.score:.3f}"
            continue
        if handle in seen_handles:
            file_reasons[item.path] = f"duplicate_handle rank={rank} handle={handle}"
            continue
        coverage_ratio = _coverage_ratio(selected_score_mass, total_score)
        should_select = len(selected) < base_limit or (
            len(selected) < max_limit and coverage_ratio < target_score_mass
        )
        if not should_select:
            file_reasons[item.path] = (
                f"pruned_by_cap rank={rank} score={item.score:.3f} "
                f"coverage={coverage_ratio:.3f} base_limit={base_limit} "
                f"max_limit={max_limit} target={target_score_mass:.3f}"
            )
            continue
        seen_handles.add(handle)
        selected_paths.add(item.path)
        selected.append((item, handle))
        score = max(item.score, 0.0)
        selected_scores.append(score)
        selected_score_mass += score
        representative = _representative_chunk(item.path, chunks_by_file, score_by_chunk)
        if representative is not None:
            estimated_fetch_tokens += _chunk_token_count(representative)
        file_reasons[item.path] = (
            f"selected_handle rank={len(selected)} score={item.score:.3f} "
            f"coverage={_coverage_ratio(selected_score_mass, total_score):.3f} "
            f"reason={item.reason} handle={handle}"
        )

    coverage_score_mass = _coverage_ratio(selected_score_mass, total_score)
    if coverage_score_mass < target_score_mass and hybrid_file_limit > 0:
        supplemental_candidates = sorted(
            (
                item
                for item in files
                if item.path not in selected_paths and item.path in direct_query_rank
            ),
            key=lambda item: (direct_query_rank[item.path], -item.score, item.path),
        )
        for item in supplemental_candidates[:hybrid_file_limit]:
            handle = file_handle(item.path)
            if handle in seen_handles:
                continue
            seen_handles.add(handle)
            selected_paths.add(item.path)
            hybrid_handles.append(handle)
            score = max(item.score, 0.0)
            selected_scores.append(score)
            selected_score_mass += score
            estimated_fetch_tokens += _file_token_count(item.path, chunks_by_file)
            file_reasons[item.path] = (
                f"selected_hybrid_file rank={len(selected) + len(hybrid_handles)} "
                f"score={item.score:.3f} coverage="
                f"{_coverage_ratio(selected_score_mass, total_score):.3f} "
                f"reason={item.reason} handle={handle}"
            )
        coverage_score_mass = _coverage_ratio(selected_score_mass, total_score)

    for item in files:
        if item.path not in file_reasons:
            file_reasons[item.path] = "missing_precise_handle"

    handles = [handle for _, handle in selected] + hybrid_handles
    estimated_fetch_files = len(selected_paths)
    direct_query_files = len(set(direct_query_file_paths))
    estimated_total_tokens = scout_tokens + estimated_fetch_tokens
    projected_precision = _projected_chunk_precision(files, selected_scores)
    direct_query_precision = 1.0 / direct_query_files if direct_query_files > 0 else 0.0
    recommended_strategy: ScoutFetchStrategy = "chunk_first"
    guardrail_reason = ""
    weak_coverage = coverage_score_mass < (target_score_mass * weak_coverage_multiplier)
    if not handles:
        recommended_strategy = "direct_query"
        guardrail_reason = "no_precise_fetch_handles"
    elif direct_query_files > 0 and direct_query_files <= INTENT_DIRECT_QUERY_FILE_LIMITS[intent]:
        recommended_strategy = "direct_query"
        guardrail_reason = "direct_query_already_narrow"
    elif weak_coverage and direct_query_files <= direct_query_fallback_limit:
        recommended_strategy = "direct_query"
        guardrail_reason = "projected_coverage_weak"
    elif direct_query_tokens > 0 and estimated_total_tokens >= direct_query_tokens:
        recommended_strategy = "direct_query"
        guardrail_reason = "estimated_total_not_better_than_query"
    elif (
        direct_query_precision >= projected_precision
        and direct_query_files <= direct_query_fallback_limit
    ):
        recommended_strategy = "direct_query"
        guardrail_reason = "direct_query_precision_proxy"
    elif hybrid_handles:
        recommended_strategy = "hybrid_fetch"
        guardrail_reason = "projected_coverage_thin"
    return ScoutFetchPlan(
        handles=handles,
        file_reasons=file_reasons,
        estimated_fetch_tokens=estimated_fetch_tokens,
        estimated_fetch_files=estimated_fetch_files,
        direct_query_tokens=direct_query_tokens,
        direct_query_files=direct_query_files,
        estimated_total_tokens=estimated_total_tokens,
        coverage_score_mass=coverage_score_mass,
        target_score_mass=target_score_mass,
        projected_precision=projected_precision,
        direct_query_precision=direct_query_precision,
        recommended_strategy=recommended_strategy,
        guardrail_reason=guardrail_reason,
    )


def _representative_chunk(
    file_path: str,
    chunks_by_file: dict[str, list[CodeChunk]],
    score_by_chunk: dict[str, float],
) -> CodeChunk | None:
    chunks = _sorted_chunks_for_scout(chunks_by_file.get(file_path, []), score_by_chunk)
    return chunks[0] if chunks else None


def _file_token_count(file_path: str, chunks_by_file: dict[str, list[CodeChunk]]) -> int:
    chunks = chunks_by_file.get(file_path, [])
    if not chunks:
        return 0
    return sum(_chunk_token_count(chunk) for chunk in chunks)


def _projected_chunk_precision(files: list[ScoutFile], selected_scores: list[float]) -> float:
    if not files or not selected_scores:
        return 0.0
    total_score = sum(max(item.score, 0.0) for item in files)
    if total_score <= 0.0:
        return 0.0
    return (sum(selected_scores) / total_score) / len(selected_scores)


def _coverage_ratio(selected_score_mass: float, total_score: float) -> float:
    if total_score <= 0.0:
        return 0.0
    return selected_score_mass / total_score


def _chunk_token_count(chunk: CodeChunk) -> int:
    if chunk.token_count > 0:
        return chunk.token_count
    return int(len(chunk.content.split()) * 1.3)


def _visibility_rank(visibility: str | None) -> int:
    if visibility == "public":
        return 0
    if visibility is None:
        return 1
    return 2


def _render_markdown(result: ScoutResult) -> str:
    lines = [
        "# archex scout",
        "",
        f"Query: {result.query}",
        f"Budget: {result.budget.token_count}/{result.budget.token_budget} tokens",
        "",
        "## Ranked files",
    ]
    if result.ranked_files:
        for item in result.ranked_files:
            lines.append(
                f"- {item.path} (`{item.handle}`, {item.language}, {item.lines} lines, "
                f"{item.symbol_count} symbols, score={item.score:.3f}, reason={item.reason})"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Module boundaries"])
    if result.modules:
        for item in result.modules:
            handles = ", ".join(item.file_handles) if item.file_handles else "no handles"
            lines.append(
                f"- {item.name} ({item.root_path}, files={item.file_count}, "
                f"cohesion={item.cohesion_score:.3f}); handles: {handles}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Top symbols"])
    if result.symbols:
        for item in result.symbols:
            handles = [item.file_handle, item.chunk_handle]
            if item.symbol_handle:
                handles.append(item.symbol_handle)
            lines.append(
                f"- {item.name} [{item.kind.value}] "
                f"{item.file_path}:{item.start_line}-{item.end_line} "
                f"handles: {', '.join(handles)}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Graph sketch"])
    if result.graph:
        for item in result.graph:
            source = item.source_path or item.source
            target = item.target_path or item.target
            source_handle = f" `{item.source_handle}`" if item.source_handle else ""
            target_handle = f" `{item.target_handle}`" if item.target_handle else ""
            evidence = f"; evidence: {'; '.join(item.evidence)}" if item.evidence else ""
            lines.append(
                f"- {source}{source_handle} --{item.kind}/{item.confidence}:"
                f"{item.confidence_score:.2f}--> {target}{target_handle}{evidence}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended fetch"])
    lines.append(f"- strategy: {result.fetch_plan.recommended_strategy}")
    lines.append(
        "- estimated_tokens: "
        f"fetch={result.fetch_plan.estimated_fetch_tokens}, "
        f"files={result.fetch_plan.estimated_fetch_files}, "
        f"total={result.fetch_plan.estimated_total_tokens}, "
        f"direct_query={result.fetch_plan.direct_query_tokens}/{result.fetch_plan.direct_query_files}"
    )
    lines.append(
        "- projected_precision: "
        f"chunk_first={result.fetch_plan.projected_precision:.3f}, "
        f"direct_query={result.fetch_plan.direct_query_precision:.3f}"
    )
    lines.append(
        "- projected_coverage: "
        f"chunk_first={result.fetch_plan.coverage_score_mass:.3f}, "
        f"target={result.fetch_plan.target_score_mass:.3f}"
    )
    if result.fetch_plan.guardrail_reason:
        lines.append(f"- guardrail: {result.fetch_plan.guardrail_reason}")
    if result.fetch_plan.handles:
        lines.append(f"- handles: {', '.join(result.fetch_plan.handles)}")
    else:
        lines.append("- handles: none")
    if result.budget.truncated:
        lines.extend(
            [
                "",
                "## Omitted",
                f"- files={result.budget.omitted_files}, "
                f"symbols={result.budget.omitted_symbols}, "
                f"modules={result.budget.omitted_modules}, "
                f"graph_edges={result.budget.omitted_graph_edges}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _validate_token_budget(token_budget: int) -> None:
    if token_budget < MIN_SCOUT_TOKEN_BUDGET:
        raise ValueError(
            f"Scout token budget must be at least {MIN_SCOUT_TOKEN_BUDGET}, got {token_budget}"
        )
