"""Token-capped structural scout maps for two-step repository context."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from archex.graph_query import GraphQuery, GraphQueryError
from archex.models import CodeChunk, Module, RankedChunk, SymbolKind
from archex.reporting import count_tokens

if TYPE_CHECKING:
    from archex.index.store import IndexStore

ScoutFormat = Literal["json", "markdown"]
ScoutHandleKind = Literal["file", "symbol", "chunk"]

DEFAULT_SCOUT_TOKEN_BUDGET = 1000
MIN_SCOUT_TOKEN_BUDGET = 64
DEFAULT_SCOUT_FILE_LIMIT = 12
DEFAULT_SCOUT_SYMBOLS_PER_FILE = 3
DEFAULT_SCOUT_MODULE_LIMIT = 6
DEFAULT_SCOUT_GRAPH_EDGE_LIMIT = 12
FILE_HANDLE_PREFIX = "file:"
SYMBOL_HANDLE_PREFIX = "symbol:"
CHUNK_HANDLE_PREFIX = "chunk:"


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


class ScoutResult(BaseModel):
    query: str
    ranked_files: list[ScoutFile] = []
    modules: list[ScoutModule] = []
    symbols: list[ScoutSymbol] = []
    graph: list[ScoutGraphEdge] = []
    budget: ScoutBudget


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
) -> ScoutResult:
    """Build a deterministic no-body structural map from an indexed repository."""
    _validate_token_budget(token_budget)
    ranked = ranked_chunks or []
    files, omitted_files = _rank_files(store, ranked, file_limit=file_limit)
    chunks_by_file = _chunks_by_file(store, [item.path for item in files])
    symbols, omitted_symbols = _top_symbols(
        chunks_by_file, ranked, symbols_per_file=symbols_per_file
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
    return enforce_scout_token_budget(result, output_format=output_format)


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
        msg = f"Scout metadata exceeds token budget {budget}; minimum practical budget is higher"
        raise ValueError(msg)


def _stable_rendered_token_count(result: ScoutResult, *, output_format: ScoutFormat) -> int:
    for _ in range(4):
        token_count = count_tokens(render_scout(result, output_format=output_format))
        if token_count == result.budget.token_count:
            return token_count
        result.budget.token_count = token_count
    return count_tokens(render_scout(result, output_format=output_format))


def render_scout(result: ScoutResult, *, output_format: ScoutFormat = "markdown") -> str:
    if output_format == "json":
        return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if output_format == "markdown":
        return _render_markdown(result)
    raise ValueError(f"Unsupported scout format {output_format!r}")


def _rank_files(
    store: IndexStore,
    ranked_chunks: list[RankedChunk],
    *,
    file_limit: int,
) -> tuple[list[ScoutFile], int]:
    metadata = {str(row["file_path"]): row for row in store.get_file_metadata()}
    file_scores: dict[str, float] = {}
    for ranked in ranked_chunks:
        score = ranked.final_score or ranked.relevance_score or ranked.structural_score
        current = file_scores.get(ranked.chunk.file_path, 0.0)
        if score > current:
            file_scores[ranked.chunk.file_path] = score
    if not file_scores:
        file_scores = {path: 0.0 for path in metadata}
    ordered_paths = sorted(file_scores, key=lambda path: (-file_scores[path], path))
    selected_paths = ordered_paths[:file_limit]
    files = [
        ScoutFile(
            path=path,
            language=str(metadata.get(path, {}).get("language", "unknown")),
            lines=int(metadata.get(path, {}).get("lines", 0)),
            symbol_count=int(metadata.get(path, {}).get("symbol_count", 0)),
            handle=file_handle(path),
            score=round(file_scores[path], 6),
            reason="query_rank" if ranked_chunks else "file_tree",
        )
        for path in selected_paths
    ]
    return files, max(0, len(ordered_paths) - len(selected_paths))


def _chunks_by_file(store: IndexStore, file_paths: list[str]) -> dict[str, list[CodeChunk]]:
    chunks = store.get_chunks_for_files(file_paths)
    result: dict[str, list[CodeChunk]] = {path: [] for path in file_paths}
    for chunk in sorted(chunks, key=lambda item: (item.file_path, item.start_line, item.id)):
        result.setdefault(chunk.file_path, []).append(chunk)
    return result


def _top_symbols(
    chunks_by_file: dict[str, list[CodeChunk]],
    ranked_chunks: list[RankedChunk],
    *,
    symbols_per_file: int,
) -> tuple[list[ScoutSymbol], int]:
    score_by_chunk = {
        ranked.chunk.id: ranked.final_score or ranked.relevance_score or ranked.structural_score
        for ranked in ranked_chunks
    }
    symbols: list[ScoutSymbol] = []
    omitted = 0
    for file_path in sorted(chunks_by_file):
        chunks = [
            chunk for chunk in chunks_by_file[file_path] if chunk.symbol_name and chunk.symbol_kind
        ]
        chunks = sorted(
            chunks,
            key=lambda chunk: (
                -(score_by_chunk.get(chunk.id, 0.0)),
                _visibility_rank(chunk.visibility),
                chunk.start_line,
                chunk.symbol_name or "",
                chunk.id,
            ),
        )
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
                f"{item.symbol_count} symbols, score={item.score:.3f})"
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
