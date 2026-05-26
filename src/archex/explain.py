"""Deterministic explain-context models and renderers."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from archex.graph_artifact import (
    ArchGraph,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    node_id_for_path,
    symbol_node_id,
)

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import CodeChunk, Edge


class ExplainTargetType(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"
    MODULE = "module"


class ExplainError(ValueError):
    """Raised when explain context cannot be built."""


class ExplainSymbol(BaseModel):
    id: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    signature: str | None = None


class ExplainComplexity(BaseModel):
    file_count: int = 0
    line_count: int = 0
    token_count: int = 0
    symbol_count: int = 0
    import_fan_in: int = 0
    import_fan_out: int = 0


class ExplainContext(BaseModel):
    target_type: ExplainTargetType
    target: str
    graph_node_ids: list[str] = []
    files: list[str] = []
    public_interfaces: list[ExplainSymbol] = []
    internal_symbols: list[ExplainSymbol] = []
    imports: list[str] = []
    imported_by: list[str] = []
    module_files: list[str] = []
    direct_dependency_edges: list[tuple[str, str]] = []
    complexity: ExplainComplexity = Field(default_factory=ExplainComplexity)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            f"# Explain: {self.target}",
            "",
            "## Target",
            "",
            f"- **Type:** `{self.target_type.value}`",
            f"- **Target:** `{self.target}`",
            "",
            "## Structural Role",
            "",
            f"- **Files:** {len(self.files)}",
            f"- **Imports:** {len(self.imports)}",
            f"- **Imported by:** {len(self.imported_by)}",
            "",
            "## Public Surface",
            "",
        ]
        lines.extend(_symbol_lines(self.public_interfaces))
        lines.extend(["", "## Internal Symbols", ""])
        lines.extend(_symbol_lines(self.internal_symbols))
        lines.extend(["", "## Imports", ""])
        lines.extend(_path_lines(self.imports))
        lines.extend(["", "## Imported By", ""])
        lines.extend(_path_lines(self.imported_by))
        lines.extend(["", "## Module Context", ""])
        lines.extend(_path_lines(self.module_files))
        lines.extend(["", "## Complexity Signals", ""])
        lines.extend(
            [
                f"- **Lines:** {self.complexity.line_count}",
                f"- **Tokens:** {self.complexity.token_count}",
                f"- **Symbols:** {self.complexity.symbol_count}",
                f"- **Fan-in:** {self.complexity.import_fan_in}",
                f"- **Fan-out:** {self.complexity.import_fan_out}",
            ]
        )
        lines.extend(["", "## Suggested Reading Context", ""])
        lines.extend(_path_lines(self.files[:10]))
        return "\n".join(lines).rstrip() + "\n"


def explain_file(store: IndexStore, file_path: str) -> ExplainContext:
    chunks = sorted(
        store.get_chunks_for_file(file_path),
        key=lambda chunk: (chunk.start_line, chunk.id),
    )
    if not chunks:
        raise ExplainError(f"Target file does not exist in index: {file_path}")
    edges = store.get_edges()
    imports, imported_by = _edge_context(edges, {file_path})
    symbols = [_symbol_from_chunk(chunk) for chunk in chunks if chunk.symbol_name]
    public, internal = _partition_symbols(symbols)
    module_files = _module_files(store, file_path)
    return ExplainContext(
        target_type=ExplainTargetType.FILE,
        target=file_path,
        graph_node_ids=[node_id_for_path(file_path)],
        files=[file_path],
        public_interfaces=public,
        internal_symbols=internal,
        imports=imports,
        imported_by=imported_by,
        module_files=module_files,
        direct_dependency_edges=_direct_edges(edges, {file_path}),
        complexity=_complexity(chunks, imports, imported_by, file_count=1),
    )


def explain_symbol(store: IndexStore, target: str) -> ExplainContext:
    chunk = _find_symbol_chunk(store, target)
    if chunk is None:
        raise ExplainError(f"Target symbol does not exist in index: {target}")
    file_context = explain_file(store, chunk.file_path)
    qualified_name = chunk.qualified_name or chunk.symbol_name or target
    kind = str(chunk.symbol_kind) if chunk.symbol_kind is not None else "symbol"
    return file_context.model_copy(
        update={
            "target_type": ExplainTargetType.SYMBOL,
            "target": target,
            "graph_node_ids": [symbol_node_id(chunk.file_path, qualified_name, kind)],
            "files": [chunk.file_path],
            "public_interfaces": [_symbol_from_chunk(chunk)]
            if _is_public_interface_chunk(chunk)
            else [],
            "internal_symbols": []
            if _is_public_interface_chunk(chunk)
            else [_symbol_from_chunk(chunk)],
            "complexity": _complexity(
                [chunk],
                file_context.imports,
                file_context.imported_by,
                file_count=1,
            ),
        }
    )


def explain_module(store: IndexStore, module_name: str) -> ExplainContext:
    files = [
        str(item["file_path"])
        for item in store.get_file_metadata()
        if _path_matches_module(str(item["file_path"]), module_name)
    ]
    files = sorted(files)
    if not files:
        raise ExplainError(f"Target module does not exist in index: {module_name}")
    chunks = sorted(
        store.get_chunks_for_files(files),
        key=lambda chunk: (chunk.file_path, chunk.start_line),
    )
    edges = store.get_edges()
    imports, imported_by = _edge_context(edges, set(files))
    symbols = [_symbol_from_chunk(chunk) for chunk in chunks if chunk.symbol_name]
    public, internal = _partition_symbols(symbols)
    return ExplainContext(
        target_type=ExplainTargetType.MODULE,
        target=module_name,
        graph_node_ids=[node_id_for_path(path) for path in files],
        files=files,
        public_interfaces=public,
        internal_symbols=internal,
        imports=imports,
        imported_by=imported_by,
        module_files=files,
        direct_dependency_edges=_direct_edges(edges, set(files)),
        complexity=_complexity(chunks, imports, imported_by, file_count=len(files)),
    )


def render_explain_context(context: ExplainContext, output_format: str) -> str:
    if output_format == "json":
        return context.to_json()
    if output_format == "markdown":
        return context.to_markdown()
    raise ExplainError(f"Unsupported explain output format: {output_format}")


def explain_graph_file(graph: ArchGraph, file_path: str) -> ExplainContext:
    node = _graph_node_by_path(graph, file_path)
    if node is None:
        raise ExplainError(f"Target file does not exist in graph: {file_path}")
    imports, imported_by = _graph_edge_context(graph, node.id)
    symbol_nodes = _graph_symbol_nodes_for_file(graph, file_path)
    symbols = [_symbol_from_graph_node(symbol) for symbol in symbol_nodes]
    public, internal = _partition_symbols(symbols)
    module_files = _graph_module_files(graph, file_path)
    return ExplainContext(
        target_type=ExplainTargetType.FILE,
        target=file_path,
        graph_node_ids=[node.id],
        files=[file_path],
        public_interfaces=public,
        internal_symbols=internal,
        imports=imports,
        imported_by=imported_by,
        module_files=module_files,
        direct_dependency_edges=_graph_direct_edges(graph, {node.id}),
        complexity=ExplainComplexity(
            file_count=1,
            line_count=node.complexity.line_count,
            token_count=node.complexity.token_count,
            symbol_count=node.complexity.symbol_count,
            import_fan_in=len(imported_by),
            import_fan_out=len(imports),
        ),
    )


def explain_graph_symbol(graph: ArchGraph, target: str) -> ExplainContext:
    node = _graph_node_by_id_or_label(graph, target, GraphNodeType.SYMBOL)
    if node is None:
        raise ExplainError(f"Target symbol does not exist in graph: {target}")
    file_path = node.path or ""
    context = explain_graph_file(graph, file_path)
    symbol = _symbol_from_graph_node(node)
    return context.model_copy(
        update={
            "target_type": ExplainTargetType.SYMBOL,
            "target": target,
            "graph_node_ids": [node.id],
            "public_interfaces": [symbol]
            if node.symbol_kind in {"function", "class", "interface"}
            else [],
            "internal_symbols": []
            if node.symbol_kind in {"function", "class", "interface"}
            else [symbol],
        }
    )


def explain_graph_module(graph: ArchGraph, module_name: str) -> ExplainContext:
    files = [
        node.path or node.label
        for node in graph.nodes
        if node.type in {GraphNodeType.FILE, GraphNodeType.TEST}
        and _path_matches_module(node.path or node.label, module_name)
    ]
    files = sorted(files)
    if not files:
        raise ExplainError(f"Target module does not exist in graph: {module_name}")
    file_node_ids = {
        node.id
        for node in graph.nodes
        if node.type in {GraphNodeType.FILE, GraphNodeType.TEST}
        and (node.path or node.label) in files
    }
    imports, imported_by = _graph_edge_context(graph, file_node_ids)
    symbols = [
        _symbol_from_graph_node(node)
        for node in graph.nodes
        if node.type == GraphNodeType.SYMBOL and node.path in files
    ]
    public, internal = _partition_symbols(symbols)
    return ExplainContext(
        target_type=ExplainTargetType.MODULE,
        target=module_name,
        graph_node_ids=sorted(file_node_ids),
        files=files,
        public_interfaces=public,
        internal_symbols=internal,
        imports=imports,
        imported_by=imported_by,
        module_files=files,
        direct_dependency_edges=_graph_direct_edges(graph, file_node_ids),
        complexity=ExplainComplexity(
            file_count=len(files),
            line_count=_graph_line_count(graph, files),
            symbol_count=len(symbols),
            import_fan_in=len(imported_by),
            import_fan_out=len(imports),
        ),
    )


def _symbol_from_chunk(chunk: CodeChunk) -> ExplainSymbol:
    name = chunk.symbol_name or chunk.qualified_name or chunk.id
    qualified_name = chunk.qualified_name or name
    kind = str(chunk.symbol_kind) if chunk.symbol_kind is not None else "symbol"
    return ExplainSymbol(
        id=symbol_node_id(chunk.file_path, qualified_name, kind),
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        file_path=chunk.file_path,
        line_start=chunk.start_line,
        line_end=chunk.end_line,
        signature=chunk.signature,
    )


def _symbol_from_graph_node(node: GraphNode) -> ExplainSymbol:
    return ExplainSymbol(
        id=node.id,
        name=node.label,
        qualified_name=node.label,
        kind=node.symbol_kind or "symbol",
        file_path=node.path or "",
        line_start=node.line_start or 0,
        line_end=node.line_end or 0,
    )


def _graph_node_by_path(graph: ArchGraph, file_path: str) -> GraphNode | None:
    for node in graph.nodes:
        if node.type in {GraphNodeType.FILE, GraphNodeType.TEST} and node.path == file_path:
            return node
    return None


def _graph_node_by_id_or_label(
    graph: ArchGraph,
    target: str,
    node_type: GraphNodeType,
) -> GraphNode | None:
    matches = [
        node
        for node in graph.nodes
        if node.type == node_type and target in {node.id, node.label}
    ]
    return sorted(matches, key=lambda node: node.id)[0] if matches else None


def _graph_symbol_nodes_for_file(graph: ArchGraph, file_path: str) -> list[GraphNode]:
    return [
        node
        for node in graph.nodes
        if node.type == GraphNodeType.SYMBOL and node.path == file_path
    ]


def _graph_module_files(graph: ArchGraph, file_path: str) -> list[str]:
    if "/" not in file_path:
        return [file_path]
    module_prefix = file_path.rsplit("/", maxsplit=1)[0]
    return sorted(
        node.path or node.label
        for node in graph.nodes
        if node.type in {GraphNodeType.FILE, GraphNodeType.TEST}
        and (node.path or node.label).startswith(f"{module_prefix}/")
    )


def _graph_edge_context(
    graph: ArchGraph,
    node_ids: str | set[str],
) -> tuple[list[str], list[str]]:
    ids = {node_ids} if isinstance(node_ids, str) else node_ids
    path_by_id = {node.id: node.path or node.label for node in graph.nodes}
    imports = sorted(
        path_by_id.get(edge.target, edge.target)
        for edge in graph.edges
        if edge.type == GraphEdgeType.IMPORTS and edge.source in ids and edge.target not in ids
    )
    imported_by = sorted(
        path_by_id.get(edge.source, edge.source)
        for edge in graph.edges
        if edge.type == GraphEdgeType.IMPORTS and edge.target in ids and edge.source not in ids
    )
    return imports, imported_by


def _graph_direct_edges(graph: ArchGraph, node_ids: set[str]) -> list[tuple[str, str]]:
    path_by_id = {node.id: node.path or node.label for node in graph.nodes}
    return sorted(
        {
            (path_by_id.get(edge.source, edge.source), path_by_id.get(edge.target, edge.target))
            for edge in graph.edges
            if edge.source in node_ids or edge.target in node_ids
        }
    )


def _graph_line_count(graph: ArchGraph, files: list[str]) -> int:
    total = 0
    for path in files:
        node = _graph_node_by_path(graph, path)
        if node is not None:
            total += node.complexity.line_count
    return total


def _partition_symbols(
    symbols: list[ExplainSymbol],
) -> tuple[list[ExplainSymbol], list[ExplainSymbol]]:
    public = [symbol for symbol in symbols if symbol.kind in {"function", "class", "interface"}]
    internal = [symbol for symbol in symbols if symbol not in public]
    return (
        sorted(public, key=lambda symbol: (symbol.file_path, symbol.line_start, symbol.id)),
        sorted(internal, key=lambda symbol: (symbol.file_path, symbol.line_start, symbol.id)),
    )


def _find_symbol_chunk(store: IndexStore, target: str) -> CodeChunk | None:
    if "::" in target and "#" in target:
        file_path, remainder = target.split("::", maxsplit=1)
        qualified_name, kind = remainder.rsplit("#", maxsplit=1)
        for chunk in store.get_chunks_for_file(file_path):
            if chunk.qualified_name == qualified_name and str(chunk.symbol_kind) == kind:
                return chunk
    matches = [
        chunk
        for chunk in store.search_symbols(target, limit=20)
        if target in {chunk.symbol_name, chunk.qualified_name, chunk.symbol_id}
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda chunk: (chunk.file_path, chunk.start_line, chunk.id))[0]


def _edge_context(edges: list[Edge], files: set[str]) -> tuple[list[str], list[str]]:
    imports = sorted(
        {edge.target for edge in edges if edge.source in files and edge.target not in files}
    )
    imported_by = sorted(
        {edge.source for edge in edges if edge.target in files and edge.source not in files}
    )
    return imports, imported_by


def _direct_edges(edges: list[Edge], files: set[str]) -> list[tuple[str, str]]:
    return sorted(
        {
            (edge.source, edge.target)
            for edge in edges
            if edge.source in files or edge.target in files
        }
    )


def _module_files(store: IndexStore, file_path: str) -> list[str]:
    if "/" not in file_path:
        return [file_path]
    module_prefix = file_path.rsplit("/", maxsplit=1)[0]
    return sorted(
        str(item["file_path"])
        for item in store.get_file_metadata()
        if str(item["file_path"]).startswith(f"{module_prefix}/")
    )


def _path_matches_module(path: str, module_name: str) -> bool:
    return path == module_name or path.startswith(f"{module_name}/") or f"/{module_name}/" in path


def _complexity(
    chunks: list[CodeChunk],
    imports: list[str],
    imported_by: list[str],
    *,
    file_count: int,
) -> ExplainComplexity:
    return ExplainComplexity(
        file_count=file_count,
        line_count=sum(max(chunk.end_line - chunk.start_line + 1, 0) for chunk in chunks),
        token_count=sum(chunk.token_count for chunk in chunks),
        symbol_count=sum(1 for chunk in chunks if chunk.symbol_name),
        import_fan_in=len(imported_by),
        import_fan_out=len(imports),
    )


def _is_public_interface_chunk(chunk: CodeChunk) -> bool:
    if chunk.symbol_name is None or chunk.symbol_kind is None:
        return False
    if chunk.visibility not in (None, "public"):
        return False
    return str(chunk.symbol_kind) in {"function", "class", "interface"}


def _symbol_lines(symbols: list[ExplainSymbol]) -> list[str]:
    if not symbols:
        return ["- None"]
    return [
        f"- `{symbol.qualified_name}` ({symbol.kind}, {symbol.file_path}:{symbol.line_start})"
        for symbol in symbols
    ]


def _path_lines(paths: list[str]) -> list[str]:
    if not paths:
        return ["- None"]
    return [f"- `{path}`" for path in paths]
