"""Deterministic graph artifact models for graph-powered CLI workflows."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from archex import __version__
from archex.models import CodeChunk, Edge, EdgeKind

if TYPE_CHECKING:
    from archex.index.store import IndexStore

GRAPH_SCHEMA_VERSION = "1.0.0"
SUPPORTED_GRAPH_SCHEMA_MAJOR = 1
_KNOWN_CONFIG_NAMES = {
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "setup.py",
}
_KNOWN_CONFIG_PATHS = tuple(sorted(_KNOWN_CONFIG_NAMES))


class GraphArtifactError(ValueError):
    """Raised when a graph artifact is malformed or unsupported."""


class GraphNodeType(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"
    MODULE = "module"
    INTERFACE = "interface"
    ENTRY_POINT = "entry_point"
    CONFIG = "config"
    TEST = "test"


class GraphEdgeType(StrEnum):
    CONTAINS = "contains"
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"
    BELONGS_TO_MODULE = "belongs_to_module"
    EXPOSES = "exposes"
    TESTS = "tests"
    CONFIGURES = "configures"


class GraphSchemaVersion(BaseModel):
    value: str = GRAPH_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_semver(self) -> GraphSchemaVersion:
        parts = self.value.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ValueError("schema version must use MAJOR.MINOR.PATCH")
        return self

    @property
    def major(self) -> int:
        return int(self.value.split(".", maxsplit=1)[0])


class GraphComplexity(BaseModel):
    line_count: int = 0
    token_count: int = 0
    symbol_count: int = 0
    public_interface_count: int = 0
    import_fan_in: int = 0
    import_fan_out: int = 0
    centrality: float = 0.0


class GraphExportMetadata(BaseModel):
    archex_version: str
    repo_root: str | None = None
    commit_hash: str | None = None
    source_identity: str | None = None
    generated_by: Literal["archex graph export"] = "archex graph export"


class GraphProject(BaseModel):
    name: str
    root_path: str | None = None
    languages: dict[str, int] = Field(default_factory=dict)
    total_files: int = 0
    total_lines: int = 0


class GraphNode(BaseModel):
    id: str
    type: GraphNodeType
    label: str
    path: str | None = None
    language: str | None = None
    symbol_kind: str | None = None
    module: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    description: str = ""
    complexity: GraphComplexity = Field(default_factory=GraphComplexity)

    @model_validator(mode="after")
    def _validate_id_prefix(self) -> GraphNode:
        expected_prefix = f"{self.type.value}:"
        if not self.id.startswith(expected_prefix):
            raise ValueError(f"{self.type.value} node id must start with {expected_prefix!r}")
        return self


class GraphEdge(BaseModel):
    source: str
    target: str
    type: GraphEdgeType
    label: str = ""
    location: str | None = None


class GraphLayer(BaseModel):
    id: str
    name: str
    node_ids: list[str] = []

    @model_validator(mode="after")
    def _sort_node_ids(self) -> GraphLayer:
        self.node_ids = sorted(dict.fromkeys(self.node_ids))
        return self


class ArchGraph(BaseModel):
    schema_version: GraphSchemaVersion = Field(default_factory=GraphSchemaVersion)
    project: GraphProject
    metadata: GraphExportMetadata
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    layers: list[GraphLayer] = []

    @model_validator(mode="after")
    def _sort_collections(self) -> ArchGraph:
        self.nodes = sorted(self.nodes, key=lambda node: (node.type.value, node.id))
        self.edges = sorted(
            self.edges,
            key=lambda edge: (edge.source, edge.target, edge.type.value, edge.location or ""),
        )
        self.layers = sorted(self.layers, key=lambda layer: (layer.name, layer.id))
        return self

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Architecture Graph: {self.project.name}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Files | {self.project.total_files} |",
            f"| Lines | {self.project.total_lines} |",
            f"| Nodes | {len(self.nodes)} |",
            f"| Edges | {len(self.edges)} |",
            "",
        ]
        if self.project.languages:
            lines.extend(["## Languages", "", "| Language | Files |", "| --- | ---: |"])
            for language, count in sorted(self.project.languages.items()):
                lines.append(f"| {language} | {count} |")
            lines.append("")
        if self.nodes:
            lines.extend(["## Nodes", "", "| Type | ID | Label |", "| --- | --- | --- |"])
            for node in self.nodes[:50]:
                lines.append(f"| {node.type.value} | `{node.id}` | {node.label} |")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class _FileFacts:
    path: str
    language: str
    line_count: int
    symbol_count: int
    token_count: int
    public_interface_count: int
    fan_in: int
    fan_out: int


def build_arch_graph_from_store(
    store: IndexStore,
    *,
    repo_root: Path | None = None,
) -> ArchGraph:
    chunks = sorted(
        store.get_chunks(),
        key=lambda chunk: (chunk.file_path, chunk.start_line, chunk.id),
    )
    edges = sorted(
        store.get_edges(),
        key=lambda edge: (edge.source, edge.target, str(edge.kind)),
    )
    file_facts = _collect_file_facts(store, chunks, edges)
    nodes = (
        _build_file_nodes(file_facts)
        + _build_symbol_nodes(chunks)
        + _build_entry_nodes(chunks)
        + _build_config_nodes(repo_root, file_facts)
    )
    graph_edges = _build_graph_edges(chunks, edges)
    languages = _language_counts(file_facts)
    total_lines = sum(fact.line_count for fact in file_facts.values())
    source_identity = store.get_metadata("source_identity")
    commit_hash = store.get_metadata("commit_hash")
    project_name = (
        Path(source_identity).name
        if source_identity
        else (repo_root.name if repo_root else "unknown")
    )
    return ArchGraph(
        project=GraphProject(
            name=project_name or "unknown",
            root_path=str(repo_root) if repo_root is not None else None,
            languages=languages,
            total_files=len(file_facts),
            total_lines=total_lines,
        ),
        metadata=GraphExportMetadata(
            archex_version=__version__,
            repo_root=str(repo_root) if repo_root is not None else None,
            commit_hash=commit_hash,
            source_identity=source_identity,
        ),
        nodes=nodes,
        edges=graph_edges,
    )


def file_node_id(path: str) -> str:
    return f"file:{_normalize_path(path)}"


def symbol_node_id(path: str, qualified_name: str, kind: str) -> str:
    return f"symbol:{_normalize_path(path)}::{qualified_name}#{kind}"


def module_node_id(name: str) -> str:
    return f"module:{name}"


def interface_node_id(path: str, name: str) -> str:
    return f"interface:{_normalize_path(path)}::{name}"


def entry_point_node_id(path: str, name: str) -> str:
    return f"entry_point:{_normalize_path(path)}::{name}"


def config_node_id(path: str) -> str:
    return f"config:{_normalize_path(path)}"


def test_node_id(path: str) -> str:
    return f"test:{_normalize_path(path)}"


def node_id_for_path(path: str) -> str:
    normalized = _normalize_path(path)
    if _is_config_path(normalized):
        return config_node_id(normalized)
    if _is_test_path(normalized):
        return test_node_id(normalized)
    return file_node_id(normalized)


def assert_supported_schema_version(version: GraphSchemaVersion) -> None:
    if version.major != SUPPORTED_GRAPH_SCHEMA_MAJOR:
        raise GraphArtifactError(
            f"Unsupported graph schema major version {version.major}; "
            f"supported major version is {SUPPORTED_GRAPH_SCHEMA_MAJOR}"
        )


def load_arch_graph(path: Path) -> ArchGraph:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphArtifactError(f"Failed to read graph artifact {path}: {exc}") from exc
    try:
        graph = ArchGraph.model_validate_json(raw)
    except ValidationError as exc:
        raise GraphArtifactError(f"Malformed graph artifact {path}: {exc}") from exc
    assert_supported_schema_version(graph.schema_version)
    return graph


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _collect_file_facts(
    store: IndexStore,
    chunks: list[CodeChunk],
    edges: list[Edge],
) -> dict[str, _FileFacts]:
    metadata_by_path = {
        str(item["file_path"]): item
        for item in store.get_file_metadata()
        if str(item["file_path"]).strip()
    }
    tokens_by_path: defaultdict[str, int] = defaultdict(int)
    public_by_path: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        tokens_by_path[chunk.file_path] += chunk.token_count
        if _is_public_interface_chunk(chunk):
            public_by_path[chunk.file_path] += 1
    fan_in: defaultdict[str, int] = defaultdict(int)
    fan_out: defaultdict[str, int] = defaultdict(int)
    for edge in edges:
        if edge.kind == EdgeKind.IMPORTS:
            fan_out[edge.source] += 1
            fan_in[edge.target] += 1
    facts: dict[str, _FileFacts] = {}
    for path, item in metadata_by_path.items():
        facts[path] = _FileFacts(
            path=path,
            language=str(item["language"]),
            line_count=int(item["lines"]),
            symbol_count=int(item["symbol_count"]),
            token_count=tokens_by_path[path],
            public_interface_count=public_by_path[path],
            fan_in=fan_in[path],
            fan_out=fan_out[path],
        )
    return facts


def _build_file_nodes(file_facts: dict[str, _FileFacts]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for fact in file_facts.values():
        normalized = _normalize_path(fact.path)
        node_type = _node_type_for_path(normalized)
        nodes.append(
            GraphNode(
                id=node_id_for_path(normalized),
                type=node_type,
                label=normalized,
                path=normalized,
                language=fact.language,
                description=_file_description(node_type, fact),
                complexity=GraphComplexity(
                    line_count=fact.line_count,
                    token_count=fact.token_count,
                    symbol_count=fact.symbol_count,
                    public_interface_count=fact.public_interface_count,
                    import_fan_in=fact.fan_in,
                    import_fan_out=fact.fan_out,
                ),
            )
        )
    return nodes


def _build_symbol_nodes(chunks: list[CodeChunk]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for chunk in chunks:
        if chunk.symbol_name is None or chunk.symbol_kind is None:
            continue
        qualified_name = chunk.qualified_name or chunk.symbol_name
        kind = str(chunk.symbol_kind)
        nodes.append(
            GraphNode(
                id=symbol_node_id(chunk.file_path, qualified_name, kind),
                type=GraphNodeType.SYMBOL,
                label=qualified_name,
                path=_normalize_path(chunk.file_path),
                language=chunk.language,
                symbol_kind=kind,
                line_start=chunk.start_line,
                line_end=chunk.end_line,
                description=f"{kind} symbol",
                complexity=GraphComplexity(
                    line_count=max(chunk.end_line - chunk.start_line + 1, 0),
                    token_count=chunk.token_count,
                ),
            )
        )
        if _is_public_interface_chunk(chunk):
            nodes.append(
                GraphNode(
                    id=interface_node_id(chunk.file_path, qualified_name),
                    type=GraphNodeType.INTERFACE,
                    label=qualified_name,
                    path=_normalize_path(chunk.file_path),
                    language=chunk.language,
                    symbol_kind=kind,
                    line_start=chunk.start_line,
                    line_end=chunk.end_line,
                    description="public interface symbol",
                )
            )
    return nodes


def _build_entry_nodes(chunks: list[CodeChunk]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for chunk in chunks:
        if not _is_entry_point_chunk(chunk):
            continue
        name = chunk.qualified_name or chunk.symbol_name or Path(chunk.file_path).stem
        nodes.append(
            GraphNode(
                id=entry_point_node_id(chunk.file_path, name),
                type=GraphNodeType.ENTRY_POINT,
                label=name,
                path=_normalize_path(chunk.file_path),
                language=chunk.language,
                symbol_kind=str(chunk.symbol_kind) if chunk.symbol_kind is not None else None,
                line_start=chunk.start_line,
                line_end=chunk.end_line,
                description="entry point detected by adapter metadata",
            )
        )
    return nodes


def _build_config_nodes(
    repo_root: Path | None,
    file_facts: dict[str, _FileFacts],
) -> list[GraphNode]:
    if repo_root is None:
        return []
    nodes: list[GraphNode] = []
    for relative_path in _KNOWN_CONFIG_PATHS:
        if relative_path in file_facts:
            continue
        if not (repo_root / relative_path).is_file():
            continue
        nodes.append(
            GraphNode(
                id=config_node_id(relative_path),
                type=GraphNodeType.CONFIG,
                label=relative_path,
                path=relative_path,
                description="config file",
            )
        )
    return nodes


def _build_graph_edges(chunks: list[CodeChunk], edges: list[Edge]) -> list[GraphEdge]:
    graph_edges = [
        GraphEdge(
            source=node_id_for_path(edge.source),
            target=node_id_for_path(edge.target),
            type=GraphEdgeType.IMPORTS,
            location=edge.location,
        )
        for edge in edges
        if edge.kind == EdgeKind.IMPORTS
    ]
    for chunk in chunks:
        if chunk.symbol_name is None or chunk.symbol_kind is None:
            continue
        qualified_name = chunk.qualified_name or chunk.symbol_name
        kind = str(chunk.symbol_kind)
        symbol_id = symbol_node_id(chunk.file_path, qualified_name, kind)
        graph_edges.append(
            GraphEdge(
                source=node_id_for_path(chunk.file_path),
                target=symbol_id,
                type=GraphEdgeType.CONTAINS,
            )
        )
        if _is_public_interface_chunk(chunk):
            graph_edges.append(
                GraphEdge(
                    source=node_id_for_path(chunk.file_path),
                    target=interface_node_id(chunk.file_path, qualified_name),
                    type=GraphEdgeType.EXPOSES,
                )
            )
    return graph_edges


def _language_counts(file_facts: dict[str, _FileFacts]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fact in file_facts.values():
        counts[fact.language] = counts.get(fact.language, 0) + 1
    return dict(sorted(counts.items()))


def _node_type_for_path(path: str) -> GraphNodeType:
    if _is_config_path(path):
        return GraphNodeType.CONFIG
    if _is_test_path(path):
        return GraphNodeType.TEST
    return GraphNodeType.FILE


def _is_config_path(path: str) -> bool:
    name = path.rsplit("/", maxsplit=1)[-1]
    return name in _KNOWN_CONFIG_NAMES


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", maxsplit=1)[-1]
    return path.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py")


def _is_public_interface_chunk(chunk: CodeChunk) -> bool:
    if chunk.symbol_name is None or chunk.symbol_kind is None:
        return False
    if chunk.visibility not in (None, "public"):
        return False
    return str(chunk.symbol_kind) in {"function", "class", "interface"}


def _is_entry_point_chunk(chunk: CodeChunk) -> bool:
    normalized = _normalize_path(chunk.file_path)
    name = chunk.symbol_name or ""
    return normalized.endswith("/main.py") and name == "run"


def _file_description(node_type: GraphNodeType, fact: _FileFacts) -> str:
    if node_type == GraphNodeType.CONFIG:
        return "config file"
    if node_type == GraphNodeType.TEST:
        return f"test file with {fact.symbol_count} symbols"
    return f"file with {fact.symbol_count} symbols and {fact.fan_out} imports"
