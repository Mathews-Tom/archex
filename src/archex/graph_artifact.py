"""Deterministic graph artifact models for graph-powered CLI workflows."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

GRAPH_SCHEMA_VERSION = "1.0.0"
SUPPORTED_GRAPH_SCHEMA_MAJOR = 1


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
    node_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_node_ids(self) -> GraphLayer:
        self.node_ids = sorted(dict.fromkeys(self.node_ids))
        return self


class ArchGraph(BaseModel):
    schema_version: GraphSchemaVersion = Field(default_factory=GraphSchemaVersion)
    project: GraphProject
    metadata: GraphExportMetadata
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    layers: list[GraphLayer] = Field(default_factory=list)

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


def assert_supported_schema_version(version: GraphSchemaVersion) -> None:
    if version.major != SUPPORTED_GRAPH_SCHEMA_MAJOR:
        raise GraphArtifactError(
            f"Unsupported graph schema major version {version.major}; "
            f"supported major version is {SUPPORTED_GRAPH_SCHEMA_MAJOR}"
        )


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized
