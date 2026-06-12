"""Graph artifact export commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click
from pydantic import BaseModel

from archex.api import index_repository
from archex.config import load_config, load_index_config
from archex.exceptions import ArchexError
from archex.graph_artifact import GraphArtifactError, build_arch_graph_from_store, load_arch_graph
from archex.graph_query import (
    GraphDirection,
    GraphEdgeSummary,
    GraphHubsResult,
    GraphNeighborsResult,
    GraphNodeSummary,
    GraphPathResult,
    GraphQuery,
    GraphQueryError,
    GraphStatsResult,
)
from archex.models import RepoSource

if TYPE_CHECKING:
    from collections.abc import Callable

_ResultT = TypeVar("_ResultT", bound=BaseModel)


@click.group("graph")
def graph_cmd() -> None:
    """Export and inspect deterministic architecture graph artifacts."""


@graph_cmd.command("export")
@click.argument("source", required=False, default=".")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output artifact path. Defaults to .archex/archgraph.json for JSON.",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def export_cmd(source: str, output: Path | None, output_format: str) -> None:
    """Export a deterministic graph artifact for SOURCE."""
    repo_root = Path(source).expanduser().resolve()
    repo_source = RepoSource(local_path=source)
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)
    try:
        store = index_repository(repo_source, config=config, index_config=index_config)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        graph = build_arch_graph_from_store(store, repo_root=repo_root)
    finally:
        store.close()

    rendered = graph.to_json() if output_format == "json" else graph.to_markdown()
    target = output
    if target is None and output_format == "json":
        target = repo_root / ".archex" / "archgraph.json"
    if target is None:
        click.echo(rendered, nl=False)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    click.echo(str(target))


@graph_cmd.command("inspect")
@click.argument("graph", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
def inspect_cmd(graph: Path, output_format: str) -> None:
    """Inspect an exported graph artifact without indexing source files."""
    try:
        artifact = load_arch_graph(graph)
    except GraphArtifactError as exc:
        raise click.ClickException(str(exc)) from exc

    summary = {
        "schema_version": artifact.schema_version.value,
        "project": artifact.project.model_dump(mode="json"),
        "metadata": artifact.metadata.model_dump(mode="json"),
        "nodes": len(artifact.nodes),
        "edges": len(artifact.edges),
        "layers": len(artifact.layers),
    }
    if output_format == "json":
        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    click.echo(f"Graph: {artifact.project.name}")
    click.echo(f"Schema: {artifact.schema_version.value}")
    click.echo(f"Nodes: {len(artifact.nodes)}")
    click.echo(f"Edges: {len(artifact.edges)}")


@graph_cmd.command("neighbors")
@click.argument("node")
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Exported architecture graph JSON artifact. Supplying this avoids reindexing.",
)
@click.option(
    "--direction",
    default="both",
    type=click.Choice(["out", "in", "both"]),
    help="Edge direction to traverse.",
)
@click.option("--depth", default=1, type=int, show_default=True, help="Traversal depth.")
@click.option("--limit", default=25, type=int, show_default=True, help="Maximum edges to return.")
@click.option(
    "--hub-degree",
    default=50,
    type=int,
    show_default=True,
    help="Degree at or above which non-seed nodes are treated as terminal hubs.",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def neighbors_cmd(
    node: str,
    graph_path: Path,
    direction: GraphDirection,
    depth: int,
    limit: int,
    hub_degree: int,
    output_format: str,
) -> None:
    """List graph neighbors for NODE without indexing source files."""
    graph_query = _load_graph_query(graph_path, hub_degree=hub_degree)
    try:
        result = graph_query.neighbors(node, direction=direction, depth=depth, limit=limit)
    except GraphQueryError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_result(result, output_format, _render_neighbors_markdown)


@graph_cmd.command("path")
@click.argument("source")
@click.argument("target")
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Exported architecture graph JSON artifact. Supplying this avoids reindexing.",
)
@click.option(
    "--direction",
    default="both",
    type=click.Choice(["out", "in", "both"]),
    help="Edge direction to traverse.",
)
@click.option(
    "--max-edges",
    default=100,
    type=int,
    show_default=True,
    help="Maximum edge expansions before truncating the search.",
)
@click.option(
    "--hub-degree",
    default=50,
    type=int,
    show_default=True,
    help="Degree at or above which intermediate nodes are treated as terminal hubs.",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def path_cmd(
    source: str,
    target: str,
    graph_path: Path,
    direction: GraphDirection,
    max_edges: int,
    hub_degree: int,
    output_format: str,
) -> None:
    """Find a shortest structural path between SOURCE and TARGET."""
    graph_query = _load_graph_query(graph_path, hub_degree=hub_degree)
    try:
        result = graph_query.shortest_path(
            source,
            target,
            direction=direction,
            max_edges=max_edges,
        )
    except GraphQueryError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_result(result, output_format, _render_path_markdown)


@graph_cmd.command("stats")
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Exported architecture graph JSON artifact. Supplying this avoids reindexing.",
)
@click.option("--hub-limit", default=10, type=int, show_default=True, help="Maximum hubs to show.")
@click.option(
    "--hub-degree",
    default=50,
    type=int,
    show_default=True,
    help="Degree at or above which nodes are reported as hubs.",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def stats_cmd(
    graph_path: Path,
    hub_limit: int,
    hub_degree: int,
    output_format: str,
) -> None:
    """Show graph-level structural statistics."""
    graph_query = _load_graph_query(graph_path, hub_degree=hub_degree)
    try:
        result = graph_query.stats(hub_limit=hub_limit)
    except GraphQueryError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_result(result, output_format, _render_stats_markdown)


@graph_cmd.command("hubs")
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Exported architecture graph JSON artifact. Supplying this avoids reindexing.",
)
@click.option("--limit", default=25, type=int, show_default=True, help="Maximum hubs to return.")
@click.option(
    "--threshold",
    default=None,
    type=int,
    help="Minimum degree for hub reporting. Defaults to --hub-degree.",
)
@click.option(
    "--hub-degree",
    default=50,
    type=int,
    show_default=True,
    help="Default hub threshold.",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def hubs_cmd(
    graph_path: Path,
    limit: int,
    threshold: int | None,
    hub_degree: int,
    output_format: str,
) -> None:
    """List high-degree graph hubs."""
    graph_query = _load_graph_query(graph_path, hub_degree=hub_degree)
    try:
        result = graph_query.hubs(limit=limit, threshold=threshold)
    except GraphQueryError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_result(result, output_format, _render_hubs_markdown)


def _load_graph_query(graph_path: Path, *, hub_degree: int) -> GraphQuery:
    try:
        return GraphQuery.from_artifact(graph_path, hub_degree=hub_degree)
    except (GraphArtifactError, GraphQueryError) as exc:
        raise click.ClickException(str(exc)) from exc


def _emit_result(
    result: _ResultT,
    output_format: str,
    markdown_renderer: Callable[[_ResultT], str],
) -> None:
    if output_format == "json":
        click.echo(result.model_dump_json(indent=2))
        return
    click.echo(markdown_renderer(result), nl=False)


def _render_neighbors_markdown(result: GraphNeighborsResult) -> str:
    lines = [
        f"# Graph Neighbors: {result.seed.id}",
        "",
        f"- Path: `{result.seed.path or result.seed.id}`",
        f"- Direction: `{result.direction}`",
        f"- Depth: {result.depth}",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
        "## Edges",
        "",
    ]
    lines.extend(_render_edge_list(result.edges))
    if result.hubs:
        lines.extend(["", "## Terminal Hubs", ""])
        lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_path_markdown(result: GraphPathResult) -> str:
    source = result.source.path if result.source is not None else result.source_query
    target = result.target.path if result.target is not None else result.target_query
    lines = [
        f"# Graph Path: {source} → {target}",
        "",
        f"- Found: {_yes_no(result.found)}",
        f"- Direction: `{result.direction}`",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    if result.nodes:
        lines.extend(["## Nodes", ""])
        lines.extend(_render_node_list(result.nodes))
        lines.extend(["", "## Edges", ""])
        lines.extend(_render_edge_list(result.edges))
    if result.avoided_hubs:
        lines.extend(["", "## Avoided Hubs", ""])
        lines.extend(_render_node_list(result.avoided_hubs))
    return _finish_markdown(lines)


def _render_stats_markdown(result: GraphStatsResult) -> str:
    lines = [
        f"# Graph Stats: {result.project}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Nodes | {result.nodes} |",
        f"| Edges | {result.edges} |",
        f"| Files | {result.files} |",
        f"| Max degree | {result.max_degree} |",
        "",
    ]
    if result.languages:
        lines.extend(["## Languages", "", "| Language | Files |", "| --- | ---: |"])
        for language, count in sorted(result.languages.items()):
            lines.append(f"| {language} | {count} |")
        lines.append("")
    if result.edge_types:
        lines.extend(["## Edge Types", "", "| Type | Count |", "| --- | ---: |"])
        for edge_type, count in sorted(result.edge_types.items()):
            lines.append(f"| {edge_type} | {count} |")
        lines.append("")
    if result.hubs:
        lines.extend(["## Hubs", ""])
        lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_hubs_markdown(result: GraphHubsResult) -> str:
    lines = [
        "# Graph Hubs",
        "",
        f"- Threshold: {result.threshold}",
        f"- Limit: {result.limit}",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_edge_list(edges: list[GraphEdgeSummary]) -> list[str]:
    if not edges:
        return ["No edges."]
    lines = [
        "| Source path | Kind | Target path | Confidence | Evidence |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for edge in edges:
        evidence = "; ".join(edge.evidence) if edge.evidence else ""
        lines.append(
            "| "
            f"`{edge.source.path or edge.source.id}` | "
            f"{edge.type} | "
            f"`{edge.target.path or edge.target.id}` | "
            f"{edge.confidence} ({edge.confidence_score:.2f}) | "
            f"{evidence} |"
        )
    return lines


def _render_node_list(nodes: list[GraphNodeSummary]) -> list[str]:
    if not nodes:
        return ["No nodes."]
    lines = ["| Path | ID | Type | Degree |", "| --- | --- | --- | ---: |"]
    for node in nodes:
        lines.append(f"| `{node.path or ''}` | `{node.id}` | {node.type} | {node.degree} |")
    return lines


def _finish_markdown(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
