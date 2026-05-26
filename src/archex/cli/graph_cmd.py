"""Graph artifact export commands."""

from __future__ import annotations

from pathlib import Path

import click

from archex.api import index_repository
from archex.config import load_config, load_index_config
from archex.exceptions import ArchexError
from archex.graph_artifact import GraphArtifactError, build_arch_graph_from_store, load_arch_graph
from archex.models import RepoSource


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
        import json

        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    click.echo(f"Graph: {artifact.project.name}")
    click.echo(f"Schema: {artifact.schema_version.value}")
    click.echo(f"Nodes: {len(artifact.nodes)}")
    click.echo(f"Edges: {len(artifact.edges)}")
