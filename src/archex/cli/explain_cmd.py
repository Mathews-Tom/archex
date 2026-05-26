"""Explain command for deterministic structural context."""

from __future__ import annotations

from pathlib import Path

import click

from archex.api import index_repository
from archex.config import load_config, load_index_config
from archex.exceptions import ArchexError
from archex.explain import (
    ExplainError,
    explain_file,
    explain_graph_file,
    explain_graph_module,
    explain_graph_symbol,
    explain_module,
    explain_symbol,
    render_explain_context,
)
from archex.graph_artifact import GraphArtifactError, load_arch_graph
from archex.models import RepoSource


@click.command("explain")
@click.argument("source", required=False, default=".")
@click.argument("target", required=False)
@click.option("--module", "module_name", default=None, help="Explain an indexed module path.")
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read an exported graph artifact instead of indexing SOURCE.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
def explain_cmd(
    source: str,
    target: str | None,
    module_name: str | None,
    graph_path: str | None,
    output_format: str,
) -> None:
    """Explain a file, symbol, or module from indexed structural data."""
    if module_name is None and target is None:
        if graph_path is None or source == ".":
            raise click.ClickException("explain requires TARGET or --module")
        target = source
        source = "."
    if module_name is not None and target is not None:
        raise click.ClickException("use either TARGET or --module, not both")

    if graph_path is not None:
        try:
            graph = load_arch_graph(Path(graph_path))
            if module_name is not None:
                context = explain_graph_module(graph, module_name)
            elif target is not None and ("::" in target or "#" in target):
                context = explain_graph_symbol(graph, target)
            elif target is not None:
                context = explain_graph_file(graph, target)
            else:
                raise click.ClickException("explain requires TARGET or --module")
        except (GraphArtifactError, ExplainError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(render_explain_context(context, output_format), nl=False)
        return

    repo_source = RepoSource(local_path=source)
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)
    try:
        store = index_repository(repo_source, config=config, index_config=index_config)
        try:
            if module_name is not None:
                context = explain_module(store, module_name)
            elif target is not None and ("::" in target or "#" in target):
                context = explain_symbol(store, target)
            elif target is not None:
                context = explain_file(store, target)
            else:
                raise click.ClickException("explain requires TARGET or --module")
        finally:
            store.close()
    except (ArchexError, ExplainError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(render_explain_context(context, output_format), nl=False)
