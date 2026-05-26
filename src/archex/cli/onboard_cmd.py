"""Onboarding guide command."""

from __future__ import annotations

from pathlib import Path

import click

from archex.api import index_repository
from archex.config import load_config, load_index_config
from archex.exceptions import ArchexError
from archex.graph_artifact import GraphArtifactError, build_arch_graph_from_store, load_arch_graph
from archex.models import RepoSource
from archex.onboarding import OnboardingError, render_onboarding_markdown


@click.command("onboard")
@click.argument("source", required=False, default=".")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output markdown path. Defaults to stdout.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown"]),
    help="Output format.",
)
@click.option(
    "--max-files",
    default=40,
    show_default=True,
    help="Maximum paths per capped section.",
)
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read an exported graph artifact instead of indexing SOURCE.",
)
def onboard_cmd(
    source: str,
    output: Path | None,
    output_format: str,
    max_files: int,
    graph_path: Path | None,
) -> None:
    """Generate a deterministic onboarding guide from graph/profile data."""
    if output_format != "markdown":
        raise click.ClickException(f"Unsupported onboarding output format: {output_format}")
    try:
        if graph_path is not None:
            graph = load_arch_graph(graph_path)
        else:
            repo_root = Path(source).expanduser().resolve()
            repo_source = RepoSource(local_path=source)
            config = load_config(repo_source)
            index_config = load_index_config(repo_source)
            store = index_repository(repo_source, config=config, index_config=index_config)
            try:
                graph = build_arch_graph_from_store(store, repo_root=repo_root)
            finally:
                store.close()
        rendered = render_onboarding_markdown(graph, max_files=max_files)
    except (ArchexError, GraphArtifactError, OnboardingError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output is None:
        click.echo(rendered, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    click.echo(str(output))
