"""Local, loopback-only explorer over a previously exported `AnalysisArtifactV1`."""

from __future__ import annotations

from pathlib import Path

import click

from archex.explorer.loader import ExplorerDataError, load_explorer_data
from archex.explorer.server import ExplorerSecurityError, create_server


@click.command("explore")
@click.argument("artifact", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional `archex graph export` artifact for module map and neighborhood views.",
)
@click.option(
    "--port",
    default=0,
    type=int,
    show_default=True,
    help="Loopback port to bind (0 selects an OS-assigned ephemeral port).",
)
def explore_cmd(artifact: Path, graph_path: Path | None, port: int) -> None:
    """Render ARTIFACT (an `archex report diff --format json` output) locally.

    Starts a loopback-only, session-token-gated HTTP server; nothing is
    reachable outside this machine and no repository indexing runs.
    """
    try:
        data = load_explorer_data(artifact, graph_path)
    except ExplorerDataError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        server = create_server(data, port=port)
    except ExplorerSecurityError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"archex explorer listening at {server.url}")
    click.echo("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
