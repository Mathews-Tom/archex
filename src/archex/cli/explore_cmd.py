"""Local, loopback-only explorer over a previously exported `AnalysisArtifactV1`."""

from __future__ import annotations

from pathlib import Path

import click

from archex.explorer.export import ExplorerExportError, export_explorer_site
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
@click.option(
    "--export",
    "export_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write the views as self-contained offline HTML to this directory instead of serving.",
)
def explore_cmd(
    artifact: Path, graph_path: Path | None, port: int, export_dir: Path | None
) -> None:
    """Render ARTIFACT (an `archex report diff --format json` output) locally.

    Without `--export`, starts a loopback-only, session-token-gated HTTP
    server; nothing is reachable outside this machine and no repository
    indexing runs. With `--export`, writes the same views as static HTML that
    opens from a `file://` URL with no server and no token, and exits.
    """
    try:
        data = load_explorer_data(artifact, graph_path)
    except ExplorerDataError as exc:
        raise click.ClickException(str(exc)) from exc

    if export_dir is not None:
        try:
            result = export_explorer_site(data, export_dir)
        except ExplorerExportError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"wrote {len(result.files)} files to {result.destination}")
        click.echo(f"open {result.destination / 'index.html'}")
        if result.node_pages_omitted:
            click.echo(
                f"per-node pages: {result.node_pages} written, "
                f"{result.node_pages_omitted} omitted at the export cap"
            )
        return

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
