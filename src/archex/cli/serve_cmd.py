"""CLI command: archex serve — start the HTTP API server."""

from __future__ import annotations

import click


@click.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind host.")
@click.option("--port", default=8080, type=int, help="Bind port.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve_cmd(host: str, port: int, reload: bool) -> None:
    """Start the archex HTTP API server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "uvicorn is required. Install with: uv add 'archex[web]'"
        ) from exc

    uvicorn.run(
        "archex.serve.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
