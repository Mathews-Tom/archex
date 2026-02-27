"""CLI query subcommand: retrieve a ContextBundle from a cached ArchProfile."""

from __future__ import annotations

import click


@click.command("query")
@click.argument("query_text")
def query_cmd(query_text: str) -> None:
    """Query a cached architecture profile and return a context bundle."""
    print("Not yet implemented. Coming in Phase 3.")
