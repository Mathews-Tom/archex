"""CLI analyze subcommand: acquire and analyze a repository, writing an ArchProfile."""

from __future__ import annotations

import click


@click.command("analyze")
@click.argument("source")
def analyze_cmd(source: str) -> None:
    """Analyze a repository and produce an architecture profile."""
    print("Not yet implemented. Coming in Phase 2.")
