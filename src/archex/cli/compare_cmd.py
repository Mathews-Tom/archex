"""CLI compare subcommand: compare two repositories and produce a ComparisonResult."""

from __future__ import annotations

import click


@click.command("compare")
@click.argument("source_a")
@click.argument("source_b")
def compare_cmd(source_a: str, source_b: str) -> None:
    """Compare two repositories across architectural dimensions."""
    print("Not yet implemented. Coming in Phase 4.")
