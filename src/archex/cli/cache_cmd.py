"""CLI cache subcommand: inspect, clear, and manage the local analysis cache."""

from __future__ import annotations

import click


@click.command("cache")
def cache_cmd() -> None:
    """Manage the local archex analysis cache."""
    print("Not yet implemented. Coming in Phase 2.")
