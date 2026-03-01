"""CLI cache subcommand: inspect, clear, and manage the local analysis cache."""

from __future__ import annotations

import click

from archex.cache import CacheManager
from archex.config import DEFAULT_CACHE_DIR


@click.group("cache")
def cache_cmd() -> None:
    """Manage the local archex analysis cache."""


@cache_cmd.command("list")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory path.")
def cache_list(cache_dir: str) -> None:
    """List all cached entries."""
    mgr = CacheManager(cache_dir=cache_dir)
    entries = mgr.list_entries()
    if not entries:
        click.echo("No cached entries.")
        return
    for entry in entries:
        click.echo(f"  {entry['key'][:12]}..  {entry['size_bytes']} bytes  {entry['path']}")


@cache_cmd.command("clean")
@click.option("--max-age", default=24, type=int, help="Remove entries older than N hours.")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory path.")
def cache_clean(max_age: int, cache_dir: str) -> None:
    """Remove expired cache entries."""
    mgr = CacheManager(cache_dir=cache_dir)
    removed = mgr.clean(max_age_hours=max_age)
    click.echo(f"Removed {removed} expired entries.")


@cache_cmd.command("info")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory path.")
def cache_info(cache_dir: str) -> None:
    """Show cache summary information."""
    mgr = CacheManager(cache_dir=cache_dir)
    info = mgr.info()
    click.echo(f"Cache directory: {info['cache_dir']}")
    click.echo(f"Total entries:   {info['total_entries']}")
    click.echo(f"Total size:      {info['total_size_bytes']} bytes")
