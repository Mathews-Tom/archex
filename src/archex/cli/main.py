"""CLI entry point: click group with --version and analyze/query/compare/cache subcommands."""

from __future__ import annotations

import click

from archex import __version__
from archex.cli.analyze_cmd import analyze_cmd
from archex.cli.cache_cmd import cache_cmd
from archex.cli.compare_cmd import compare_cmd
from archex.cli.query_cmd import query_cmd


@click.group()
@click.version_option(version=__version__, prog_name="archex")
def cli() -> None:
    """archex — architecture extraction and analysis toolkit."""


cli.add_command(analyze_cmd)
cli.add_command(query_cmd)
cli.add_command(compare_cmd)
cli.add_command(cache_cmd)


if __name__ == "__main__":
    cli()
