"""CLI entry point: click group with --version and analyze/query/compare/cache subcommands."""

from __future__ import annotations

import click

from archex import __version__
from archex.cli.analyze_cmd import analyze_cmd
from archex.cli.benchmark_cmd import benchmark_cmd
from archex.cli.cache_cmd import cache_cmd
from archex.cli.compare_cmd import compare_cmd
from archex.cli.context_cmd import context_cmd
from archex.cli.doctor_cmd import doctor_cmd
from archex.cli.dogfood_cmd import dogfood_cmd
from archex.cli.explain_cmd import explain_cmd
from archex.cli.explore_cmd import explore_cmd
from archex.cli.graph_cmd import graph_cmd
from archex.cli.impact_cmd import impact_cmd
from archex.cli.index_cmd import index_cmd
from archex.cli.init_cmd import init_cmd
from archex.cli.install_client_cmd import install_client_cmd
from archex.cli.mcp_cmd import mcp_cmd
from archex.cli.metrics_cmd import metrics_cmd
from archex.cli.onboard_cmd import onboard_cmd
from archex.cli.outline_cmd import outline_cmd
from archex.cli.query_cmd import query_cmd
from archex.cli.report_cmd import report_cmd
from archex.cli.reset_cmd import reset_cmd
from archex.cli.scout_cmd import scout_cmd
from archex.cli.setup_cmd import setup_cmd
from archex.cli.status_cmd import status_cmd
from archex.cli.symbol_cmd import symbol_cmd
from archex.cli.symbols_cmd import symbols_cmd
from archex.cli.tree_cmd import tree_cmd


@click.group()
@click.version_option(version=__version__, prog_name="archex")
def cli() -> None:
    """archex — architecture extraction and analysis toolkit."""


cli.add_command(analyze_cmd)
cli.add_command(benchmark_cmd)
cli.add_command(query_cmd)
cli.add_command(context_cmd)
cli.add_command(compare_cmd)
cli.add_command(cache_cmd)
cli.add_command(init_cmd)
cli.add_command(index_cmd)
cli.add_command(install_client_cmd)
cli.add_command(setup_cmd)
cli.add_command(status_cmd)
cli.add_command(reset_cmd)
cli.add_command(doctor_cmd)
cli.add_command(dogfood_cmd)
cli.add_command(mcp_cmd)
cli.add_command(metrics_cmd)
cli.add_command(tree_cmd)
cli.add_command(outline_cmd)
cli.add_command(scout_cmd)
cli.add_command(symbols_cmd)
cli.add_command(symbol_cmd)
cli.add_command(graph_cmd)
cli.add_command(explain_cmd)
cli.add_command(impact_cmd)
cli.add_command(onboard_cmd)
cli.add_command(report_cmd)
cli.add_command(explore_cmd)


if __name__ == "__main__":
    cli()
