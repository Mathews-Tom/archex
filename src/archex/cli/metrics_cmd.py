"""Metrics command group for local token-savings reporting."""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.metrics.health import clear_metrics_health, read_metrics_health
from archex.metrics.policy import set_metrics_enabled, set_trace_enabled
from archex.metrics.reporter import (
    MetricsReporter,
    render_inspect_text,
    render_repos_text,
    render_summary_text,
)
from archex.metrics.storage import metrics_db_path


@click.group("metrics", invoke_without_command=True)
@click.pass_context
def metrics_cmd(ctx: click.Context) -> None:
    """Report and control local archex usage metrics."""
    if ctx.invoked_subcommand is not None:
        return
    payload = MetricsReporter().summary(source=".")
    click.echo(render_summary_text(payload), nl=False)


@metrics_cmd.command("summary")
@click.argument("source", required=False, default=".")
@click.option("--global", "global_scope", is_flag=True, default=False, help="Summarize all repos.")
@click.option("--workspace", type=click.Path(path_type=Path), help="Summarize repos under PATH.")
@click.option("--since", default="7d", show_default=True, help="Time window, e.g. 24h or 7d.")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def summary_cmd(
    source: str,
    global_scope: bool,
    workspace: Path | None,
    since: str,
    output_format: str,
) -> None:
    """Print current repo, global, or workspace savings summary."""
    payload = MetricsReporter().summary(
        source=source,
        global_scope=global_scope,
        workspace=workspace,
        since=since,
    )
    _emit(payload, output_format, render_summary_text)


@metrics_cmd.command("repos")
@click.option("--since", default="30d", show_default=True, help="Time window, e.g. 24h or 30d.")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def repos_cmd(since: str, output_format: str) -> None:
    """List known local repos with savings totals."""
    payload = MetricsReporter().repos(since=since)
    _emit(payload, output_format, render_repos_text)


@metrics_cmd.command("inspect")
@click.argument("source", required=False, default=".")
@click.option("--since", default="24h", show_default=True, help="Time window, e.g. 24h or 7d.")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def inspect_cmd(source: str, since: str, output_format: str) -> None:
    """Inspect recent local metrics events."""
    payload = MetricsReporter().inspect(source=source, since=since)
    _emit(payload, output_format, render_inspect_text)


@metrics_cmd.command("export")
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("usage.json"),
    show_default=True,
    help="Destination JSON file.",
)
@click.option("--since", default="90d", show_default=True, help="Time window, e.g. 90d.")
@click.option("--include-local-paths", is_flag=True, default=False, help="Include repo root paths.")
def export_cmd(output: Path, since: str, include_local_paths: bool) -> None:
    """Export local metrics JSON."""
    payload = MetricsReporter().export(since=since, include_local_paths=include_local_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    click.echo(f"Exported metrics to {output}")


def _is_tty() -> bool:
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


@metrics_cmd.command("setup")
@click.option("--yes", is_flag=True, help="Apply default settings without prompting.")
def setup_cmd(yes: bool) -> None:
    """Configure local metrics and traces interactively."""

    is_tty = _is_tty()
    if not is_tty and not yes:
        raise click.UsageError(
            "metrics setup is interactive by default, but stdin/stdout are not TTY.\n"
            "Pass --yes to apply default settings."
        )

    if not yes:
        click.echo("Configure local token-savings metrics?")
        click.echo()
        click.echo("Local metrics:")
        click.echo("- stored only at ~/.archex/usage.sqlite")
        click.echo("- no hosted upload")
        click.echo("- no LLM calls")
        click.echo(
            "- counters do not store query text, file paths, snippets, prompts, Git remotes, "
            "org names, or repo names"
        )
        click.echo()
        counters_yes = click.confirm("Enable local counters?", default=True)

        click.echo()
        click.echo("Detailed traces:")
        click.echo("- local only")
        click.echo("- retained for 14 days")
        click.echo("- can include query text and returned file paths")
        click.echo("- useful for debugging MCP/CLI usage")
        click.echo("- not needed for normal savings totals")
        click.echo()
        trace_yes = click.confirm("Enable detailed traces?", default=False)
    else:
        counters_yes = True
        trace_yes = False

    set_metrics_enabled(counters_yes)
    set_trace_enabled(trace_yes)

    click.echo()
    click.echo(f"Metrics counters: {'enabled' if counters_yes else 'disabled'}")
    click.echo(f"Metrics trace: {'enabled' if trace_yes else 'disabled'}")


@metrics_cmd.command("enable")
def enable_cmd() -> None:
    """Enable anonymous local metrics counters."""
    set_metrics_enabled(True)
    click.echo("Metrics recording enabled")


@metrics_cmd.command("disable")
def disable_cmd() -> None:
    """Disable anonymous local metrics counters."""
    set_metrics_enabled(False)
    click.echo("Metrics recording disabled")


@metrics_cmd.command("repair")
def repair_cmd() -> None:
    """Clear a stale metrics health warning once recording is working again."""
    health = read_metrics_health()
    if health.status == "ok":
        click.echo("Metrics health: ok (nothing to repair)")
        return
    clear_metrics_health()
    click.echo("Cleared metrics health warning")


@metrics_cmd.command("delete")
@click.option(
    "--all",
    "delete_all",
    is_flag=True,
    default=False,
    help="Delete all local metrics state.",
)
def delete_cmd(delete_all: bool) -> None:
    """Delete local metrics state."""
    if not delete_all:
        raise click.UsageError("metrics delete requires --all")
    db_path = metrics_db_path()
    sidecars = (
        db_path.with_suffix(db_path.suffix + "-wal"),
        db_path.with_suffix(db_path.suffix + "-shm"),
    )
    for path in (db_path, *sidecars):
        if path.exists():
            path.unlink()
    click.echo("Deleted local metrics state")


@metrics_cmd.group("trace")
def trace_cmd() -> None:
    """Control opt-in detailed local traces."""


@trace_cmd.command("enable")
def trace_enable_cmd() -> None:
    """Enable detailed local traces."""
    set_trace_enabled(True)
    click.echo("Metrics trace enabled")


@trace_cmd.command("disable")
def trace_disable_cmd() -> None:
    """Disable detailed local traces."""
    set_trace_enabled(False)
    click.echo("Metrics trace disabled")


def _emit(payload: dict[str, object], output_format: str, renderer: object) -> None:
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(renderer(payload), nl=False)  # type: ignore[operator]
