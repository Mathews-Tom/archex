from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import click

from archex.cli.indexing import run_indexing_and_get_summary
from archex.client_setup import discover_agent_files, discover_clients
from archex.doctor import mcp_runtime_available
from archex.metrics.policy import resolve_metrics_policy
from archex.project import init_project
from archex.status import inspect_project_status


@dataclass
class PreflightState:
    has_dot_archex: bool
    has_index: bool
    is_index_fresh: bool
    metrics_enabled: bool
    metrics_trace_enabled: bool
    mcp_runtime_available: bool
    discovered_clients: list[str]
    discovered_agent_files: list[str]


def run_preflight(source: Path) -> PreflightState:
    source_resolved = source.resolve()
    dot_archex = source_resolved / ".archex"
    has_dot_archex = dot_archex.exists() and dot_archex.is_dir()

    status = inspect_project_status(source_resolved)
    has_index = bool(status.indexed_commit)
    is_index_fresh = status.state == "fresh"

    metrics_policy = resolve_metrics_policy()
    metrics_enabled = metrics_policy.metrics_enabled
    metrics_trace_enabled = metrics_policy.trace_enabled

    mcp_runtime = mcp_runtime_available(source_resolved)

    clients = discover_clients(source_resolved)
    client_names = [c.client for c in clients]

    agent_files = discover_agent_files(source_resolved)
    agent_file_strs = [
        str(f.relative_to(source_resolved)) if f.is_relative_to(source_resolved) else str(f)
        for f in agent_files
    ]

    return PreflightState(
        has_dot_archex=has_dot_archex,
        has_index=has_index,
        is_index_fresh=is_index_fresh,
        metrics_enabled=metrics_enabled,
        metrics_trace_enabled=metrics_trace_enabled,
        mcp_runtime_available=mcp_runtime,
        discovered_clients=client_names,
        discovered_agent_files=agent_file_strs,
    )


def apply_init_index(
    source: Path, preflight: PreflightState, dry_run: bool
) -> dict[str, list[dict[str, str]]]:
    """Apply project init and conditional indexing."""
    actions: list[dict[str, str]] = []

    if not preflight.has_dot_archex:
        actions.append({"type": "init", "status": "planned" if dry_run else "executed"})
        if not dry_run:
            init_project(str(source))
    else:
        actions.append({"type": "init", "status": "skipped_exists"})

    if not preflight.has_index or not preflight.is_index_fresh:
        actions.append({"type": "index", "status": "planned" if dry_run else "executed"})
        if not dry_run:
            run_indexing_and_get_summary(
                source=str(source),
                splade=False,
                module_prefilter=False,
                allow_remote_code=False,
                quantize_vectors=None,
                quantize_bits=None,
                export_artifact_path=None,
            )
    else:
        actions.append({"type": "index", "status": "skipped_fresh"})

    return {"init_index": actions}


@click.command("setup")
@click.argument(
    "source",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
)
@click.option("--dry-run", is_flag=True, help="Print plan without making changes.")
@click.option("--yes", is_flag=True, help="Execute default setup without prompting.")
@click.option("--clients/--no-clients", default=None, help="Install available clients.")
@click.option("--metrics/--no-metrics", default=None, help="Enable local metrics.")
@click.option("--hooks", is_flag=True, help="Install optional hooks.")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
def setup_cmd(
    source: Path,
    dry_run: bool,
    yes: bool,
    clients: bool | None,
    metrics: bool | None,
    hooks: bool,
    format_: Literal["text", "json"],
) -> None:
    """Guided onboarding wizard."""
    preflight = run_preflight(source)

    if (
        not sys.stdin.isatty()
        and not sys.stdout.isatty()
        and not yes
        and not dry_run
        and format_ != "json"
    ):
        click.echo("setup is interactive by default, but stdin/stdout are not TTY.", err=True)
        click.echo("Use --dry-run to print a plan, or pass --yes with explicit options.", err=True)
        sys.exit(1)

    if format_ == "json":
        actions: list[dict[str, Any]] = []
        if not preflight.has_dot_archex:
            actions.append({"type": "init", "status": "planned"})
        else:
            actions.append({"type": "init", "status": "skipped_exists"})
        if not preflight.has_index or not preflight.is_index_fresh:
            actions.append({"type": "index", "status": "planned"})
        else:
            actions.append({"type": "index", "status": "skipped_fresh"})

        plan: dict[str, Any] = {
            "preflight": asdict(preflight),
            "planned_actions": actions,
        }
        click.echo(json.dumps(plan, indent=2))
        return

    if dry_run:
        click.echo("--- Setup Preflight ---")
        for k, v in asdict(preflight).items():
            click.echo(f"{k}: {v}")
        click.echo("--- Planned Actions ---")
        if not preflight.has_dot_archex:
            click.echo("- Initialize repository")
        else:
            click.echo("- Repository initialized (skipped)")

        if not preflight.has_index:
            click.echo("- Build fresh index")
        elif not preflight.is_index_fresh:
            click.echo("- Refresh stale index")
        else:
            click.echo("- Index is fresh (skipped)")
        return
    if yes:
        click.echo("--- Executing Setup ---")
        results = apply_init_index(source, preflight, dry_run=False)
        for action in results["init_index"]:
            click.echo(f"- {action['type']}: {action['status']}")
        return

    click.echo("Interactive mode not fully implemented. Use --dry-run or --yes.")
    sys.exit(1)
