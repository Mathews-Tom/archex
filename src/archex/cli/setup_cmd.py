from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import click

from archex.cli.indexing import run_indexing_and_get_summary
from archex.client_setup import (
    ClientName,
    append_agent_guidance,
    build_discovered_install_plans,
    build_hook_install_plan,
    discover_agent_files,
    discover_clients,
    write_client_install_plan,
    write_hook_install_plan,
)
from archex.doctor import mcp_runtime_available
from archex.integrations.mcp import resolve_tool_scope
from archex.metrics.policy import resolve_metrics_policy, set_metrics_enabled, set_trace_enabled
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
) -> dict[str, list[dict[str, str | bool]]]:
    """Apply project init and conditional indexing."""
    actions: list[dict[str, str | bool]] = []

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


def apply_clients_guidance(
    source: Path,
    preflight: PreflightState,
    dry_run: bool,
    clients_flag: bool | None,
    tool_scope: str | None = None,
) -> dict[str, list[dict[str, str | bool]]]:
    """Apply MCP client registration and agent guidance.

    Both registration and guidance are opt-in: an unset ``clients_flag``
    (``None``, e.g. ``setup --yes`` without ``--clients``) skips both, the
    same way ``apply_metrics_hooks`` treats an unset ``--metrics``/``--hooks``
    flag as skip rather than a silent default-on write.
    """
    actions: list[dict[str, str | bool]] = []

    if not preflight.mcp_runtime_available:
        should_configure_clients = False
        actions.append({"type": "client_install", "status": "skipped_mcp_unstartable"})
    elif not clients_flag:
        should_configure_clients = False
        actions.append({"type": "client_install", "status": "skipped_by_flag"})
    else:
        should_configure_clients = True
        clients = discover_clients(source)
        plans = build_discovered_install_plans(clients, source, tool_scope=tool_scope)
        if not plans:
            actions.append({"type": "client_install", "status": "skipped_no_clients"})
        else:
            for plan in plans:
                action: dict[str, str | bool] = {
                    "type": "client_install",
                    "client": plan.client,
                    "status": "planned" if dry_run else "executed",
                }
                actions.append(action)
                if not dry_run:
                    try:
                        write_client_install_plan(plan)
                    except ValueError:
                        action["status"] = "skipped_already_configured"

    if not should_configure_clients:
        actions.append(
            {
                "type": "agent_guidance",
                "status": "skipped_mcp_unstartable"
                if not preflight.mcp_runtime_available
                else "skipped_by_flag",
            }
        )
    else:
        agent_files = discover_agent_files(source)
        if not agent_files:
            actions.append({"type": "agent_guidance", "status": "skipped_no_files"})
        else:
            for agent_file in agent_files:
                action: dict[str, str | bool] = {
                    "type": "agent_guidance",
                    "file": str(agent_file),
                    "status": "planned" if dry_run else "executed",
                }
                actions.append(action)
                if not dry_run and not append_agent_guidance(agent_file):
                    action["status"] = "skipped_already_present"

    return {"clients_guidance": actions}


def apply_metrics_hooks(
    source: Path,
    preflight: PreflightState,
    dry_run: bool,
    metrics_flag: bool | None,
    hooks_flag: bool,
) -> dict[str, list[dict[str, str | bool]]]:
    actions: list[dict[str, str | bool]] = []

    if metrics_flag is not None:
        actions.append(
            {
                "type": "metrics_config",
                "enabled": metrics_flag,
                "status": "planned" if dry_run else "executed",
            }
        )
        if not dry_run:
            set_metrics_enabled(metrics_flag)
            set_trace_enabled(False)
    else:
        actions.append({"type": "metrics_config", "status": "skipped_by_flag"})

    if not hooks_flag:
        actions.append({"type": "hooks_install", "status": "skipped_by_flag"})
    elif not preflight.discovered_clients:
        actions.append({"type": "hooks_install", "status": "skipped_no_clients"})
    else:
        for client_name in preflight.discovered_clients:
            action: dict[str, str | bool] = {
                "type": "hook_install",
                "client": client_name,
                "status": "planned" if dry_run else "executed",
            }
            actions.append(action)
            if not dry_run:
                try:
                    plan = build_hook_install_plan(
                        cast("ClientName", client_name), source, action="install"
                    )
                except ValueError as exc:
                    action["status"] = "skipped_unsupported"
                    action["error"] = str(exc)
                else:
                    write_hook_install_plan(plan)

    return {"metrics_hooks": actions}


def print_final_summary(
    results: dict[str, list[dict[str, str | bool]]], preflight: PreflightState
) -> None:
    click.echo("\n--- Setup Complete ---")
    click.echo("Your archex project is ready to use.")
    click.echo("\nNext commands:")
    click.echo('  archex query "How does this work?"')
    if not preflight.mcp_runtime_available:
        click.echo("\nNote: MCP clients cannot be used until the 'mcp' extra is installed.")
        click.echo("Fix for uv tool users:\n  uv tool install --force 'archex[mcp]'")


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
@click.option(
    "--tool-scope",
    default=None,
    help=(
        "Scope the tools any registered `archex mcp` server advertises: "
        "'all' (default, every tool), a named profile ('core' excludes "
        "the graph_* cluster, 'graph' is only the graph_* cluster), or a "
        "comma-separated explicit tool-name allowlist."
    ),
)
def setup_cmd(
    source: Path,
    dry_run: bool,
    yes: bool,
    clients: bool | None,
    metrics: bool | None,
    hooks: bool,
    format_: Literal["text", "json"],
    tool_scope: str | None,
) -> None:
    """Guided onboarding wizard."""
    if tool_scope is not None:
        try:
            resolve_tool_scope(tool_scope)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
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
            "clients_guidance": apply_clients_guidance(
                source, preflight, dry_run=True, clients_flag=clients, tool_scope=tool_scope
            )["clients_guidance"],
            "metrics_hooks": apply_metrics_hooks(
                source, preflight, dry_run=True, metrics_flag=metrics, hooks_flag=hooks
            )["metrics_hooks"],
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

        click.echo("--- Clients & Agent Guidance ---")
        cg_actions = apply_clients_guidance(
            source, preflight, dry_run=True, clients_flag=clients, tool_scope=tool_scope
        )["clients_guidance"]
        for action in cg_actions:
            name = action.get("client") or action.get("file") or action["type"]
            click.echo(f"- {name}: {action['status']}")

        click.echo("--- Metrics & Hooks ---")
        mh_actions = apply_metrics_hooks(
            source, preflight, dry_run=True, metrics_flag=metrics, hooks_flag=hooks
        )["metrics_hooks"]
        for action in mh_actions:
            name = action.get("client") or action["type"]
            click.echo(f"- {name}: {action['status']}")
        return
    if yes:
        click.echo("--- Executing Setup ---")
        results = apply_init_index(source, preflight, dry_run=False)
        cg_results = apply_clients_guidance(
            source, preflight, dry_run=False, clients_flag=clients, tool_scope=tool_scope
        )
        mh_results = apply_metrics_hooks(
            source, preflight, dry_run=False, metrics_flag=metrics, hooks_flag=hooks
        )
        for action in (
            results["init_index"] + cg_results["clients_guidance"] + mh_results["metrics_hooks"]
        ):
            name = action.get("client") or action.get("file") or action["type"]
            click.echo(f"- {name}: {action['status']}")
        print_final_summary(results, preflight)
        return

    click.echo("\n--- archex Setup ---")
    results: dict[str, list[dict[str, str | bool]]] = {
        "init_index": [],
        "clients_guidance": [],
        "metrics_hooks": [],
    }

    if not preflight.has_dot_archex:
        if click.confirm("Initialize archex repository?", default=True):
            init_project(str(source))
            results["init_index"].append({"type": "init", "status": "executed"})
        else:
            results["init_index"].append({"type": "init", "status": "skipped_interactive"})
    else:
        click.echo("Repository already initialized.")
        results["init_index"].append({"type": "init", "status": "skipped_exists"})

    if not preflight.has_index or not preflight.is_index_fresh:
        if click.confirm("Build or refresh index?", default=True):
            run_indexing_and_get_summary(
                source=str(source),
                splade=False,
                module_prefilter=False,
                allow_remote_code=False,
                quantize_vectors=None,
                quantize_bits=None,
                export_artifact_path=None,
            )
            results["init_index"].append({"type": "index", "status": "executed"})
        else:
            results["init_index"].append({"type": "index", "status": "skipped_interactive"})
    else:
        click.echo("Index is fresh.")
        results["init_index"].append({"type": "index", "status": "skipped_fresh"})

    if not preflight.mcp_runtime_available:
        click.echo("\nWarning: archex mcp runtime is not available.")
        click.echo("MCP clients cannot be used until the 'mcp' extra is installed.")

    # Hooks — offered first: zero context cost, augments existing Grep/Glob
    # calls instead of registering a new tool surface. Default on when a
    # supported client is discovered, since it has no downside worth an
    # explicit opt-out.
    do_hooks = hooks
    if not do_hooks and preflight.discovered_clients:
        click.echo(
            "\nHooks quietly augment your client's existing Grep/Glob calls with archex "
            "results — no new tool schema, no added context cost per turn."
        )
        do_hooks = click.confirm("Install optional shell/editor hooks?", default=True)

    # Clients (MCP) — registers the full 19-tool surface (context, query,
    # scout, graph inspection via graph_query or the five graph_* tools,
    # impact analysis, etc.). Every tool's schema is resent on every turn
    # regardless of use, so this is worth it when you want that full
    # surface, not just grep/glob augmentation. Pass --tool-scope (e.g.
    # 'core' -- excludes the five raw graph_* tools but keeps the single
    # graph_query dispatch tool -- 'graph', or an explicit tool-name
    # allowlist) to register a narrower surface and cut that per-turn cost.
    do_clients = clients
    if do_clients is None and preflight.discovered_clients and preflight.mcp_runtime_available:
        click.echo(
            "\nMCP registers all 19 archex tools with your client (context, query, scout, "
            "graph inspection, impact analysis, and more) — the richest surface, at the cost "
            "of resending every tool's schema on every turn. Use --tool-scope core to drop "
            "the five raw graph_* tools in favor of the lighter graph_query dispatch tool, "
            "or pass an explicit allowlist for a narrower surface still."
        )
        do_clients = click.confirm(
            f"Configure {len(preflight.discovered_clients)} discovered MCP clients?",
            default=True,
        )

    cg_results = apply_clients_guidance(
        source, preflight, dry_run=False, clients_flag=do_clients, tool_scope=tool_scope
    )
    results["clients_guidance"].extend(cg_results["clients_guidance"])

    # Metrics
    do_metrics = metrics
    if do_metrics is None:
        click.echo("\nUsage metrics are anonymous and local-only.")
        do_metrics = click.confirm("Enable local usage metrics?", default=False)

    mh_results = apply_metrics_hooks(
        source, preflight, dry_run=False, metrics_flag=do_metrics, hooks_flag=do_hooks
    )
    results["metrics_hooks"].extend(mh_results["metrics_hooks"])

    print_final_summary(results, preflight)
