"""Project lifecycle status command.

Two modes over the same subject. The default inspects the index directly --
authoritative, and the point at which the cached status snapshot (R23) is
refreshed. `--cached` reads only that snapshot: no index open, no parse, and
therefore the mode a scripted or repeated caller should use.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.status import ProjectStatus, inspect_project_status
from archex.status_snapshot import (
    STATUS_SNAPSHOT_VERSION,
    ReceiptState,
    StatusState,
    StatusView,
    WatchState,
    clear_snapshot,
    publish_status,
    read_status,
    resolve_status_root,
    status_snapshot_path,
)

#: Inspected states that mean the index describes the current working tree.
_FRESH_STATES = frozenset({"fresh"})

#: Inspected states with no index to describe, where a leftover snapshot would
#: keep advertising a measurement of something that is gone.
_UNDESCRIBABLE_STATES = frozenset({"uninitialized", "missing_index", "corrupt"})


@click.command("status")
@click.argument("source", required=False, default=".")
@click.option("--strict", is_flag=True, default=False, help="Fail unless the index is fresh.")
@click.option(
    "--cached",
    is_flag=True,
    default=False,
    help="Read the cached status snapshot only; never opens the index.",
)
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def status_cmd(source: str, strict: bool, cached: bool, output_format: str) -> None:
    """Inspect repo-local archex project status without indexing."""
    if cached:
        _run_cached(source, strict=strict, output_format=output_format)
        return

    try:
        status = inspect_project_status(source)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _refresh_snapshot(status)

    if output_format == "json":
        click.echo(json.dumps(_status_payload(status), indent=2, sort_keys=True))
    else:
        _render_text(status)

    if status.state == "corrupt" or (strict and status.state != "fresh"):
        raise click.exceptions.Exit(1)


def _run_cached(source: str, *, strict: bool, output_format: str) -> None:
    """Render the cached snapshot for ``source`` without opening the index."""
    repo_root = resolve_status_root(source)
    view = read_status(repo_root)

    if output_format == "json":
        click.echo(json.dumps(_cached_payload(repo_root, view), indent=2, sort_keys=True))
    else:
        _render_cached_text(repo_root, view)

    unusable = view.state in {StatusState.CORRUPT, StatusState.UNSUPPORTED}
    if unusable or (strict and view.state is not StatusState.FRESH):
        raise click.exceptions.Exit(1)


def _refresh_snapshot(status: ProjectStatus) -> None:
    """Publish what this inspection just measured, or clear a stale claim.

    `archex status` is the command a user reaches for when a client's status
    surface looks wrong, so it is also the command that should repair it. It
    has already opened the index, so republishing costs nothing extra.
    """
    if status.state in _UNDESCRIBABLE_STATES:
        clear_snapshot(status.repo_root)
        return
    publish_status(
        status.repo_root,
        index_fresh=status.state in _FRESH_STATES,
        index_revision=status.index_revision,
        generation_id=status.generation_id,
        indexed_commit=status.indexed_commit,
        current_commit=status.current_commit,
        files_indexed=status.files_indexed,
        chunks_indexed=status.chunks_indexed,
        working_tree_dirty=status.working_tree != "clean",
        reindex_required=status.state == "needs_reindex",
    )


def _status_payload(status: ProjectStatus) -> dict[str, object]:
    return {
        "repo_root": str(status.repo_root),
        "initialized": status.initialized,
        "state": status.state,
        "index_path": str(status.index_path),
        "current_commit": status.current_commit,
        "indexed_commit": status.indexed_commit,
        "working_tree": status.working_tree,
        "files_indexed": status.files_indexed,
        "chunks_indexed": status.chunks_indexed,
        "languages": status.languages,
        "vector_index_available": status.vector_index_available,
        "generation_id": status.generation_id,
        "index_revision": status.index_revision,
        "dogfood_latest_path": (
            str(status.dogfood_latest_path) if status.dogfood_latest_path is not None else ""
        ),
        "error": status.error,
        "metrics_savings": status.metrics_savings,
    }


def _cached_payload(repo_root: Path, view: StatusView) -> dict[str, object]:
    return {
        "repo_root": str(repo_root),
        "snapshot_path": str(status_snapshot_path(repo_root)),
        "reader_version": STATUS_SNAPSHOT_VERSION,
        "state": view.state.value,
        "detail": view.detail,
        "age_seconds": view.age_seconds,
        "watch_state": view.watch_state.value,
        "snapshot": view.snapshot.model_dump(mode="json") if view.snapshot is not None else None,
    }


def _render_text(status: ProjectStatus) -> None:
    click.echo(f"Repository:         {status.repo_root}")
    click.echo(f"State:              {status.state}")
    click.echo(f"Initialized:        {'yes' if status.initialized else 'no'}")
    click.echo(f"Index path:         {status.index_path}")
    click.echo(f"Current commit:     {status.current_commit or 'none'}")
    click.echo(f"Indexed commit:     {status.indexed_commit or 'none'}")
    click.echo(f"Working tree:       {status.working_tree}")
    click.echo(f"Files indexed:      {status.files_indexed}")
    click.echo(f"Chunks indexed:     {status.chunks_indexed}")
    if status.languages:
        languages = ", ".join(f"{language}={count}" for language, count in status.languages.items())
    else:
        languages = "none"
    click.echo(f"Languages:          {languages}")
    click.echo(f"Vector index:       {'yes' if status.vector_index_available else 'no'}")
    if status.metrics_savings is not None and status.metrics_savings["event_count"] > 0:
        click.echo(f"Metrics saved:      {status.metrics_savings['tokens_saved']} tokens")
    if status.dogfood_latest_path is not None:
        click.echo(f"Dogfood latest:     {status.dogfood_latest_path}")
    if status.error:
        click.echo(f"Error:              {status.error}")


def _render_cached_text(repo_root: Path, view: StatusView) -> None:
    click.echo(f"Repository:         {repo_root}")
    click.echo(f"Cached state:       {view.state.value}")
    if view.detail:
        click.echo(f"Detail:             {view.detail}")
    snapshot = view.snapshot
    if snapshot is None:
        click.echo(f"Snapshot:           {status_snapshot_path(repo_root)}")
        click.echo("Remedy:             run `archex index` or `archex status`")
        return
    age = "unknown" if view.age_seconds is None else f"{view.age_seconds}s ago"
    click.echo(f"Measured:           {snapshot.written_at} ({age})")
    click.echo(f"Index revision:     {snapshot.index_revision or 'none'}")
    click.echo(f"Indexed commit:     {snapshot.indexed_commit or 'none'}")
    click.echo(f"Files indexed:      {snapshot.files_indexed}")
    click.echo(f"Chunks indexed:     {snapshot.chunks_indexed}")
    click.echo(f"Working tree:       {'dirty' if snapshot.working_tree_dirty else 'clean'}")
    click.echo(f"Reindex required:   {'yes' if snapshot.reindex_required else 'no'}")
    pending = f"{snapshot.pending_delta_files}"
    if not snapshot.pending_view_complete:
        pending = f"{pending} (incomplete list)"
    click.echo(f"Pending edits:      {pending}")
    if snapshot.last_edit_at:
        edit_source = snapshot.last_edit_client or "unknown client"
        click.echo(f"Last edit:          {snapshot.last_edit_at} via {edit_source}")
    click.echo(f"Last sync:          {snapshot.last_successful_sync or 'none'}")
    click.echo(f"Receipt:            {_receipt_label(snapshot.receipt_state)}")
    click.echo(f"Watch:              {_watch_label(view.watch_state)}")


def _receipt_label(state: ReceiptState) -> str:
    if state is ReceiptState.COMPLETE:
        return "complete"
    if state is ReceiptState.PARTIAL:
        return "partial (some recorded edits were dropped by a cap)"
    return "unknown (no persisted generation identity)"


def _watch_label(state: WatchState) -> str:
    if state is WatchState.ACTIVE:
        return "active (a watch refresh was observed recently)"
    return "unobserved (no recent watch refresh; a watcher may still be idle)"
