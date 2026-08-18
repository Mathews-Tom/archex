"""Explicit CLI operations for the repo-local project-session ledger."""

from __future__ import annotations

import json

import click

from archex.session import (
    DEFAULT_SESSION_TOKEN_BUDGET,
    SessionRecordKind,
    capture_session_record,
    delete_session_record,
    invalidate_session_record,
    list_session_records,
    render_session_primer,
)

_RECORD_KIND_CHOICES = [kind.value for kind in SessionRecordKind]


@click.group("session")
def session_cmd() -> None:
    """Capture and render explicit local project-session context."""


@session_cmd.command("record")
@click.argument("kind", type=click.Choice(_RECORD_KIND_CHOICES))
@click.argument("content")
@click.option("--source", default=".", show_default=True, help="Local Git repository path.")
@click.option("--file-path", default=None, help="Optional relative repository file anchor.")
@click.option("--symbol-id", default=None, help="Optional indexed symbol identifier anchor.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default="json")
def record_cmd(
    kind: str,
    content: str,
    source: str,
    file_path: str | None,
    symbol_id: str | None,
    output_format: str,
) -> None:
    """Persist one explicit decision, task, blocker, or rationale record."""
    del output_format
    try:
        record = capture_session_record(
            source,
            kind=SessionRecordKind(kind),
            content=content,
            creator="cli",
            file_path=file_path,
            symbol_id=symbol_id,
        )
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@session_cmd.command("list")
@click.option("--source", default=".", show_default=True, help="Local Git repository path.")
@click.option(
    "--all",
    "include_inactive",
    is_flag=True,
    default=False,
    help="Include inactive records.",
)
def list_cmd(source: str, include_inactive: bool) -> None:
    """List explicit records scoped to the current worktree and branch."""
    try:
        records = list_session_records(source, include_inactive=include_inactive)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps([record.model_dump(mode="json") for record in records], indent=2))


@session_cmd.command("invalidate")
@click.argument("record_id")
@click.option("--source", default=".", show_default=True, help="Local Git repository path.")
def invalidate_cmd(record_id: str, source: str) -> None:
    """Invalidate one active record without erasing its audit trail."""
    try:
        record = invalidate_session_record(source, record_id)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@session_cmd.command("delete")
@click.argument("record_id")
@click.option("--source", default=".", show_default=True, help="Local Git repository path.")
@click.option("--force", is_flag=True, default=False, help="Confirm permanent deletion.")
def delete_cmd(record_id: str, source: str, force: bool) -> None:
    """Permanently delete one explicitly identified session record."""
    if not force:
        raise click.UsageError("session delete requires --force")
    try:
        delete_session_record(source, record_id)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({"deleted": record_id}, indent=2))


@session_cmd.command("prime")
@click.argument("source", required=False, default=".")
@click.option(
    "--budget",
    type=int,
    default=DEFAULT_SESSION_TOKEN_BUDGET,
    show_default=True,
    help="Hard token budget for rendered project-session context.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
)
def prime_cmd(source: str, budget: int, output_format: str) -> None:
    """Render bounded session context; stale indexes deliberately return no context."""
    try:
        primer = render_session_primer(source, token_budget=budget)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "markdown":
        click.echo(primer.content, nl=False)
        return
    click.echo(json.dumps(primer.model_dump(mode="json"), indent=2))
