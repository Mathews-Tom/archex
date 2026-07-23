"""CLI context subcommand: the primary agent-facing context() facade."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import click
from pydantic import ValidationError

from archex.api import context, get_files_token_count
from archex.context_facade import (
    ContextBudgets,
    ContextFilters,
    ContextRequest,
    ContextResult,
    render_context_markdown,
)
from archex.exceptions import ArchexError
from archex.metrics.capture import record_query_usage
from archex.metrics.health import note_metrics_recording_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import RetrievalProfile
from archex.serve.intent import QueryIntent
from archex.utils import resolve_source

if TYPE_CHECKING:
    from archex.models import Config, RepoSource

logger = logging.getLogger(__name__)

_INTENT_CHOICES = [intent.value for intent in QueryIntent]
_PROFILE_CHOICES = [profile.value for profile in RetrievalProfile]


@click.command("context")
@click.argument("args", nargs=-1, required=True)
@click.option(
    "--intent",
    type=click.Choice(_INTENT_CHOICES),
    default=None,
    help="Pin the query intent instead of auto-classifying it from the query text.",
)
@click.option(
    "--profile",
    type=click.Choice(_PROFILE_CHOICES),
    default=None,
    help="Named retrieval profile: 'fast', 'balanced', or 'deep'. Omit for the repo default.",
)
@click.option(
    "--include",
    "include_paths",
    multiple=True,
    help="Glob pattern(s) a candidate's file path must match (fnmatch, e.g. 'src/auth/**').",
)
@click.option(
    "--exclude",
    "exclude_paths",
    multiple=True,
    help="Glob pattern(s) that exclude a candidate by file path.",
)
@click.option(
    "-l",
    "--language",
    "languages",
    multiple=True,
    help="Restrict returned candidates to these languages (post-retrieval, exact match).",
)
@click.option(
    "--budget",
    type=int,
    default=None,
    help="Explicit token budget override. Omit to resolve from --intent or query()'s auto-scaling.",
)
@click.option(
    "--handle",
    "handles",
    multiple=True,
    help="Exact fetch handle(s) — bypasses broad search and returns exactly these candidates.",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def context_cmd(
    args: tuple[str, ...],
    intent: str | None,
    profile: str | None,
    include_paths: tuple[str, ...],
    exclude_paths: tuple[str, ...],
    languages: tuple[str, ...],
    budget: int | None,
    handles: tuple[str, ...],
    output_format: str,
) -> None:
    """Retrieve the primary agent-facing context result for a repository question.

    Returns a compact candidate map, exact fetch handles, selected code,
    relation paths, the route decision, the receipt, and a recommended next
    action — a thin facade over `archex query`. Existing specialized
    commands (`query`, `scout`, `symbol`, ...) remain fully supported.
    """
    from archex.config import load_config, load_index_config

    source, question = _source_and_question(args)
    repo_source = resolve_source(source)
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)

    try:
        request = ContextRequest(
            query=question,
            intent=QueryIntent(intent) if intent is not None else None,
            profile=RetrievalProfile(profile) if profile is not None else None,
            filters=ContextFilters(
                include_paths=list(include_paths),
                exclude_paths=list(exclude_paths),
                languages=list(languages),
            ),
            budgets=ContextBudgets(token_budget=budget),
            handles=list(handles),
        )
    except ValidationError as exc:
        raise click.ClickException(f"invalid context request: {exc}") from exc

    try:
        result = context(repo_source, request, config=config, index_config=index_config)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_render(result, output_format))
    _record_metrics(repo_source, result, config)


def _render(result: ContextResult, output_format: str) -> str:
    if output_format == "markdown":
        return render_context_markdown(result)
    envelope = {
        "content": [chunk.model_dump(mode="json") for chunk in result.selected_code],
        "candidate_map": [item.model_dump(mode="json") for item in result.candidate_map],
        "fetch_handles": result.fetch_handles,
        "relation_paths": result.relation_paths.model_dump(mode="json"),
        "route": result.route.model_dump(mode="json"),
        "receipt": result.receipt.model_dump(mode="json") if result.receipt else None,
        "next_action": result.next_action.value if result.next_action else None,
    }
    return json.dumps(envelope, indent=2)


def _record_metrics(repo_source: RepoSource, result: ContextResult, config: Config) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        unique_files = list({c.chunk.file_path for c in result.bundle.chunks})
        raw_tokens = get_files_token_count(repo_source, unique_files, config)
        whole_repo_tokens = _repo_total_tokens(repo_source, config)
        record_query_usage(
            repo_source,
            result.bundle,
            surface="cli",
            tool_name="context",
            tokens_raw_equivalent=raw_tokens,
            whole_repo_tokens=whole_repo_tokens,
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def _repo_total_tokens(repo_source: RepoSource, config: Config) -> int | None:
    try:
        from archex.api import get_repo_total_tokens

        return get_repo_total_tokens(repo_source, config=config)
    except ArchexError:
        return None


def _source_and_question(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) >= 2:
        return args[0], " ".join(args[1:])
    raise click.UsageError("context requires a question")
