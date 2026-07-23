"""Tests for the CLI query subcommand's --profile flag."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from archex.cli.main import cli
from archex.models import (
    ContextBundle,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextReceipt,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
)


def _make_bundle(question: str = "how does auth work?") -> ContextBundle:
    return ContextBundle(
        query=question,
        token_count=10,
        token_budget=8000,
        receipt=ContextReceipt(
            query=question,
            token_budget=ContextReceiptTokenBudget(requested=8000, consumed=10),
            index_revision="rev",
            freshness=ContextFreshness.CLEAN,
            context_complete=ContextCompletenessStatus.COMPLETE,
            context_complete_reason=ContextCompletenessReason.COMPLETE,
            recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
        ),
    )


def test_profile_fast_disables_vector_module_prefilter_and_rerank(
    python_simple_repo: Path,
) -> None:
    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", return_value=_make_bundle()) as mock_query:
        result = runner.invoke(
            cli, ["query", str(python_simple_repo), "question", "--profile", "fast"]
        )
    assert result.exit_code == 0, result.output
    index_config = mock_query.call_args.kwargs["index_config"]
    assert index_config.vector is False
    assert index_config.module_prefilter is False
    assert index_config.rerank is False


def test_profile_deep_enables_vector_module_prefilter_and_rerank(
    python_simple_repo: Path,
) -> None:
    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", return_value=_make_bundle()) as mock_query:
        result = runner.invoke(
            cli, ["query", str(python_simple_repo), "question", "--profile", "deep"]
        )
    assert result.exit_code == 0, result.output
    index_config = mock_query.call_args.kwargs["index_config"]
    assert index_config.vector is True
    assert index_config.module_prefilter is True
    assert index_config.rerank is True


def test_profile_balanced_enables_only_module_prefilter(python_simple_repo: Path) -> None:
    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", return_value=_make_bundle()) as mock_query:
        result = runner.invoke(
            cli, ["query", str(python_simple_repo), "question", "--profile", "balanced"]
        )
    assert result.exit_code == 0, result.output
    index_config = mock_query.call_args.kwargs["index_config"]
    assert index_config.vector is False
    assert index_config.module_prefilter is True
    assert index_config.rerank is False


def test_no_profile_leaves_index_config_from_repo_settings(python_simple_repo: Path) -> None:
    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", return_value=_make_bundle()) as mock_query:
        result = runner.invoke(cli, ["query", str(python_simple_repo), "question"])
    assert result.exit_code == 0, result.output
    index_config = mock_query.call_args.kwargs["index_config"]
    assert index_config.vector is False
    assert index_config.module_prefilter is False
    assert index_config.rerank is False


def test_module_prefilter_flag_overrides_profile_fast(python_simple_repo: Path) -> None:
    """Individual flags still apply on top of a profile, per the documented order."""
    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", return_value=_make_bundle()) as mock_query:
        result = runner.invoke(
            cli,
            [
                "query",
                str(python_simple_repo),
                "question",
                "--profile",
                "fast",
                "--module-prefilter",
            ],
        )
    assert result.exit_code == 0, result.output
    index_config = mock_query.call_args.kwargs["index_config"]
    assert index_config.vector is False
    assert index_config.module_prefilter is True
    assert index_config.rerank is False


def test_invalid_profile_choice_rejected(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["query", str(python_simple_repo), "question", "--profile", "ultra"]
    )
    assert result.exit_code != 0
    assert "Invalid value for '--profile'" in result.output
