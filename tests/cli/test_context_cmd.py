"""Tests for the CLI `context` subcommand — the primary agent-facing facade."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def test_context_json_default_returns_full_envelope(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["context", str(python_simple_repo), "how does authentication work?"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    for key in (
        "content",
        "candidate_map",
        "fetch_handles",
        "relation_paths",
        "route",
        "receipt",
        "next_action",
    ):
        assert key in parsed
    assert len(parsed["content"]) > 0
    assert len(parsed["candidate_map"]) > 0
    assert parsed["fetch_handles"]
    assert parsed["route"]["intent_source"] == "auto"
    assert parsed["receipt"]["index_revision"]


def test_context_markdown_format(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "context",
            str(python_simple_repo),
            "how does authentication work?",
            "--format",
            "markdown",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Route" in result.output
    assert "## Candidate map" in result.output
    assert "## Selected code" in result.output


def test_context_intent_pins_budget(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["context", str(python_simple_repo), "anything", "--intent", "debugging"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["route"]["resolved_intent"] == "debugging"
    assert parsed["route"]["intent_source"] == "explicit"
    assert parsed["route"]["budget_source"] == "intent_default"


def test_context_invalid_intent_rejected(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["context", str(python_simple_repo), "anything", "--intent", "not_a_real_intent"],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--intent'" in result.output


def test_context_invalid_profile_rejected(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["context", str(python_simple_repo), "anything", "--profile", "ultra"],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--profile'" in result.output


def test_context_non_positive_budget_rejected(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["context", str(python_simple_repo), "anything", "--budget", "0"],
    )
    assert result.exit_code != 0
    assert "invalid context request" in result.output


def test_context_exclude_filter_removes_matching_files(python_simple_repo: Path) -> None:
    runner = CliRunner()
    baseline = runner.invoke(
        cli, ["context", str(python_simple_repo), "how does authentication work?"]
    )
    assert baseline.exit_code == 0, baseline.output
    excluded_path = json.loads(baseline.output)["candidate_map"][0]["file_path"]

    filtered = runner.invoke(
        cli,
        [
            "context",
            str(python_simple_repo),
            "how does authentication work?",
            "--exclude",
            excluded_path,
        ],
    )
    assert filtered.exit_code == 0, filtered.output
    parsed = json.loads(filtered.output)
    assert all(item["file_path"] != excluded_path for item in parsed["candidate_map"])
    assert parsed["route"]["filters_active"] is True
    assert any(
        skipped["reason"] == "filter_excluded"
        for skipped in parsed["receipt"]["skipped_candidates"]
    )


def test_context_handle_bypasses_broad_search(python_simple_repo: Path) -> None:
    runner = CliRunner()
    baseline = runner.invoke(
        cli, ["context", str(python_simple_repo), "how does authentication work?"]
    )
    assert baseline.exit_code == 0, baseline.output
    handle = json.loads(baseline.output)["fetch_handles"][0]

    fetched = runner.invoke(
        cli, ["context", str(python_simple_repo), "ignored", "--handle", handle]
    )
    assert fetched.exit_code == 0, fetched.output
    parsed = json.loads(fetched.output)
    assert parsed["route"]["handles_mode"] is True
    assert parsed["fetch_handles"] == [handle]


def test_context_no_model_first_use(python_simple_repo: Path) -> None:
    """The primary agent path reaches a useful result on a plain, model-free invocation."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["context", str(python_simple_repo), "how does archex score and rank candidates"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert len(parsed["candidate_map"]) > 0
    assert parsed["receipt"]["freshness"] in {"clean", "unknown"}
