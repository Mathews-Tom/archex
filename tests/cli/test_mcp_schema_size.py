from __future__ import annotations

import json

from click.testing import CliRunner

from archex.cli.main import cli


def test_mcp_schema_size_default_text_format() -> None:
    """R5 changed the bare command's meaning: it reports the gated default, which
    is what `archex mcp` really advertises to a fresh session."""
    result = CliRunner().invoke(cli, ["mcp-schema-size"])
    assert result.exit_code == 0
    assert "Scope: disclosure (gated default)" in result.output
    assert "Total serialized schema size:" in result.output
    assert "query_repo:" in result.output


def test_mcp_schema_size_ungated_text_format() -> None:
    """`--no-disclosure` keeps the pre-R5 reading available."""
    result = CliRunner().invoke(cli, ["mcp-schema-size", "--no-disclosure"])
    assert result.exit_code == 0
    assert "Scope: all" in result.output
    assert "get_impact:" in result.output


def test_mcp_schema_size_json_format_reports_full_report() -> None:
    result = CliRunner().invoke(cli, ["mcp-schema-size", "--no-disclosure", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["scope"] == "all"
    assert payload["tool_count"] == len(payload["per_tool_chars"])
    assert payload["total_chars"] == sum(payload["per_tool_chars"].values())


def test_mcp_schema_size_scoped_is_strictly_smaller_than_all() -> None:
    full = json.loads(
        CliRunner().invoke(cli, ["mcp-schema-size", "--no-disclosure", "--format", "json"]).output
    )
    scoped = json.loads(
        CliRunner().invoke(cli, ["mcp-schema-size", "--tools", "core", "--format", "json"]).output
    )
    assert scoped["scope"] == "core"
    assert scoped["tool_count"] < full["tool_count"]
    assert scoped["total_chars"] < full["total_chars"]
    assert set(scoped["per_tool_chars"]).issubset(set(full["per_tool_chars"]))


def test_mcp_schema_size_explicit_allowlist() -> None:
    result = CliRunner().invoke(
        cli, ["mcp-schema-size", "--tools", "query_repo,context", "--format", "json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tool_count"] == 2
    assert set(payload["per_tool_chars"]) == {"query_repo", "context"}


def test_mcp_schema_size_unknown_tool_name_fails_cleanly() -> None:
    result = CliRunner().invoke(cli, ["mcp-schema-size", "--tools", "not_a_real_tool"])
    assert result.exit_code != 0
    assert "Unknown MCP tool name" in result.output
