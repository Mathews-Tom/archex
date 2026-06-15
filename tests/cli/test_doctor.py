from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli
from archex.project import init_project


def test_doctor_json_reports_healthy_project(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output

    result = runner.invoke(cli, ["doctor", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["index_health"]["status"] == "ok"
    assert checks["index_staleness"]["status"] == "ok"
    assert checks["model_cache"]["details"]["required"] is False
    assert checks["grammars"]["details"]["full"]["available"] > 0
    assert checks["disk_usage"]["details"]["total_bytes"] > 0


def test_doctor_json_fails_on_corrupt_index(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    index_path = python_simple_repo / ".archex" / "index.db"
    index_path.write_text("not sqlite", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli, ["doctor", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["index_health"]["status"] == "error"
    assert checks["index_health"]["details"]["state"] == "corrupt"
    assert checks["index_staleness"]["status"] == "error"


def test_doctor_text_includes_required_sections(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output

    result = runner.invoke(cli, ["doctor", str(python_simple_repo)])

    assert result.exit_code == 0, result.output
    assert "archex doctor: ok" in result.output
    assert "index_health" in result.output
    assert "index_staleness" in result.output
    assert "model_cache" in result.output
    assert "grammars" in result.output
    assert "mcp_registration" in result.output
    assert "disk_usage" in result.output
