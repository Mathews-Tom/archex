from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli
from archex.project import init_project

if TYPE_CHECKING:
    import pytest


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
    assert checks["model_security"]["status"] == "ok"
    assert checks["model_security"]["details"]["allow_remote_code"] is False
    assert checks["model_security"]["details"]["embedding"]["enabled"] is False


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
    assert "model_security" in result.output
    assert "allow_remote_code: False" in result.output


def test_doctor_security_reports_remote_code_block(
    python_simple_repo: Path,
) -> None:
    init_project(python_simple_repo)
    settings = python_simple_repo / ".archex" / "settings.toml"
    settings.write_text(
        """\
[index]
cache_dir = ".archex"
vector = true
embedder = "nomic"
""",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["doctor", str(python_simple_repo), "--security", "--format", "json"],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert [check["name"] for check in payload["checks"]] == ["model_security"]
    details = payload["checks"][0]["details"]
    assert details["allow_remote_code"] is False
    assert details["embedding"]["provider"] == "nomic"
    assert details["embedding"]["requires_remote_code"] is True
    assert details["embedding"]["model_revision"] == "11114029805cee545ef111d5144b623787462a52"


def test_doctor_security_reports_remote_code_opt_in_and_cache_state(
    python_simple_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(python_simple_repo)
    settings = python_simple_repo / ".archex" / "settings.toml"
    settings.write_text(
        """\
[index]
cache_dir = ".archex"
vector = true
embedder = "nomic"
allow_remote_code = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["doctor", str(python_simple_repo), "--security", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "warning"
    details = payload["checks"][0]["details"]
    assert details["allow_remote_code"] is True
    assert details["embedding"]["remote_code_allowed"] is True
    assert details["embedding"]["cache_present"] is False
    assert details["network_downloads_required"] == ["nomic"]


def test_doctor_security_reports_vector_without_embedder_as_no_model(
    python_simple_repo: Path,
) -> None:
    init_project(python_simple_repo)
    settings = python_simple_repo / ".archex" / "settings.toml"
    settings.write_text(
        """\
[index]
cache_dir = ".archex"
vector = true
""",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["doctor", str(python_simple_repo), "--security", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    details = payload["checks"][0]["details"]
    assert details["embedding"]["provider"] == "none"
    assert details["embedding"]["model"] is None
    assert details["embedding"]["vector_requested"] is True
    assert details["network_downloads_required"] == []
