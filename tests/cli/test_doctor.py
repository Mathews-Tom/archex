from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from archex import doctor
from archex.cli.main import cli
from archex.doctor import DoctorCheck, DoctorReport, render_doctor_text
from archex.languages import LanguageSupport
from archex.metrics.health import record_metrics_failure
from archex.metrics.storage import metrics_db_path
from archex.models import LanguageTier
from archex.project import init_project


@pytest.fixture(autouse=True)
def _isolated_metrics_home(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


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


def test_doctor_json_reports_metrics_health(
    python_simple_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    init_project(python_simple_repo)
    record_metrics_failure("record", "disk full", db_path=metrics_db_path(home=tmp_path))
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output

    result = runner.invoke(cli, ["doctor", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    checks = {check["name"]: check for check in payload["checks"]}
    metrics = checks["metrics_health"]
    assert metrics["status"] == "warning"
    assert metrics["details"]["db_path"] == str(metrics_db_path(home=tmp_path))
    assert metrics["details"]["enabled"] is False
    assert metrics["details"]["trace_enabled"] is False
    assert metrics["details"]["raw_event_retention_days"] == 90
    assert metrics["details"]["trace_retention_days"] == 14
    assert metrics["details"]["latest_failure"] == "record: disk full"


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


def test_grammar_check_reports_structured_bucket_distinct_from_full_and_chunk_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_support = LanguageSupport(
        language_id="python",
        display_name="Python",
        extensions=(".py",),
        tier=LanguageTier.FULL,
        pack_name="python",
    )
    chunk_support = LanguageSupport(
        language_id="sql",
        display_name="SQL",
        extensions=(".sql",),
        tier=LanguageTier.CHUNK_ONLY,
        pack_name="sql",
    )
    structured_support = LanguageSupport(
        language_id="structured_stub",
        display_name="Structured Stub",
        extensions=(".structstub",),
        tier=LanguageTier.STRUCTURED,
        pack_name="structured_stub",
        chunk_node_types=frozenset({"section"}),
    )
    monkeypatch.setattr(
        doctor,
        "LANGUAGE_SUPPORT",
        {"python": full_support, "sql": chunk_support, "structured_stub": structured_support},
    )

    class _AlwaysLoadsEngine:
        def get_language(self, language_id: str) -> object:
            return object()

    monkeypatch.setattr(doctor, "TreeSitterEngine", _AlwaysLoadsEngine)

    check = doctor._grammar_check()  # pyright: ignore[reportPrivateUsage]

    assert check.status == "ok"
    assert check.details["full"] == {"available": 1, "total": 1}
    assert check.details["chunk_only"] == {"available": 1, "total": 1}
    assert check.details["structured"] == {"available": 1, "total": 1}


def test_grammar_check_missing_structured_grammar_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_support = LanguageSupport(
        language_id="python",
        display_name="Python",
        extensions=(".py",),
        tier=LanguageTier.FULL,
        pack_name="python",
    )
    structured_support = LanguageSupport(
        language_id="structured_stub",
        display_name="Structured Stub",
        extensions=(".structstub",),
        tier=LanguageTier.STRUCTURED,
        pack_name="structured_stub",
        chunk_node_types=frozenset({"section"}),
    )
    monkeypatch.setattr(
        doctor, "LANGUAGE_SUPPORT", {"python": full_support, "structured_stub": structured_support}
    )

    class _SelectiveEngine:
        def get_language(self, language_id: str) -> object:
            if language_id == "structured_stub":
                raise RuntimeError("no grammar available")
            return object()

    monkeypatch.setattr(doctor, "TreeSitterEngine", _SelectiveEngine)

    check = doctor._grammar_check()  # pyright: ignore[reportPrivateUsage]

    assert check.status == "warning"
    assert check.details["structured"] == {"available": 0, "total": 1}
    missing = check.details["missing"]
    assert isinstance(missing, dict)
    assert "structured_stub" in missing


def test_render_doctor_text_includes_structured_grammar_line() -> None:
    report = DoctorReport(
        repo_root=Path("/tmp/repo"),
        status="ok",
        checks=[
            DoctorCheck(
                name="grammars",
                status="ok",
                message="all declared tree-sitter grammars load",
                details={
                    "full": {"available": 3, "total": 3},
                    "chunk_only": {"available": 2, "total": 2},
                    "structured": {"available": 1, "total": 1},
                    "missing": {},
                },
            )
        ],
    )

    text = render_doctor_text(report)

    assert "full grammars: 3/3 available" in text
    assert "chunk-only grammars: 2/2 available" in text
    assert "structured grammars: 1/1 available" in text
