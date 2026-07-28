"""Guards over the checked-in R4 corpus validity artifact.

R4's verification names the validate command; nothing ran it, so a later edit
could relax the audit or record an unvalidated power projection with nothing
noticing. These run on every suite invocation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from archex.benchmark.corpus_audit import (
    CorpusAuditError,
    validate_corpus_audit_artifact,
)
from archex.cli.benchmark_cmd import benchmark_cmd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "benchmarks" / "evidence" / "s2-corpus-validity.json"


def _mutated(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    payload: dict[str, Any] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_checked_in_artifact_validates() -> None:
    artifact = validate_corpus_audit_artifact(ARTIFACT)
    assert artifact.milestone == "R4"
    assert artifact.total_tasks > 0
    assert artifact.calibration["within_tolerance"] is True


def test_the_artifact_records_the_corpus_it_audited() -> None:
    """A projection about a different corpus than the one on disk is worthless."""
    artifact = validate_corpus_audit_artifact(ARTIFACT)
    payload: dict[str, Any] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact.tasks_dir == "benchmarks/tasks"
    assert len(payload["clustering"]["cluster_sizes"]) == payload["clustering"]["cluster_count"]
    assert sum(payload["clustering"]["cluster_sizes"].values()) == artifact.total_tasks


def test_a_calibration_verdict_contradicting_its_own_numbers_is_rejected(
    tmp_path: Path,
) -> None:
    """The schema recomputes the tolerance rather than trusting the flag."""

    def flip(payload: dict[str, Any]) -> None:
        payload["calibration"]["within_tolerance"] = False

    with pytest.raises(CorpusAuditError, match="implies True"):
        validate_corpus_audit_artifact(_mutated(tmp_path, flip))


def test_a_genuinely_failed_calibration_cannot_be_recorded(tmp_path: Path) -> None:
    """A power projection whose simulator was never validated means nothing."""

    def fail(payload: dict[str, Any]) -> None:
        payload["calibration"]["simulated_ci_width"] = 999.0
        payload["calibration"]["within_tolerance"] = False

    with pytest.raises(CorpusAuditError, match="did not reproduce the reference interval"):
        validate_corpus_audit_artifact(_mutated(tmp_path, fail))


def test_a_calibration_claiming_tolerance_it_does_not_have_is_rejected(
    tmp_path: Path,
) -> None:
    """The failure mode a miscomputing producer would actually emit."""

    def lie(payload: dict[str, Any]) -> None:
        payload["calibration"]["simulated_ci_width"] = 999.0

    with pytest.raises(CorpusAuditError, match="implies False"):
        validate_corpus_audit_artifact(_mutated(tmp_path, lie))


def test_cluster_sizes_must_sum_to_the_reported_task_count(tmp_path: Path) -> None:
    """An audit whose blocks disagree about the corpus is not evidence."""

    def drop(payload: dict[str, Any]) -> None:
        payload["clustering"]["cluster_sizes"].popitem()

    with pytest.raises(CorpusAuditError, match="disagree"):
        validate_corpus_audit_artifact(_mutated(tmp_path, drop))


def test_a_leak_rate_inconsistent_with_its_own_task_list_is_rejected(
    tmp_path: Path,
) -> None:
    def tamper(payload: dict[str, Any]) -> None:
        payload["leakage"]["symbol_leak_rate"] = 0.01

    with pytest.raises(CorpusAuditError, match="does not match its own symbol-tier"):
        validate_corpus_audit_artifact(_mutated(tmp_path, tamper))


def test_a_missing_calibration_field_is_rejected(tmp_path: Path) -> None:
    def drop(payload: dict[str, Any]) -> None:
        del payload["calibration"]["measured_ci_width"]

    with pytest.raises(CorpusAuditError, match="measured_ci_width"):
        validate_corpus_audit_artifact(_mutated(tmp_path, drop))


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    def add(payload: dict[str, Any]) -> None:
        payload["conclusion"] = "looks fine"

    with pytest.raises(CorpusAuditError, match="failed validation"):
        validate_corpus_audit_artifact(_mutated(tmp_path, add))


def test_a_short_source_revision_is_rejected(tmp_path: Path) -> None:
    def truncate(payload: dict[str, Any]) -> None:
        payload["source_revision"] = "abc1234"

    with pytest.raises(CorpusAuditError, match="failed validation"):
        validate_corpus_audit_artifact(_mutated(tmp_path, truncate))


def test_a_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusAuditError, match="not a file"):
        validate_corpus_audit_artifact(tmp_path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusAuditError, match="not readable JSON"):
        validate_corpus_audit_artifact(path)


def test_cli_validates_the_checked_in_artifact() -> None:
    result = CliRunner().invoke(
        benchmark_cmd, ["validate", "--kind", "corpus-audit", "--input", str(ARTIFACT)]
    )
    assert result.exit_code == 0, result.output
    assert "Valid corpus audit" in result.output


def test_cli_requires_input() -> None:
    result = CliRunner().invoke(benchmark_cmd, ["validate", "--kind", "corpus-audit"])
    assert result.exit_code != 0
    assert "--input is required" in result.output
