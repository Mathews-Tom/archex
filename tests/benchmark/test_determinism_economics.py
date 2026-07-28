from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

import archex.cli.benchmark_cmd as benchmark_module
from archex.benchmark.determinism_economics import (
    DeterminismEconomicsArtifact,
    OrderingArm,
    load_sessions,
    measure_economics,
    validate_determinism_economics_artifact,
)
from archex.cli.benchmark_cmd import benchmark_cmd

FIXTURE = Path("benchmarks/determinism_economics/sessions.json")
HASH = "a" * 40


def _artifact() -> DeterminismEconomicsArtifact:
    return measure_economics(
        load_sessions(FIXTURE),
        preregistration_commit=HASH,
        source_revision=HASH,
        generated_at="2026-07-29T00:00:00Z",
        session_fixture=str(FIXTURE),
        measurement_command="test command",
        resamples=100,
        seed=7,
    )


def test_measurement_uses_same_sessions_and_reports_cache_economics() -> None:
    artifact = _artifact()
    arms = {arm.arm: arm for arm in artifact.arms}

    assert arms[OrderingArm.DETERMINISTIC].cache_hit_rate == 0.0
    assert arms[OrderingArm.PERTURBED].cache_hit_rate == 0.0
    assert (
        arms[OrderingArm.DETERMINISTIC].input_cost_usd_per_resolved_task
        == arms[OrderingArm.PERTURBED].input_cost_usd_per_resolved_task
    )
    for arm in arms.values():
        assert (
            arm.cache_hit_rate_interval.low
            <= arm.cache_hit_rate
            <= arm.cache_hit_rate_interval.high
        )
        assert (
            arm.input_cost_usd_per_resolved_task_interval.low
            <= arm.input_cost_usd_per_resolved_task
            <= arm.input_cost_usd_per_resolved_task_interval.high
        )


def test_validator_rejects_missing_fixed_arm(tmp_path: Path) -> None:
    payload = _artifact().model_dump(mode="json")
    payload["arms"][0]["arm"] = "perturbed"
    evidence = tmp_path / "invalid.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="R6 arms"):
        validate_determinism_economics_artifact(evidence)


def test_validator_rejects_ledger_that_does_not_reproduce_from_fixture(tmp_path: Path) -> None:
    payload = _artifact().model_dump(mode="json")
    ledger = payload["arms"][0]["sessions"][0]
    prefix = ledger["rendered_prefixes"][0] + "tampered"
    ledger["rendered_prefixes"][0] = prefix
    ledger["prefix_sha256"][0] = hashlib.sha256(prefix.encode()).hexdigest()
    evidence = tmp_path / "tampered.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reproduce"):
        validate_determinism_economics_artifact(evidence)


def test_validator_rejects_interval_that_excludes_its_estimate() -> None:
    payload = _artifact().model_dump(mode="json")
    interval = payload["intervals"][0]
    interval["low_percent"] = interval["point_estimate_percent"] + 0.01
    interval["high_percent"] = interval["point_estimate_percent"] + 0.02

    with pytest.raises(ValidationError, match="point estimate"):
        DeterminismEconomicsArtifact.model_validate(payload)


def test_cli_validates_standalone_economics_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "s7.json"
    evidence.write_text(_artifact().model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        benchmark_cmd,
        ["validate", "--kind", "evidence", "--input", str(evidence)],
    )

    assert result.exit_code == 0, result.output
    assert "Valid determinism economics" in result.output


def test_cli_runs_frozen_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = tmp_path / "s7.json"

    def source_revision_stub(_root: Path) -> str:
        return HASH

    monkeypatch.setattr(benchmark_module, "source_revision", source_revision_stub)

    result = CliRunner().invoke(
        benchmark_cmd,
        [
            "determinism-economics",
            "--output",
            str(evidence),
            "--preregistration-commit",
            HASH,
            "--resamples",
            "100",
            "--pricing-retrieved-at",
            "2026-07-29T00:00:00Z",
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = validate_determinism_economics_artifact(evidence)
    assert artifact.preregistration_commit == HASH
