from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

import archex.benchmark.determinism_economics as economics
import archex.cli.benchmark_cmd as benchmark_module
from archex.benchmark.determinism_economics import (
    DeterminismEconomicsArtifact,
    OrderingArm,
    PricingSchedule,
    SessionFixture,
    SessionLedger,
    SessionTurn,
    load_sessions,
    measure_economics,
    validate_determinism_economics_artifact,
)
from archex.cli.benchmark_cmd import benchmark_cmd

FIXTURE = Path("benchmarks/determinism_economics/sessions.json")
PREREGISTRATION_COMMIT = "501636abb09cbcf6edee5783305af6bcb313606a"
UNKNOWN_COMMIT = "a" * 40
SOURCE_REVISION = "b73ba365be43bf73ad566e96161bb8e045faa843"


def _artifact() -> DeterminismEconomicsArtifact:
    return measure_economics(
        load_sessions(FIXTURE),
        preregistration_commit=PREREGISTRATION_COMMIT,
        source_revision=SOURCE_REVISION,
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
        return SOURCE_REVISION

    monkeypatch.setattr(benchmark_module, "source_revision", source_revision_stub)

    result = CliRunner().invoke(
        benchmark_cmd,
        [
            "determinism-economics",
            "--output",
            str(evidence),
            "--preregistration-commit",
            PREREGISTRATION_COMMIT,
            "--resamples",
            "100",
            "--pricing-retrieved-at",
            "2026-07-29T00:00:00Z",
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = validate_determinism_economics_artifact(evidence)
    assert artifact.preregistration_commit == PREREGISTRATION_COMMIT


@pytest.mark.parametrize("prefix_tokens", [512, 513])
def test_cache_eligible_prefixes_write_then_read(
    monkeypatch: pytest.MonkeyPatch, prefix_tokens: int
) -> None:
    def token_count(text: str) -> int:
        return prefix_tokens if text.startswith("SYSTEM") else 11

    monkeypatch.setattr(economics, "count_tokens", token_count)
    session = SessionFixture(
        session_id="cache-session",
        repository="cache-repository",
        resolved=True,
        turns=[
            SessionTurn(question="first", contexts=["alpha", "beta"]),
            SessionTurn(question="second", contexts=["alpha", "beta"]),
        ],
    )
    second_session = SessionFixture(
        session_id="cache-session-2",
        repository="cache-repository-2",
        resolved=True,
        turns=[
            SessionTurn(question="first", contexts=["alpha", "beta"]),
            SessionTurn(question="second", contexts=["alpha", "beta"]),
        ],
    )
    artifact = measure_economics(
        [session, second_session],
        preregistration_commit=PREREGISTRATION_COMMIT,
        source_revision=SOURCE_REVISION,
        generated_at="2026-07-29T00:00:00Z",
        session_fixture=str(FIXTURE),
        measurement_command="test command",
        resamples=20,
        pricing=PricingSchedule(retrieved_at="test"),
    )
    deterministic = next(arm for arm in artifact.arms if arm.arm is OrderingArm.DETERMINISTIC)
    ledger = deterministic.sessions[0]

    assert ledger.cache_write_tokens == prefix_tokens
    assert ledger.cache_read_tokens == prefix_tokens
    assert ledger.cacheable_tokens == prefix_tokens * 2
    assert (
        ledger.input_cost_usd == (prefix_tokens * 1.25 + prefix_tokens * 0.1 + 22) * 5.0 / 1_000_000
    )


def _ledger(session_id: str, repository: str, cost: float) -> SessionLedger:
    prefix = f"prefix-{session_id}"
    return SessionLedger(
        session_id=session_id,
        repository=repository,
        resolved=True,
        cacheable_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        uncached_input_tokens=0,
        rendered_prefixes=[prefix],
        prefix_sha256=[hashlib.sha256(prefix.encode()).hexdigest()],
        input_cost_usd=cost,
    )


def test_cluster_bootstrap_retains_repeated_repository_draws() -> None:
    control_a = _ledger("a", "repository-a", 1.0)
    control_b = _ledger("b", "repository-b", 1.0)
    comparator_a = _ledger("a", "repository-a", 2.0)
    comparator_b = _ledger("b", "repository-b", 4.0)

    reduction = economics.relative_cost_reduction(
        [control_a, control_a, control_b],
        [comparator_a, comparator_a, comparator_b],
    )

    assert reduction == 62.5


def test_validator_rejects_unknown_preregistration_commit(tmp_path: Path) -> None:
    payload = _artifact().model_dump(mode="json")
    payload["preregistration_commit"] = UNKNOWN_COMMIT
    evidence = tmp_path / "invalid-preregistration.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pre-registration commit"):
        validate_determinism_economics_artifact(evidence)


def test_validator_wraps_replay_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "s7.json"
    evidence.write_text(_artifact().model_dump_json(), encoding="utf-8")

    def invalid_measurement(*_args: object, **_kwargs: object) -> DeterminismEconomicsArtifact:
        SessionFixture.model_validate({})
        raise AssertionError("unreachable")

    monkeypatch.setattr(economics, "measure_economics", invalid_measurement)

    with pytest.raises(ValueError, match="failed validation"):
        validate_determinism_economics_artifact(evidence)


def test_cli_rejects_non_object_evidence_input(tmp_path: Path) -> None:
    evidence = tmp_path / "not-an-object.json"
    evidence.write_text("[1, 2, 3]", encoding="utf-8")

    result = CliRunner().invoke(
        benchmark_cmd,
        ["validate", "--kind", "evidence", "--input", str(evidence)],
    )

    assert result.exit_code != 0
    assert "AttributeError" not in result.output
