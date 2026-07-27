"""Tests for external-replication evidence artifacts and their CLI validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from archex.benchmark.replication import (
    ReplicationEvidenceError,
    ReplicationVerdict,
    validate_replication_artifact,
)
from archex.cli.benchmark_cmd import benchmark_cmd

PINS: dict[str, Any] = {
    "harness_repo": "https://github.com/DeepSoftwareAnalytics/RLCoder",
    "harness_commit": "164d8d88cde324a38f5da70c4f858cc4679ef08e",
    "dataset": "nov3630/Data4RLCoder",
    "dataset_revision": "0" * 40,
    "dataset_split": "repoeval/line_level",
    "models": {"generator": "deepseek-ai/deepseek-coder-1.3b-base@abc123"},
    "environment": {"torch": "2.13.0", "transformers": "4.44.2"},
    "command": "python run_replication.py --arm rlcoder",
}


def _arm(**overrides: Any) -> dict[str, Any]:
    arm: dict[str, Any] = {
        "arm_id": "rlcoder-repoeval-line",
        "evidence_class": "replication",
        "paper": "arXiv:2407.19487",
        "paper_cell": "Table II, RepoEval (Line), DeepSeekCoder-1B",
        "metric": "exact_match_delta_points",
        "reported_delta": 4.88,
        "equivalence_band": {"low": 2.88, "high": 6.88, "method": "pre-registered"},
        "reproduced_delta": 4.5,
        "reproduced_interval": {"low": 1.2, "high": 7.4, "method": "cluster bootstrap"},
        "verdict": "pass",
        "rationale": "Reproduced inside the pre-registered band.",
        "pins": dict(PINS),
    }
    arm.update(overrides)
    return arm


def _artifact(*arms: dict[str, Any]) -> dict[str, Any]:
    return {
        "replication_version": 1,
        "spike_id": "S0",
        "preregistration": ".docs/spikes/S0-replication-gate.md",
        "preregistration_commit": "1" * 40,
        "generated_at": "2026-07-27T00:00:00Z",
        "hardware": "Darwin arm64",
        "arms": list(arms) or [_arm()],
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_artifact_round_trips(tmp_path: Path) -> None:
    artifact = validate_replication_artifact(_write(tmp_path, _artifact()))
    assert [arm.arm_id for arm in artifact.passing_arms] == ["rlcoder-repoeval-line"]


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplicationEvidenceError, match="not a file"):
        validate_replication_artifact(tmp_path / "absent.json")


def test_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplicationEvidenceError, match="not a file"):
        validate_replication_artifact(tmp_path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReplicationEvidenceError, match="not readable JSON"):
        validate_replication_artifact(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    payload = _artifact()
    payload["conclusion"] = "looks good"
    with pytest.raises(ReplicationEvidenceError, match="failed validation"):
        validate_replication_artifact(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "dropped",
    ["reported_delta", "equivalence_band", "reproduced_delta", "reproduced_interval", "pins"],
)
def test_scored_arm_must_carry_every_field(tmp_path: Path, dropped: str) -> None:
    payload = _artifact(_arm(**{dropped: None}))
    with pytest.raises(ReplicationEvidenceError, match=dropped):
        validate_replication_artifact(_write(tmp_path, payload))


@pytest.mark.parametrize("pin", sorted(PINS))
def test_every_pin_is_required(tmp_path: Path, pin: str) -> None:
    pins = dict(PINS)
    del pins[pin]
    payload = _artifact(_arm(pins=pins))
    with pytest.raises(ReplicationEvidenceError, match="failed validation"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_short_harness_commit_is_rejected(tmp_path: Path) -> None:
    payload = _artifact(_arm(pins={**PINS, "harness_commit": "164d8d8"}))
    with pytest.raises(ReplicationEvidenceError, match="failed validation"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_unpinned_model_revision_is_rejected(tmp_path: Path) -> None:
    payload = _artifact(_arm(pins={**PINS, "models": {"generator": "  "}}))
    with pytest.raises(ReplicationEvidenceError, match="must pin a version"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_unpinned_environment_version_is_rejected(tmp_path: Path) -> None:
    """Greedy decoding is not reproducible if the library versions are unrecorded."""
    payload = _artifact(_arm(pins={**PINS, "environment": {"torch": ""}}))
    with pytest.raises(ReplicationEvidenceError, match="must pin a version"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_unknown_evidence_class_is_rejected(tmp_path: Path) -> None:
    payload = _artifact(_arm(evidence_class="vibes"))
    with pytest.raises(ReplicationEvidenceError, match="failed validation"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_inverted_interval_is_rejected(tmp_path: Path) -> None:
    payload = _artifact(
        _arm(reproduced_interval={"low": 7.4, "high": 1.2, "method": "cluster bootstrap"})
    )
    with pytest.raises(ReplicationEvidenceError, match="exceeds high"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_duplicate_arm_ids_are_rejected(tmp_path: Path) -> None:
    payload = _artifact(_arm(), _arm())
    with pytest.raises(ReplicationEvidenceError, match="duplicate arm_id"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_miss_cannot_be_filed_as_a_pass(tmp_path: Path) -> None:
    """A delta below the band is a fail no matter what the verdict field claims."""
    payload = _artifact(
        _arm(
            reproduced_delta=0.4,
            reproduced_interval={"low": -1.0, "high": 1.8, "method": "cluster bootstrap"},
        )
    )
    with pytest.raises(ReplicationEvidenceError, match="implies 'fail'"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_overshoot_is_a_fail(tmp_path: Path) -> None:
    payload = _artifact(
        _arm(
            reproduced_delta=9.9,
            reproduced_interval={"low": 7.0, "high": 12.0, "method": "cluster bootstrap"},
            verdict="fail",
        )
    )
    artifact = validate_replication_artifact(_write(tmp_path, payload))
    assert artifact.arms[0].verdict is ReplicationVerdict.FAIL
    assert artifact.passing_arms == []


def test_interval_spanning_zero_inside_the_band_is_inconclusive(tmp_path: Path) -> None:
    payload = _artifact(
        _arm(
            reproduced_interval={"low": -0.5, "high": 9.0, "method": "cluster bootstrap"},
            verdict="inconclusive",
        )
    )
    artifact = validate_replication_artifact(_write(tmp_path, payload))
    assert artifact.passing_arms == []


def test_inconclusive_cannot_be_filed_as_a_pass(tmp_path: Path) -> None:
    payload = _artifact(
        _arm(reproduced_interval={"low": -0.5, "high": 9.0, "method": "cluster bootstrap"})
    )
    with pytest.raises(ReplicationEvidenceError, match="implies 'inconclusive'"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_unrunnable_arm_needs_no_numbers(tmp_path: Path) -> None:
    payload = _artifact(
        {
            "arm_id": "cast-repoeval-recall",
            "evidence_class": "replication",
            "paper": "arXiv:2506.15655",
            "paper_cell": "Table 5, RepoEval, GIST-base, Recall@5",
            "metric": "recall_at_5_delta_points",
            "verdict": "unrunnable",
            "rationale": "The released artifact is the chunker only.",
        }
    )
    artifact = validate_replication_artifact(_write(tmp_path, payload))
    assert artifact.arms[0].verdict is ReplicationVerdict.UNRUNNABLE
    assert artifact.passing_arms == []


def test_unrunnable_arm_may_not_smuggle_in_a_result(tmp_path: Path) -> None:
    payload = _artifact(_arm(verdict="unrunnable"))
    with pytest.raises(ReplicationEvidenceError, match="unrunnable but reports"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_cli_accepts_a_valid_artifact(tmp_path: Path) -> None:
    path = _write(tmp_path, _artifact())
    result = CliRunner().invoke(
        benchmark_cmd, ["validate", "--kind", "replication", "--input", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert "rlcoder-repoeval-line=pass" in result.output


def test_cli_rejects_a_defective_artifact(tmp_path: Path) -> None:
    payload = _artifact(_arm(pins={**PINS, "harness_commit": "164d8d8"}))
    path = _write(tmp_path, payload)
    result = CliRunner().invoke(
        benchmark_cmd, ["validate", "--kind", "replication", "--input", str(path)]
    )
    assert result.exit_code != 0
    assert "failed validation" in result.output


def test_cli_requires_input(tmp_path: Path) -> None:
    result = CliRunner().invoke(benchmark_cmd, ["validate", "--kind", "replication"])
    assert result.exit_code != 0
    assert "--input is required" in result.output


@pytest.mark.parametrize(
    ("delta", "ci_low", "ci_high", "verdict"),
    [
        (2.88, 1.0, 4.0, "pass"),
        (6.88, 5.0, 8.0, "pass"),
        (2.8799, 1.0, 4.0, "fail"),
        (6.8801, 5.0, 8.0, "fail"),
        (4.0, 0.0, 6.0, "inconclusive"),
        (4.0, -0.0, 6.0, "inconclusive"),
    ],
)
def test_band_boundaries_survive_validation(
    tmp_path: Path, delta: float, ci_low: float, ci_high: float, verdict: str
) -> None:
    """The gate turned on 0.0675 points; the schema's boundaries are pinned too."""
    payload = _artifact(
        _arm(
            reproduced_delta=delta,
            reproduced_interval={"low": ci_low, "high": ci_high, "method": "cluster bootstrap"},
            verdict=verdict,
        )
    )
    artifact = validate_replication_artifact(_write(tmp_path, payload))
    assert artifact.arms[0].verdict.value == verdict


def test_a_band_that_does_not_contain_its_reported_delta_is_rejected(tmp_path: Path) -> None:
    """Tampering with the band after the fact is the failure mode this gate exists for."""
    payload = _artifact(
        _arm(
            equivalence_band={"low": -9.0, "high": -1.0, "method": "tampered"},
            reproduced_delta=-5.0,
            reproduced_interval={"low": -8.0, "high": -2.0, "method": "cluster bootstrap"},
        )
    )
    with pytest.raises(ReplicationEvidenceError, match="does not contain the reported delta"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_a_sign_flipped_reproduction_cannot_be_recorded_as_a_pass(tmp_path: Path) -> None:
    """excludes_zero is sign-agnostic, so a band straddling zero is refused outright."""
    payload = _artifact(
        _arm(
            equivalence_band={"low": -2.0, "high": 12.0, "method": "straddles zero"},
            reproduced_delta=-1.5,
            reproduced_interval={"low": -3.0, "high": -0.5, "method": "cluster bootstrap"},
        )
    )
    with pytest.raises(ReplicationEvidenceError, match="straddling zero"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_a_delta_outside_its_own_interval_is_rejected(tmp_path: Path) -> None:
    payload = _artifact(
        _arm(
            reproduced_delta=4.5,
            reproduced_interval={"low": -9.0, "high": -1.0, "method": "cluster bootstrap"},
            verdict="fail",
        )
    )
    with pytest.raises(ReplicationEvidenceError, match="outside its own interval"):
        validate_replication_artifact(_write(tmp_path, payload))


def test_a_zero_width_band_is_rejected(tmp_path: Path) -> None:
    payload = _artifact(
        _arm(
            equivalence_band={"low": 0.0, "high": 0.0, "method": "degenerate"},
            reproduced_delta=0.0,
            reproduced_interval={"low": -1.0, "high": 1.0, "method": "cluster bootstrap"},
            verdict="inconclusive",
        )
    )
    with pytest.raises(ReplicationEvidenceError, match="does not contain the reported delta"):
        validate_replication_artifact(_write(tmp_path, payload))
