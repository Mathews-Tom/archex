"""Guards over the checked-in Gate A artifacts.

R3's verification criterion is that both artifacts validate. Nothing executed
those commands, so a later edit could have quietly relaxed a verdict, re-pointed
the pre-registration, or widened a band without any check noticing. These tests
run the validator over the committed files on every suite run.

Gate A's verdict cancels R7 through R16. It may not silently become a pass.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from archex.benchmark.replication import (
    ReplicationVerdict,
    validate_replication_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "benchmarks" / "evidence"
PREREGISTRATION_COMMIT = "557c5683e5a2622e0a96370a379365c8498d1dc4"
BAND = (2.88, 6.88)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("s0-rlcoder-replication.json", ReplicationVerdict.FAIL),
        ("s0-cast-replication.json", ReplicationVerdict.UNRUNNABLE),
    ],
)
def test_checked_in_artifact_validates_and_keeps_its_verdict(
    name: str, expected: ReplicationVerdict
) -> None:
    artifact = validate_replication_artifact(EVIDENCE_DIR / name)
    assert [arm.verdict for arm in artifact.arms] == [expected]
    assert artifact.passing_arms == []
    assert artifact.preregistration_commit == PREREGISTRATION_COMMIT
    assert artifact.preregistration == ".docs/spikes/S0-replication-gate.md"


def test_every_gate_a_arm_is_replication_class() -> None:
    """An adaptation-class arm could never have answered the Gate A question."""
    for name in ("s0-rlcoder-replication.json", "s0-cast-replication.json"):
        artifact = validate_replication_artifact(EVIDENCE_DIR / name)
        for arm in artifact.arms:
            assert arm.evidence_class is not None
            assert arm.evidence_class.value == "replication"


def test_the_rlcoder_band_is_the_pre_registered_one() -> None:
    """Widening the band is the one edit that would turn this fail into a pass."""
    artifact = validate_replication_artifact(EVIDENCE_DIR / "s0-rlcoder-replication.json")
    band = artifact.arms[0].equivalence_band
    assert band is not None
    assert (band.low, band.high) == BAND


def test_gate_a_records_a_fail() -> None:
    """R7 through R16 are cancelled on this line."""
    verdict = (REPO_ROOT / "GATE-A.md").read_text(encoding="utf-8")
    assert "GATE A FAIL" in verdict
    assert "GATE A PASS" not in verdict


def test_the_analyzer_uses_the_pre_registered_constants() -> None:
    """`benchmarks/` is outside the type gate, so pin its frozen parameters here."""
    source = (REPO_ROOT / "benchmarks" / "replication" / "rlcoder" / "analyze.py").read_text(
        encoding="utf-8"
    )
    constants: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    assert constants["BAND_LOW"] == BAND[0]
    assert constants["BAND_HIGH"] == BAND[1]
    assert constants["REPORTED_DELTA"] == 4.88
    assert constants["SEED"] == 20260727
    assert constants["RESAMPLES"] == 10000
    assert constants["EXPECTED_TASKS"] == 1600
    assert constants["EXPECTED_CLUSTERS"] == 8


def test_the_rlcoder_arm_pins_everything_a_rerun_needs() -> None:
    payload = json.loads((EVIDENCE_DIR / "s0-rlcoder-replication.json").read_text(encoding="utf-8"))
    pins = payload["arms"][0]["pins"]
    assert pins["harness_commit"] == "164d8d88cde324a38f5da70c4f858cc4679ef08e"
    assert pins["dataset_split"] == "repoeval/line_level"
    assert set(pins["models"]) == {"generator", "retriever_control", "retriever_treatment"}
    assert "torch" in pins["environment"]
    assert "<" not in pins["command"], "the recorded command is still a placeholder template"
