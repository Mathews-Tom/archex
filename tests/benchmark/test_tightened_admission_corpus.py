"""Validation for M0.4 round 3's tightened context-candidate frontier evidence.

This reader validates the checked-in round 3 corpus-proof artifact and its
identity against the two prior NO-GO rounds. It never runs the full corpus or
reads expected regions into runtime retrieval logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
ROUND1_ARTIFACT = ROOT / "benchmarks" / "evidence" / "m0.4-context-candidate-run1.json"
ROUND2_ARTIFACT = ROOT / "benchmarks" / "evidence" / "m0.4-task-contract-calibration.json"
ROUND3_ARTIFACT = ROOT / "benchmarks" / "evidence" / "m0.4-tightened-admission-run3.json"
FROZEN_TASK_MANIFEST_DIGEST = "c9b6eb53901a572372131cb0d748af5ba487686ee6604f9baa40bcea09ae5721"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_round1_and_round2_evidence_are_unmodified() -> None:
    # Constraint: round 3 must never overwrite or delete the first two
    # attempts' evidence. Pin their content identity so a future edit to
    # either file is caught here rather than silently destroying the record.
    round1 = _load(ROUND1_ARTIFACT)
    round2 = _load(ROUND2_ARTIFACT)
    assert round1["verdict"] == "NO-GO"
    assert round1["aggregate"]["wall_p95_ms"] == 3812.877666001441
    assert round2["calibrated_candidate_verdict"]["verdict"] == "NO-GO"
    assert round2["calibrated_candidate_verdict"]["total_violations"] == 122


def test_round3_evidence_has_immutable_corpus_identity() -> None:
    payload = _load(ROUND3_ARTIFACT)

    assert payload["schema_version"] == 1
    assert payload["milestone"] == "M0.4"
    assert payload["round"] == 3
    assert payload["task_manifest_digest"] == FROZEN_TASK_MANIFEST_DIGEST
    assert len(payload["source_revision"]) == 40
    assert len(payload["runs"]) == 2
    for run in payload["runs"]:
        assert run["task_count"] == 64
        assert len(run["manifest_sha256"]) == 64


def test_round3_evidence_shows_two_run_agreement_on_the_same_revision() -> None:
    payload = _load(ROUND3_ARTIFACT)

    assert payload["two_run_agreement"]["same_source_revision"] is True
    assert payload["two_run_agreement"]["per_task_quality_metrics_identical"] is True
    # The two runs must be genuinely distinct evidence, not the same directory
    # copied twice, or "agreement" would be a tautology.
    assert payload["runs"][0]["manifest_sha256"] != payload["runs"][1]["manifest_sha256"]


def test_round3_evidence_resolves_every_protected_region_regression() -> None:
    payload = _load(ROUND3_ARTIFACT)
    regressions = payload["protected_evidence_regressions_vs_archex_query"]

    assert regressions["required_file_recall"] == []
    assert regressions["region_recall"] == []
    assert regressions["line_recall"] == []


def test_round3_evidence_records_a_measured_improvement_over_round2() -> None:
    payload = _load(ROUND3_ARTIFACT)
    round2 = _load(ROUND2_ARTIFACT)
    comparison = payload["comparison_to_prior_rounds"]

    round2_total = comparison["round2_task_contract_calibration"]["total_violations"]
    round3_total = comparison["round3_tightened_admission"]["total_violations"]
    assert round2_total == round2["calibrated_candidate_verdict"]["total_violations"]
    assert round3_total < round2_total

    round2_region = comparison["round2_task_contract_calibration"]["region_recall_regressions"]
    round2_line = comparison["round2_task_contract_calibration"]["line_recall_regressions"]
    assert round2_region > 0
    assert round2_line > 0
    assert comparison["round3_tightened_admission"]["region_recall_regressions"] == 0
    assert comparison["round3_tightened_admission"]["line_recall_regressions"] == 0

    round2_precision = comparison["round2_task_contract_calibration"]["precision_failures"]
    round2_f1 = comparison["round2_task_contract_calibration"]["f1_failures"]
    assert comparison["round3_tightened_admission"]["precision_failures"] < round2_precision
    assert comparison["round3_tightened_admission"]["f1_failures"] < round2_f1


def test_round3_evidence_blocks_promotion_on_residual_absolute_failures() -> None:
    payload = _load(ROUND3_ARTIFACT)

    assert payload["verdict"] == "NO-GO"
    promotion = payload["promotion"]
    assert promotion["default_changed"] is False
    assert promotion["canonical_baseline_changed"] is False
    assert promotion["m1_unblocked"] is False
    assert "41 absolute-row violations" in promotion["reason"]
    assert "16 MRR" in promotion["reason"]
    assert "partial pass is NO-GO" in promotion["reason"]


def test_round3_evidence_shows_every_residual_failure_matches_the_control() -> None:
    # The residual-failure analysis is the load-bearing claim for this round's
    # NO-GO: every remaining absolute violation is bounded by out-of-scope
    # base retrieval/reranking, not by anything this round's file-set/
    # region-elision changes could have fixed.
    payload = _load(ROUND3_ARTIFACT)
    analysis = payload["comparison_to_prior_rounds"]["residual_failure_analysis"]

    assert analysis["mrr_failures_all_match_control_exactly"] is True
    assert analysis["precision_f1_failures_all_match_control_exactly"] is True
    assert analysis["token_efficiency_failures_unchanged_from_round1"] is True
    assert analysis["recall_failure_matches_control"] is True


def test_round3_evidence_is_evaluation_only() -> None:
    payload = _load(ROUND3_ARTIFACT)
    assert "evaluation-only" in payload["runtime_boundary"]
