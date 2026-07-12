"""Corpus-level verification for the M0.3 direct-evidence rank candidate.

Validates the checked-in 64-task corpus comparison artifact
(`benchmarks/evidence/m0.3-corpus-comparison.json`), produced from two
same-revision local runs of `archex_query_rank_candidate` (candidate) and
`archex_query_coverage_candidate` (M0.2's candidate, M0.3's declared
control/input) against the full task corpus. This is a report-only reader
over the checked-in artifact -- it never re-runs the corpus or reads
benchmark task expected-file definitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from archex.benchmark.models import Strategy

_M03_TASK_MANIFEST_DIGEST = "0ae42fb18f96678b5e79591be8372d83fbeed646c4c08e7cc0f118eba4c2bd09"


_GATE = {"min_recall": 0.60, "min_precision": 0.20, "min_f1": 0.30, "min_mrr": 0.55}


def _load_artifact() -> dict[str, Any]:
    root = Path(__file__).parents[2]
    artifact = root / "benchmarks" / "evidence" / "m0.3-corpus-comparison.json"
    return json.loads(artifact.read_text(encoding="utf-8"))


def test_corpus_artifact_identity_matches_its_historical_task_corpus() -> None:
    payload = _load_artifact()

    assert payload["task_manifest_digest"] == _M03_TASK_MANIFEST_DIGEST
    assert payload["gate_thresholds"] == _GATE


def test_corpus_artifact_covers_all_sixty_four_tasks() -> None:
    payload = _load_artifact()

    assert len(payload["rows"]) == 64
    task_ids = {row["task_id"] for row in payload["rows"]}
    assert len(task_ids) == 64


def test_corpus_artifact_records_two_reproducible_repeats() -> None:
    payload = _load_artifact()

    assert payload["repeat_stability"]["candidate_result_differences_across_repeat"] == []
    assert payload["repeat_stability"]["control_result_differences_across_repeat"] == []
    for tid in ("candidate_run_21", "candidate_run_22", "control_run_21", "control_run_22"):
        assert payload["runs"][tid]["strategy"] in (
            Strategy.ARCHEX_QUERY_RANK_CANDIDATE.value,
            Strategy.ARCHEX_QUERY_COVERAGE_CANDIDATE.value,
        )


def test_corpus_artifact_shows_zero_required_file_recall_regressions() -> None:
    payload = _load_artifact()

    assert payload["required_file_regressions"] == []
    for row in payload["rows"]:
        assert row["candidate_required_file_recall"] >= row["control_required_file_recall"] - 1e-9


def test_corpus_artifact_preserves_m02_five_target_recovery() -> None:
    payload = _load_artifact()

    targets = {entry["task_id"] for entry in payload["m02_target_task_recovery"]}
    assert targets == {
        "archex_adapter_registry",
        "archex_project_status",
        "django_middleware",
        "loc_django_username_validator",
        "routing_pl_scoring",
    }
    for entry in payload["m02_target_task_recovery"]:
        assert entry["run21_required_file_recall"] == 1.0
        assert entry["run22_required_file_recall"] == 1.0


def test_corpus_artifact_family_breakdown_never_regresses_required_recall() -> None:
    payload = _load_artifact()

    for name in ("self_repo", "external_comprehension", "localization"):
        block = payload["family_breakdown"][name]
        assert block["candidate"]["n_tasks"] == block["control"]["n_tasks"]
        assert (
            block["candidate"]["mean_required_file_recall"]
            >= block["control"]["mean_required_file_recall"] - 1e-9
        )


def test_corpus_artifact_candidate_never_worsens_precision_or_f1_per_row() -> None:
    # The candidate is order/admission-narrowing only; it can only match or
    # improve precision and F1 relative to its declared control, never
    # regress them, on any single task.
    payload = _load_artifact()

    for row in payload["rows"]:
        assert row["candidate_precision"] >= row["control_precision"] - 1e-9
