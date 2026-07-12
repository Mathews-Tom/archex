"""Validation for the first immutable M0.4 context-candidate frontier run.

This reader validates the checked-in NO-GO evidence summary. It never runs the
full corpus or reads expected regions into runtime retrieval logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from archex.benchmark.evidence import task_manifest_digest
from archex.benchmark.loader import load_tasks
from archex.benchmark.models import Strategy
from archex.serve.context import generic_query_terms

ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "benchmarks" / "evidence" / "m0.4-context-candidate-run1.json"
CALIBRATION_ARTIFACT = ROOT / "benchmarks" / "evidence" / "m0.4-task-contract-calibration.json"
FROZEN_TASK_MANIFEST_DIGEST = "0ae42fb18f96678b5e79591be8372d83fbeed646c4c08e7cc0f118eba4c2bd09"


def _load_artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_task_contract_calibration_binds_the_reviewed_current_corpus() -> None:
    payload = json.loads(CALIBRATION_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["milestone"] == "M0.4"
    assert payload["decision"] == "evaluation-only task contract calibration"
    assert payload["prior_task_manifest_digest"] == FROZEN_TASK_MANIFEST_DIGEST
    assert payload["task_manifest_digest"] == task_manifest_digest(ROOT / "benchmarks" / "tasks")
    assert {change["task_id"] for change in payload["changes"]} == {
        "archex_adapter_registry",
        "archex_project_status",
        "django_middleware",
        "loc_django_username_validator",
    }
    assert "evaluation-only" in payload["runtime_boundary"]
    assert payload["calibrated_candidate_run"] == {
        "source_revision": "56bf07a97aea07a5ee6980e5cf6a8416475f748d",
        "manifest_sha256": "39c886239d20116449fab68745a8ce5e06d9b4585f6619bdc89cdcbc69161877",
        "strategy": Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value,
        "task_count": 64,
        "warm_samples": 64,
        "warm_p95_ms": 727.041,
    }
    assert payload["calibrated_candidate_verdict"]["verdict"] == "NO-GO"
    assert payload["calibrated_candidate_verdict"]["total_violations"] == 122


def test_calibrated_tasks_name_their_distinguishing_runtime_evidence() -> None:
    tasks = {task.task_id: task for task in load_tasks(ROOT / "benchmarks" / "tasks")}
    required_terms = {
        "archex_adapter_registry": {"pythonadapter"},
        "archex_project_status": {"status", "command", "signature"},
        "django_middleware": {"basehandler", "wsgi", "commonmiddleware"},
        "loc_django_username_validator": {"validator"},
    }

    for task_id, expected_terms in required_terms.items():
        assert expected_terms <= set(generic_query_terms(tasks[task_id].question))


def test_context_candidate_evidence_has_immutable_corpus_identity() -> None:
    payload = _load_artifact()

    assert payload["schema_version"] == 1
    assert payload["milestone"] == "M0.4"
    assert payload["task_manifest_digest"] == FROZEN_TASK_MANIFEST_DIGEST
    assert payload["candidate_run"]["strategy"] == Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value
    assert payload["candidate_run"]["task_count"] == 64
    assert len(payload["candidate_run"]["manifest_sha256"]) == 64


def test_context_candidate_evidence_blocks_promotion_on_observed_failures() -> None:
    payload = _load_artifact()

    assert payload["verdict"] == "NO-GO"
    assert payload["absolute_gate_failures"]["precision"]
    assert payload["absolute_gate_failures"]["f1"]
    assert payload["absolute_gate_failures"]["mrr"]
    assert payload["absolute_gate_failures"]["token_efficiency"] == [
        "loc_fastapi_solve_dependencies",
        "loc_tokio_current_thread_block_on",
    ]
    assert payload["context_economy_failures"]["wall_p95_exceeded"] is True
    assert payload["recall_regressions_vs_archex_query"]["required_file_recall"] == []
    assert payload["recall_regressions_vs_archex_query"]["region_recall"]
    assert payload["recall_regressions_vs_archex_query"]["line_recall"]
    assert payload["promotion"] == {
        "default_changed": False,
        "canonical_baseline_changed": False,
        "m1_unblocked": False,
        "reason": (
            "Candidate failed absolute, region, line, context-economy, and wall-latency gates "
            "on the first immutable full-corpus run; a second promotion run is forbidden."
        ),
    }
