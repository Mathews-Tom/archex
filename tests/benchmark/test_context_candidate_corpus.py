"""Validation for the first immutable M0.4 context-candidate frontier run.

This reader validates the checked-in NO-GO evidence summary. It never runs the
full corpus or reads expected regions into runtime retrieval logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from archex.benchmark.evidence import task_manifest_digest
from archex.benchmark.models import Strategy

ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "benchmarks" / "evidence" / "m0.4-context-candidate-run1.json"


def _load_artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_context_candidate_evidence_has_current_corpus_identity() -> None:
    payload = _load_artifact()

    assert payload["schema_version"] == 1
    assert payload["milestone"] == "M0.4"
    assert payload["task_manifest_digest"] == task_manifest_digest(ROOT / "benchmarks" / "tasks")
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
