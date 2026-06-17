from __future__ import annotations

from archex.models import (
    ContextBundle,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextReceipt,
    ContextReceiptTokenBudget,
    ContextReceiptEdge,
    ContextRecommendedAction,
    ContextSkippedCandidate,
    ContextSkippedReason,
    EdgeKind,
)
from archex.receipt import (
    build_context_receipt,
    build_scout_receipt,
    stale_index_skipped_candidate,
)
from archex.scout import (
    ScoutBudget,
    ScoutFetchPlan,
    ScoutFile,
    ScoutResult,
    file_handle,
    symbol_handle,
)


def test_context_receipt_preserves_stale_marker_when_skipped_candidates_are_capped() -> None:
    bundle = ContextBundle(query="q", token_count=10, token_budget=100, truncated=True)
    skipped = [
        ContextSkippedCandidate(
            file_path=f"file_{index}.py",
            reason=ContextSkippedReason.BELOW_THRESHOLD,
        )
        for index in range(25)
    ]
    skipped.append(stale_index_skipped_candidate())

    receipt = build_context_receipt(
        bundle,
        index_revision="rev",
        freshness=ContextFreshness.UNKNOWN,
        skipped_candidates=skipped,
    )

    assert len(receipt.skipped_candidates) == 20
    assert receipt.skipped_candidates[0].reason == ContextSkippedReason.STALE_INDEX
    assert receipt.context_complete_reason == ContextCompletenessReason.STALE_INDEX
    assert receipt.recommended_next_action == ContextRecommendedAction.REFRESH_INDEX


def test_context_receipt_totals_preserve_uncapped_counts() -> None:
    bundle = ContextBundle(query="q", token_count=10, token_budget=100, truncated=True)
    skipped = [
        ContextSkippedCandidate(
            file_path=f"file_{index}.py",
            reason=ContextSkippedReason.BELOW_THRESHOLD,
        )
        for index in range(25)
    ]
    included_edges = [
        ContextReceiptEdge(source=f"src/{index}.py", target="src/shared.py", kind=EdgeKind.IMPORTS)
        for index in range(45)
    ]
    omitted_edges = [
        ContextReceiptEdge(source="src/root.py", target=f"src/{index}.py", kind=EdgeKind.IMPORTS)
        for index in range(23)
    ]

    receipt = build_context_receipt(
        bundle,
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        freshness_checked_at="2026-06-17T00:00:00Z",
        index_fresh_at="2026-06-17T00:00:00Z",
        included_edges=included_edges,
        omitted_edges=omitted_edges,
        skipped_candidates=skipped,
    )

    assert len(receipt.skipped_candidates) == 20
    assert len(receipt.included_edges) == 40
    assert len(receipt.omitted_edges) == 20
    assert receipt.returned_total == 0
    assert receipt.skipped_total == 25
    assert receipt.included_edges_total == 45
    assert receipt.omitted_edges_total == 23
    assert receipt.freshness_checked_at == "2026-06-17T00:00:00Z"
    assert receipt.index_fresh_at == "2026-06-17T00:00:00Z"


def test_scout_receipt_does_not_mark_selected_handle_files_as_skipped() -> None:
    direct_receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=100, consumed=20),
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        context_complete=ContextCompletenessStatus.COMPLETE,
        context_complete_reason=ContextCompletenessReason.COMPLETE,
        recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
    )
    scout = ScoutResult(
        query="q",
        ranked_files=[
            ScoutFile(
                path="src/service.py",
                language="python",
                lines=12,
                symbol_count=1,
                handle=file_handle("src/service.py"),
                score=1.0,
            )
        ],
        budget=ScoutBudget(token_budget=100),
        fetch_plan=ScoutFetchPlan(
            handles=[symbol_handle("src/service.py::Service#class")],
            file_reasons={
                "src/service.py": (
                    "selected_handle rank=1 score=1.000 coverage=1.000 "
                    "reason=ranked handle=symbol:src/service.py::Service#class"
                )
            },
        ),
    )

    receipt = build_scout_receipt(scout, direct_receipt)

    assert receipt is not None
    assert receipt.skipped_candidates == []
    assert receipt.context_complete == ContextCompletenessStatus.COMPLETE
    assert receipt.context_complete_reason == ContextCompletenessReason.COMPLETE
    assert receipt.recommended_next_action == ContextRecommendedAction.USE_BUNDLE
