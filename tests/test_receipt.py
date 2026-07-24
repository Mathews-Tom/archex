from __future__ import annotations

from archex.integrations.semantic.models import (
    ProviderAvailability,
    SemanticProviderName,
    SemanticProviderReceipt,
)
from archex.models import (
    CompressionLossRisk,
    CompressionMetadata,
    CompressionMode,
    ContextBundle,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextReceipt,
    ContextReceiptEdge,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
    ContextSkippedCandidate,
    ContextSkippedReason,
    Edge,
    EdgeConfidence,
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
        token_budget=ContextReceiptTokenBudget(requested=500, consumed=250),
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        context_complete=ContextCompletenessStatus.COMPLETE,
        context_complete_reason=ContextCompletenessReason.COMPLETE,
        recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
        skipped_candidates=[
            ContextSkippedCandidate(
                file_path="src/service.py",
                reason=ContextSkippedReason.BELOW_THRESHOLD,
                handle=file_handle("src/service.py"),
            )
        ],
        skipped_total=1,
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
        budget=ScoutBudget(token_budget=100, token_count=64),
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

    receipt = build_scout_receipt(
        scout,
        direct_receipt,
        file_hashes={"src/service.py": "sha256-service"},
    )

    assert receipt is not None
    assert receipt.skipped_candidates == []
    assert receipt.token_budget.requested == 100
    assert receipt.token_budget.consumed == 64
    assert receipt.returned_context[0].content_hash == "sha256-service"
    assert receipt.returned_total == 1
    assert receipt.skipped_total == 0
    assert receipt.context_complete == ContextCompletenessStatus.COMPLETE
    assert receipt.context_complete_reason == ContextCompletenessReason.COMPLETE
    assert receipt.recommended_next_action == ContextRecommendedAction.USE_BUNDLE


def _compressed_item(handle: str) -> ContextReceiptItem:
    return ContextReceiptItem(
        handle=handle,
        file_path="src/widget.py",
        start_line=1,
        end_line=40,
        content_hash="orig-hash",
        compression=CompressionMetadata(
            compression_mode=CompressionMode.STRUCTURAL_CODE_ELISION,
            original_tokens=200,
            compressed_tokens=70,
            compression_ratio=0.35,
            original_content_hash="orig-hash",
            compressed_content_hash="elided-hash",
            fetch_original_handle=handle,
            compression_loss_risk=CompressionLossRisk.LOW,
        ),
    )


def test_compression_metadata_does_not_upgrade_receipt_completeness() -> None:
    receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=500, consumed=250),
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        returned_context=[_compressed_item("chunk:widget")],
        returned_total=1,
        context_complete=ContextCompletenessStatus.INCOMPLETE,
        context_complete_reason=ContextCompletenessReason.BUDGET_EXHAUSTED,
        recommended_next_action=ContextRecommendedAction.RAISE_BUDGET,
    )

    # Compression metadata is orthogonal to completeness: an incomplete receipt
    # stays incomplete and keeps its reason/action even when rows are compressed.
    assert receipt.context_complete == ContextCompletenessStatus.INCOMPLETE
    assert receipt.context_complete_reason == ContextCompletenessReason.BUDGET_EXHAUSTED
    assert receipt.recommended_next_action == ContextRecommendedAction.RAISE_BUDGET
    assert receipt.returned_context[0].compression is not None
    assert receipt.returned_context[0].compression.is_compressed is True


def test_receipt_round_trips_with_compressed_rows() -> None:
    receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=500, consumed=250),
        index_revision="rev",
        returned_context=[_compressed_item("chunk:widget")],
        returned_total=1,
        context_complete=ContextCompletenessStatus.COMPLETE,
        context_complete_reason=ContextCompletenessReason.COMPLETE,
        recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
    )
    restored = ContextReceipt.model_validate_json(receipt.model_dump_json())
    assert restored == receipt
    # A compressed row never flips a complete receipt; completeness is preserved.
    assert restored.context_complete == ContextCompletenessStatus.COMPLETE


def test_build_context_receipt_carries_semantic_providers() -> None:
    bundle = ContextBundle(query="q", token_count=10, token_budget=100)
    receipts = [
        SemanticProviderReceipt(
            provider=SemanticProviderName.SCIP,
            availability=ProviderAvailability.AVAILABLE,
            evidence_count=3,
        )
    ]

    receipt = build_context_receipt(
        bundle,
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        semantic_providers=receipts,
    )

    assert receipt.semantic_providers == receipts


def test_build_context_receipt_defaults_semantic_providers_empty() -> None:
    bundle = ContextBundle(query="q", token_count=10, token_budget=100)
    receipt = build_context_receipt(bundle, index_revision="rev", freshness=ContextFreshness.CLEAN)
    assert receipt.semantic_providers == []


def test_receipt_edge_from_edge_helper_propagates_provider() -> None:
    from archex.receipt import _receipt_edge  # pyright: ignore[reportPrivateUsage]

    edge = Edge(
        source="a.py",
        target="b.py",
        kind=EdgeKind.SEMANTIC_DEFINITION,
        confidence=EdgeConfidence.EXTRACTED,
        confidence_score=0.9,
        provider="scip",
        provider_version="0.5.0",
    )

    receipt_edge = _receipt_edge(edge)

    assert receipt_edge.provider == "scip"
    assert receipt_edge.provider_version == "0.5.0"


def test_receipt_edge_from_edge_helper_syntax_edge_has_no_provider() -> None:
    from archex.receipt import _receipt_edge  # pyright: ignore[reportPrivateUsage]

    edge = Edge(source="a.py", target="b.py", kind=EdgeKind.IMPORTS)
    receipt_edge = _receipt_edge(edge)

    assert receipt_edge.provider is None
    assert receipt_edge.provider_version is None
