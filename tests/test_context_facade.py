"""Contract tests for the primary agent-facing context() facade.

Covers ContextRequest/ContextFilters/ContextBudgets validation, deterministic
route resolution, post-retrieval filtering, ContextResult's computed views,
and an end-to-end context() call against the python_simple fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from archex.api import context
from archex.context_facade import (
    ContextBudgets,
    ContextFilters,
    ContextRequest,
    ContextResult,
    ContextRouteDecision,
    apply_context_filters,
    render_context_markdown,
    resolve_context_route,
)
from archex.models import (
    Config,
    ContextBundle,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextReceipt,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
    ContextSkippedReason,
    RankedChunk,
    RepoSource,
    RetrievalProfile,
    StructuralContext,
)
from archex.receipt import build_context_receipt
from archex.scout import chunk_handle
from archex.serve.intent import (
    DEFAULT_TOKEN_BUDGET,
    INTENT_TOKEN_BUDGETS,
    INTENT_WEIGHTS,
    QueryIntent,
    classify_intent,
)
from archex.serve.modality import BudgetTier, QueryModality, classify_modality

if TYPE_CHECKING:
    from pathlib import Path

from tests.serve.test_context import make_chunk

# ---------------------------------------------------------------------------
# ContextRequest / ContextFilters / ContextBudgets validation
# ---------------------------------------------------------------------------


class TestContextRequestValidation:
    def test_blank_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextRequest(query="   ")

    def test_blank_handle_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextRequest(query="how does auth work?", handles=["chunk:x", "  "])

    def test_minimal_request_defaults(self) -> None:
        request = ContextRequest(query="how does auth work?")
        assert request.intent is None
        assert request.profile is None
        assert request.filters.is_empty()
        assert request.budgets.token_budget is None
        assert request.handles == []


class TestContextBudgetsValidation:
    def test_non_positive_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextBudgets(token_budget=0)
        with pytest.raises(ValidationError):
            ContextBudgets(token_budget=-1)

    def test_positive_budget_accepted(self) -> None:
        assert ContextBudgets(token_budget=1024).token_budget == 1024


class TestContextFiltersValidation:
    def test_blank_filter_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextFilters(include_paths=["  "])
        with pytest.raises(ValidationError):
            ContextFilters(exclude_paths=[""])
        with pytest.raises(ValidationError):
            ContextFilters(languages=["  "])

    def test_is_empty(self) -> None:
        assert ContextFilters().is_empty()
        assert not ContextFilters(languages=["python"]).is_empty()

    def test_matches_language_filter(self) -> None:
        filters = ContextFilters(languages=["python"])
        py_chunk = make_chunk("c1", "a.py")
        assert filters.matches(py_chunk)
        js_chunk = py_chunk.model_copy(update={"language": "javascript"})
        assert not filters.matches(js_chunk)

    def test_matches_include_glob(self) -> None:
        filters = ContextFilters(include_paths=["src/auth/**"])
        assert filters.matches(make_chunk("c1", "src/auth/login.py"))
        assert not filters.matches(make_chunk("c2", "src/models/user.py"))

    def test_matches_exclude_glob(self) -> None:
        filters = ContextFilters(exclude_paths=["*/tests/*"])
        assert filters.matches(make_chunk("c1", "src/auth/login.py"))
        assert not filters.matches(make_chunk("c2", "src/tests/test_login.py"))

    def test_matches_combines_all_clauses(self) -> None:
        filters = ContextFilters(
            include_paths=["src/**"], exclude_paths=["src/tests/**"], languages=["python"]
        )
        assert filters.matches(make_chunk("c1", "src/auth/login.py"))
        assert not filters.matches(make_chunk("c2", "src/tests/test_login.py"))
        assert not filters.matches(make_chunk("c3", "other/login.py"))


# ---------------------------------------------------------------------------
# Route resolution
# ---------------------------------------------------------------------------


class TestResolveContextRoute:
    def test_auto_intent_matches_classifier(self) -> None:
        question = "how does authentication work?"
        resolution = resolve_context_route(ContextRequest(query=question))
        assert resolution.route.resolved_intent == classify_intent(question)
        assert resolution.route.intent_source == "auto"
        assert resolution.scoring_weights is None
        assert resolution.explicit_token_budget is False
        assert resolution.token_budget == DEFAULT_TOKEN_BUDGET

    def test_explicit_intent_pins_weights_and_budget(self) -> None:
        request = ContextRequest(query="anything", intent=QueryIntent.DEBUGGING)
        resolution = resolve_context_route(request)
        assert resolution.route.resolved_intent == QueryIntent.DEBUGGING
        assert resolution.route.intent_source == "explicit"
        assert resolution.scoring_weights == INTENT_WEIGHTS[QueryIntent.DEBUGGING]
        assert resolution.token_budget == INTENT_TOKEN_BUDGETS[QueryIntent.DEBUGGING]
        assert resolution.explicit_token_budget is True
        assert resolution.route.budget_source == "intent_default"

    def test_explicit_budget_wins_over_intent_default(self) -> None:
        request = ContextRequest(
            query="anything",
            intent=QueryIntent.DEBUGGING,
            budgets=ContextBudgets(token_budget=512),
        )
        resolution = resolve_context_route(request)
        assert resolution.token_budget == 512
        assert resolution.route.budget_source == "explicit"

    def test_resolved_modality_matches_classifier(self) -> None:
        question = "how does authentication work?"
        resolution = resolve_context_route(ContextRequest(query=question))
        assert resolution.route.resolved_modality == classify_modality(question)

    def test_handles_mode_flag(self) -> None:
        resolution = resolve_context_route(ContextRequest(query="x", handles=["chunk:a.py:1"]))
        assert resolution.route.handles_mode is True
        no_handles = resolve_context_route(ContextRequest(query="x"))
        assert no_handles.route.handles_mode is False

    def test_filters_active_flag(self) -> None:
        resolution = resolve_context_route(
            ContextRequest(query="x", filters=ContextFilters(languages=["python"]))
        )
        assert resolution.route.filters_active is True
        no_filters = resolve_context_route(ContextRequest(query="x"))
        assert no_filters.route.filters_active is False

    def test_profile_source_reported(self) -> None:
        with_profile = resolve_context_route(
            ContextRequest(query="x", profile=RetrievalProfile.FAST)
        )
        assert with_profile.route.profile_source == "explicit"
        assert with_profile.route.resolved_profile == RetrievalProfile.FAST
        without_profile = resolve_context_route(ContextRequest(query="x"))
        assert without_profile.route.profile_source == "none"
        assert without_profile.route.resolved_profile is None

    def test_reasons_are_non_empty(self) -> None:
        resolution = resolve_context_route(ContextRequest(query="x"))
        assert resolution.route.reasons


# ---------------------------------------------------------------------------
# apply_context_filters
# ---------------------------------------------------------------------------


def _bundle_with_chunks(paths: list[str]) -> ContextBundle:
    chunks = [make_chunk(f"c{i}", path) for i, path in enumerate(paths)]
    ranked = [RankedChunk(chunk=chunk, relevance_score=1.0, final_score=1.0) for chunk in chunks]
    bundle = ContextBundle(
        query="q",
        chunks=ranked,
        token_count=sum(c.token_count for c in chunks),
        token_budget=8192,
    )
    bundle.receipt = build_context_receipt(
        bundle,
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
    )
    return bundle


class TestApplyContextFilters:
    def test_removes_non_matching_chunks(self) -> None:
        bundle = _bundle_with_chunks(["src/auth/login.py", "src/tests/test_login.py"])
        filtered = apply_context_filters(bundle, ContextFilters(exclude_paths=["src/tests/**"]))
        assert [rc.chunk.file_path for rc in filtered.chunks] == ["src/auth/login.py"]

    def test_no_op_when_nothing_filtered(self) -> None:
        bundle = _bundle_with_chunks(["src/auth/login.py"])
        original_chunks = list(bundle.chunks)
        filtered = apply_context_filters(bundle, ContextFilters(languages=["python"]))
        assert filtered.chunks == original_chunks

    def test_moves_filtered_items_to_skipped_with_reason(self) -> None:
        bundle = _bundle_with_chunks(["src/auth/login.py", "src/tests/test_login.py"])
        assert bundle.receipt is not None
        assert len(bundle.receipt.returned_context) == 2
        filtered = apply_context_filters(bundle, ContextFilters(exclude_paths=["src/tests/**"]))
        assert filtered.receipt is not None
        assert len(filtered.receipt.returned_context) == 1
        skipped = [
            item
            for item in filtered.receipt.skipped_candidates
            if item.reason == ContextSkippedReason.FILTER_EXCLUDED
        ]
        assert len(skipped) == 1
        assert skipped[0].handle == chunk_handle("c1")

    def test_recomputes_totals_and_token_count(self) -> None:
        bundle = _bundle_with_chunks(["src/auth/login.py", "src/tests/test_login.py"])
        assert bundle.receipt is not None
        original_skipped_total = bundle.receipt.skipped_total
        filtered = apply_context_filters(bundle, ContextFilters(exclude_paths=["src/tests/**"]))
        assert filtered.receipt is not None
        assert filtered.receipt.returned_total == 1
        assert filtered.receipt.skipped_total == original_skipped_total + 1
        assert filtered.token_count == make_chunk("c0", "src/auth/login.py").token_count

    def test_resums_receipt_consumed_budget(self) -> None:
        bundle = _bundle_with_chunks(["src/auth/login.py", "src/tests/test_login.py"])
        assert bundle.receipt is not None
        original_consumed = bundle.receipt.token_budget.consumed
        filtered = apply_context_filters(bundle, ContextFilters(exclude_paths=["src/tests/**"]))
        assert filtered.receipt is not None
        assert filtered.receipt.token_budget.consumed == filtered.token_count
        assert filtered.receipt.token_budget.consumed < original_consumed


# ---------------------------------------------------------------------------
# ContextResult computed views
# ---------------------------------------------------------------------------


class TestContextResultComputedViews:
    def _make_result(self) -> ContextResult:
        question = "how does auth work?"
        chunk = make_chunk("c1", "src/auth/login.py")
        ranked = RankedChunk(chunk=chunk, relevance_score=0.9, final_score=0.9)
        bundle = ContextBundle(
            query=question,
            chunks=[ranked],
            structural_context=StructuralContext(relevant_modules=["auth"]),
            token_count=chunk.token_count,
            token_budget=8192,
            receipt=ContextReceipt(
                query=question,
                token_budget=ContextReceiptTokenBudget(requested=8192, consumed=chunk.token_count),
                index_revision="rev",
                freshness=ContextFreshness.CLEAN,
                returned_context=[],
                returned_total=1,
                context_complete=ContextCompletenessStatus.COMPLETE,
                context_complete_reason=ContextCompletenessReason.COMPLETE,
                recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
            ),
        )
        bundle.receipt = build_context_receipt(
            bundle, index_revision="rev", freshness=ContextFreshness.CLEAN
        )
        route = ContextRouteDecision(
            resolved_intent=QueryIntent.GENERAL,
            intent_source="auto",
            resolved_modality=QueryModality.NL_TO_PL,
            resolved_profile=None,
            profile_source="none",
            resolved_budget_tier=BudgetTier.STANDARD,
            token_budget_requested=8192,
            budget_source="intent_default",
            handles_mode=False,
            filters_active=False,
        )
        return ContextResult(bundle=bundle, route=route)

    def test_candidate_map_matches_receipt(self) -> None:
        result = self._make_result()
        assert result.candidate_map == result.bundle.receipt.returned_context  # type: ignore[union-attr]

    def test_fetch_handles_match_candidate_map(self) -> None:
        result = self._make_result()
        assert result.fetch_handles == [item.handle for item in result.candidate_map]
        assert result.fetch_handles == [chunk_handle("c1")]

    def test_selected_code_matches_bundle_chunks(self) -> None:
        result = self._make_result()
        assert result.selected_code == result.bundle.chunks

    def test_relation_paths_matches_structural_context(self) -> None:
        result = self._make_result()
        assert result.relation_paths == result.bundle.structural_context
        assert result.relation_paths.relevant_modules == ["auth"]

    def test_receipt_matches_bundle_receipt(self) -> None:
        result = self._make_result()
        assert result.receipt is result.bundle.receipt

    def test_next_action_matches_receipt(self) -> None:
        result = self._make_result()
        assert result.next_action == result.bundle.receipt.recommended_next_action  # type: ignore[union-attr]

    def test_next_action_none_without_receipt(self) -> None:
        result = self._make_result()
        result.bundle.receipt = None
        assert result.next_action is None
        assert result.candidate_map == []
        assert result.fetch_handles == []

    def test_markdown_render_includes_route_and_candidates(self) -> None:
        result = self._make_result()
        rendered = render_context_markdown(result)
        assert "## Route" in rendered
        assert "## Candidate map" in rendered
        assert chunk_handle("c1") in rendered
        assert "## Selected code" in rendered


# ---------------------------------------------------------------------------
# End-to-end context() against a real fixture repo
# ---------------------------------------------------------------------------
_QUESTION = "How does authentication work?"


class TestContextEndToEnd:
    def test_context_returns_candidate_map_and_handles(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        result = context(source, ContextRequest(query=_QUESTION), config=Config(cache=False))

        assert isinstance(result, ContextResult)
        assert len(result.selected_code) > 0
        assert len(result.candidate_map) > 0
        assert result.fetch_handles
        assert result.receipt is not None
        assert result.next_action is not None
        assert result.route.resolved_intent is not None

    def test_context_explicit_intent_sets_route_and_budget(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        request = ContextRequest(query=_QUESTION, intent=QueryIntent.DEBUGGING)
        result = context(source, request, config=Config(cache=False))

        assert result.route.resolved_intent == QueryIntent.DEBUGGING
        assert result.route.intent_source == "explicit"
        assert result.route.token_budget_requested == INTENT_TOKEN_BUDGETS[QueryIntent.DEBUGGING]

    def test_context_profile_propagates_to_metadata(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        request = ContextRequest(query=_QUESTION, profile=RetrievalProfile.FAST)
        result = context(source, request, config=Config(cache=False))

        assert result.bundle.retrieval_metadata.retrieval_profile == "fast"
        assert result.route.resolved_profile == RetrievalProfile.FAST

    def test_context_filters_exclude_matching_files(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        baseline = context(source, ContextRequest(query=_QUESTION), config=config)
        excluded_path = baseline.bundle.chunks[0].chunk.file_path

        filtered_request = ContextRequest(
            query=_QUESTION,
            filters=ContextFilters(exclude_paths=[excluded_path]),
        )
        filtered = context(source, filtered_request, config=config)

        assert all(rc.chunk.file_path != excluded_path for rc in filtered.selected_code)
        assert filtered.route.filters_active is True
        assert any(
            item.reason == ContextSkippedReason.FILTER_EXCLUDED
            for item in filtered.receipt.skipped_candidates  # type: ignore[union-attr]
        )

    def test_context_handles_mode_fetches_exact_chunk(self, python_simple_repo: Path) -> None:
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        baseline = context(source, ContextRequest(query=_QUESTION), config=config)
        handle = baseline.fetch_handles[0]
        expected_chunk = next(
            rc.chunk for rc in baseline.selected_code if chunk_handle(rc.chunk.id) == handle
        )

        fetched = context(source, ContextRequest(query="ignored", handles=[handle]), config=config)

        assert fetched.route.handles_mode is True
        assert len(fetched.selected_code) == 1
        assert fetched.selected_code[0].chunk.id == expected_chunk.id
        assert fetched.selected_code[0].chunk.content == expected_chunk.content

    def test_context_invalid_request_fails_loud(self) -> None:
        with pytest.raises(ValidationError):
            ContextRequest(query="")
