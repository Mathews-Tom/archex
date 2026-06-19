"""Tests for deterministic query-modality and budget-tier classification."""

from __future__ import annotations

import pytest

from archex.serve.intent import QueryIntent
from archex.serve.modality import (
    BudgetTier,
    ModalityClassification,
    QueryModality,
    budget_tier,
    classify_modality,
    classify_query,
    extract_signals,
)

_STACK_TRACE = (
    'Traceback (most recent call last): File "auth.py", line 42, in validate_token raise TokenError'
)
_MIXED_ISSUE = (
    "Login fails with a 500 error after the session expires. "
    'Traceback (most recent call last): File "auth.py", line 88, '
    "in refresh raise SessionError"
)


class TestClassifyModality:
    def test_natural_language_query_is_nl_to_pl(self) -> None:
        assert classify_modality("How does authentication work in this project?") == (
            QueryModality.NL_TO_PL
        )

    def test_code_heavy_query_is_pl_to_pl(self) -> None:
        assert classify_modality("AuthManager.validate_token JWTError refresh_session") == (
            QueryModality.PL_TO_PL
        )

    def test_stack_trace_is_pl_to_pl(self) -> None:
        # A bare stack trace has trace boilerplate but no task description.
        assert classify_modality(_STACK_TRACE) == QueryModality.PL_TO_PL

    def test_path_symbol_query_is_pl_to_pl(self) -> None:
        assert classify_modality("where is validate_token defined in services/auth.py") == (
            QueryModality.PL_TO_PL
        )

    def test_mixed_issue_query_is_mixed(self) -> None:
        # Natural-language description plus a stack trace.
        assert classify_modality(_MIXED_ISSUE) == QueryModality.MIXED

    def test_direct_target_with_long_description_is_mixed(self) -> None:
        question = (
            "I keep seeing intermittent login failures and I think the bug lives "
            "in the token refresh logic in services/auth.py please investigate"
        )
        assert classify_modality(question) == QueryModality.MIXED

    def test_inline_code_fence_with_description_is_mixed(self) -> None:
        question = (
            "The session cache never clears even after logout, can you check why "
            "`cache.invalidate(session_id)` is not being called here"
        )
        assert classify_modality(question) == QueryModality.MIXED

    @pytest.mark.parametrize("question", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_query_is_nl_to_pl(self, question: str) -> None:
        assert classify_modality(question) == QueryModality.NL_TO_PL

    def test_plural_prose_does_not_flip_to_code(self) -> None:
        # ``worker(s)``/``e.g.`` must not register as symbol/identifier mentions.
        question = (
            "How do the worker(s) and scheduler(s) coordinate, e.g. when the "
            "queue is full and back-pressure kicks in across the cluster"
        )
        assert classify_modality(question) == QueryModality.NL_TO_PL

    def test_lowercase_error_colon_prose_is_not_a_trace(self) -> None:
        question = (
            "What is the error: handling strategy used across the validation "
            "layer for malformed user input and partial writes"
        )
        assert classify_modality(question) == QueryModality.NL_TO_PL


class TestExtractSignals:
    def test_identifier_density_counts_identifiers(self) -> None:
        signals = extract_signals("AuthManager validate_token refresh")
        assert signals.identifier_count == 2
        assert signals.word_count == 3
        assert abs(signals.identifier_density - 2 / 3) < 1e-9

    def test_code_fence_detected(self) -> None:
        assert extract_signals("see `do_thing()` here").has_code_fence is True
        assert extract_signals("plain english question only").has_code_fence is False

    def test_stack_trace_detected(self) -> None:
        assert extract_signals(_STACK_TRACE).has_stack_trace is True
        assert extract_signals("no trace here at all").has_stack_trace is False

    def test_path_and_symbol_mentions(self) -> None:
        signals = extract_signals("look at services/auth.py and Session.refresh")
        assert signals.path_mentions >= 1
        assert signals.symbol_mentions >= 1

    def test_trace_boilerplate_is_not_description(self) -> None:
        # Trace boilerplate words must not inflate the description count.
        assert extract_signals(_STACK_TRACE).natural_language_word_count < 8
        assert extract_signals(_MIXED_ISSUE).natural_language_word_count >= 8

    def test_query_intent_recorded(self) -> None:
        signals = extract_signals("where is validate_token defined")
        assert isinstance(signals.query_intent, QueryIntent)

    def test_empty_query_signals_do_not_raise(self) -> None:
        signals = extract_signals("")
        assert signals.identifier_density == 0.0
        assert signals.natural_language_ratio == 0.0
        assert signals.word_count == 0


class TestBudgetTier:
    @pytest.mark.parametrize(
        ("token_budget", "expected"),
        [
            (1024, BudgetTier.TIGHT),
            (2048, BudgetTier.TIGHT),
            (3072, BudgetTier.TIGHT),
            (3073, BudgetTier.STANDARD),
            (4096, BudgetTier.STANDARD),
            (8192, BudgetTier.STANDARD),
            (8193, BudgetTier.LARGE),
            (16384, BudgetTier.LARGE),
        ],
    )
    def test_budget_tier_boundaries(self, token_budget: int, expected: BudgetTier) -> None:
        assert budget_tier(token_budget) == expected


class TestClassifyQuery:
    def test_returns_modality_and_tier(self) -> None:
        result = classify_query("How does the auth pipeline work?", 8192)
        assert isinstance(result, ModalityClassification)
        assert result.modality == QueryModality.NL_TO_PL
        assert result.budget_tier == BudgetTier.STANDARD
        assert result.token_budget == 8192
        assert result.reasons  # non-empty rationale

    def test_provenance_is_string_valued(self) -> None:
        result = classify_query("AuthManager.validate_token refresh_session", 2048)
        provenance = result.to_provenance()
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in provenance.items())
        assert provenance["modality"] == QueryModality.PL_TO_PL.value
        assert provenance["budget_tier"] == BudgetTier.TIGHT.value
        assert provenance["token_budget"] == "2048"
        assert "query_intent" in provenance
        assert provenance["reasons"]

    def test_reasons_include_modality_and_budget(self) -> None:
        result = classify_query("How does retrieval work end to end?", 16384)
        joined = " ".join(result.reasons)
        assert "nl_to_pl" in joined
        assert "large" in joined
