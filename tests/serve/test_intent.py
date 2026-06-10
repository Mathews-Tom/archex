"""Tests for query intent classification and scoring weight routing."""

from __future__ import annotations

import pytest

from archex.models import ScoringWeights
from archex.serve.intent import (
    DEFAULT_TOKEN_BUDGET,
    INTENT_TOKEN_BUDGETS,
    INTENT_WEIGHTS,
    QueryIntent,
    classify_intent,
    token_budget_for_query,
    weights_for_query,
)

# ---------------------------------------------------------------------------
# Definition lookup tests
# ---------------------------------------------------------------------------


def test_classify_where_is_defined() -> None:
    # Arrange
    query = "Where is QuerySet defined?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEFINITION_LOOKUP


def test_classify_definition_of() -> None:
    # Arrange
    query = "definition of SessionMiddleware"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEFINITION_LOOKUP


def test_classify_identifier_heavy() -> None:
    # Arrange — all tokens are identifiers (snake_case / CamelCase)
    query = "QuerySet SQLCompiler execute"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEFINITION_LOOKUP


def test_classify_camelcase_query() -> None:
    # Arrange — "HttpClientFactory" is CamelCase, dominates short query
    query = "How does HttpClientFactory work"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEFINITION_LOOKUP


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------


def test_classify_how_does() -> None:
    # Arrange — no identifiers, generic "how does" phrase
    query = "How does the routing pipeline work?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.ARCHITECTURE_BROAD


def test_classify_middleware() -> None:
    # Arrange
    query = "Explain the middleware architecture"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.ARCHITECTURE_BROAD


def test_classify_lifecycle() -> None:
    # Arrange
    query = "What is the request lifecycle?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.ARCHITECTURE_BROAD


def test_classify_archex_reset_as_cli_command() -> None:
    # Arrange
    query = "archex reset"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.CLI


@pytest.mark.parametrize(
    "query",
    [
        "How does archex initialize repo-local project state?",
        "How does archex inspect whether a repo-local index is fresh, stale, dirty, or corrupt?",
        "How does archex explicitly build or refresh a repo-local index?",
        "How does archex resolve project settings into runtime configuration?",
    ],
)
def test_classify_archex_lifecycle_questions_as_cli(query: str) -> None:
    assert classify_intent(query) == QueryIntent.CLI


# ---------------------------------------------------------------------------
# Usage search tests
# ---------------------------------------------------------------------------


def test_classify_who_calls() -> None:
    # Arrange
    query = "Who calls authenticate()?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.USAGE_SEARCH


def test_classify_where_used() -> None:
    # Arrange — "where is X used" pattern (no "defined")
    query = "Where is session_factory used?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.USAGE_SEARCH


def test_classify_references_to() -> None:
    # Arrange
    query = "Find references to dispatch"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.USAGE_SEARCH


# ---------------------------------------------------------------------------
# Debugging tests
# ---------------------------------------------------------------------------


def test_classify_error() -> None:
    # Arrange
    query = "Why does login throw AuthError?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEBUGGING


def test_classify_exception() -> None:
    # Arrange
    query = "NullPointerException in getData"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEBUGGING


def test_classify_failure() -> None:
    # Arrange
    query = "Tests failing after migration"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEBUGGING


# ---------------------------------------------------------------------------
# General tests
# ---------------------------------------------------------------------------


def test_classify_general() -> None:
    # Arrange
    query = "Show me the user model"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.GENERAL


def test_classify_generic_question() -> None:
    # Arrange
    query = "What does this code do?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.GENERAL


# ---------------------------------------------------------------------------
# Priority / interaction tests
# ---------------------------------------------------------------------------


def test_definition_beats_architecture() -> None:
    # Arrange — "where is...defined" is more specific than "pipeline"
    query = "Where is the middleware pipeline defined?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEFINITION_LOOKUP


def test_debugging_beats_usage() -> None:
    # Arrange — "fail" triggers debugging before usage patterns
    query = "Why does authenticate() fail?"
    # Act
    result = classify_intent(query)
    # Assert
    assert result == QueryIntent.DEBUGGING


# ---------------------------------------------------------------------------
# Weight / ScoringWeights tests
# ---------------------------------------------------------------------------


def test_weights_for_query_returns_scoring_weights() -> None:
    # Arrange
    query = "Where is QuerySet defined?"
    # Act
    result = weights_for_query(query)
    # Assert
    assert isinstance(result, ScoringWeights)


def test_intent_weights_all_sum_to_one() -> None:
    # Arrange / Act / Assert
    for intent, weights in INTENT_WEIGHTS.items():
        total = weights.relevance + weights.structural + weights.type_coverage + weights.cohesion
        assert abs(total - 1.0) < 1e-9, f"{intent} weights sum to {total}, expected 1.0"


def test_intent_token_budgets_route_simple_queries_smaller_than_broad_queries() -> None:
    assert INTENT_TOKEN_BUDGETS[QueryIntent.DEFINITION_LOOKUP] < DEFAULT_TOKEN_BUDGET
    assert (
        INTENT_TOKEN_BUDGETS[QueryIntent.DEFINITION_LOOKUP]
        < INTENT_TOKEN_BUDGETS[QueryIntent.ARCHITECTURE_BROAD]
    )
    assert INTENT_TOKEN_BUDGETS[QueryIntent.ARCHITECTURE_BROAD] == DEFAULT_TOKEN_BUDGET


def test_token_budget_for_query_uses_detected_intent() -> None:
    assert (
        token_budget_for_query("Where is QuerySet defined?")
        == INTENT_TOKEN_BUDGETS[QueryIntent.DEFINITION_LOOKUP]
    )
    assert (
        token_budget_for_query("Explain the middleware architecture")
        == INTENT_TOKEN_BUDGETS[QueryIntent.ARCHITECTURE_BROAD]
    )


def test_behavioral_framework_query_is_not_symbol_lookup() -> None:
    assert classify_intent("How does SQLAlchemy manage sessions?") == QueryIntent.GENERAL
    assert (
        token_budget_for_query("How does SQLAlchemy manage sessions?")
        == INTENT_TOKEN_BUDGETS[QueryIntent.GENERAL]
    )


def test_architecture_weights_higher_structural() -> None:
    # Arrange
    arch_weights = INTENT_WEIGHTS[QueryIntent.ARCHITECTURE_BROAD]
    general_weights = INTENT_WEIGHTS[QueryIntent.GENERAL]
    # Act / Assert
    assert arch_weights.structural > general_weights.structural


def test_definition_weights_higher_relevance() -> None:
    # Arrange
    def_weights = INTENT_WEIGHTS[QueryIntent.DEFINITION_LOOKUP]
    general_weights = INTENT_WEIGHTS[QueryIntent.GENERAL]
    # Act / Assert
    assert def_weights.relevance > general_weights.relevance


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Where is QuerySet defined?", QueryIntent.DEFINITION_LOOKUP),
        ("definition of SessionMiddleware", QueryIntent.DEFINITION_LOOKUP),
        ("How does the routing pipeline work?", QueryIntent.ARCHITECTURE_BROAD),
        ("Explain the middleware architecture", QueryIntent.ARCHITECTURE_BROAD),
        ("Who calls authenticate()?", QueryIntent.USAGE_SEARCH),
        ("Find references to dispatch", QueryIntent.USAGE_SEARCH),
        ("NullPointerException in getData", QueryIntent.DEBUGGING),
        ("Tests failing after migration", QueryIntent.DEBUGGING),
        ("archex reset", QueryIntent.CLI),
        ("Show me the user model", QueryIntent.GENERAL),
    ],
)
def test_classify_parametrized(query: str, expected: QueryIntent) -> None:
    assert classify_intent(query) == expected
