"""Query intent classification and scoring weight routing."""

from __future__ import annotations

import re
from enum import StrEnum

from archex.models import ScoringWeights


class QueryIntent(StrEnum):
    """Classification of query intent for scoring weight routing."""

    DEFINITION_LOOKUP = "definition_lookup"
    ARCHITECTURE_BROAD = "architecture_broad"
    USAGE_SEARCH = "usage_search"
    DEBUGGING = "debugging"
    CLI = "cli"
    GENERAL = "general"


# Scoring weight presets per intent — each sums to 1.0.
INTENT_WEIGHTS: dict[QueryIntent, ScoringWeights] = {
    # Symbol lookup: maximize relevance, minimize structural noise
    QueryIntent.DEFINITION_LOOKUP: ScoringWeights(
        relevance=0.90, structural=0.04, type_coverage=0.04, cohesion=0.02
    ),
    # Architecture: balance relevance with structural and cohesion signals
    QueryIntent.ARCHITECTURE_BROAD: ScoringWeights(
        relevance=0.60, structural=0.15, type_coverage=0.05, cohesion=0.20
    ),
    # Usage search: structural signal reveals callers and importers
    QueryIntent.USAGE_SEARCH: ScoringWeights(
        relevance=0.65, structural=0.20, type_coverage=0.05, cohesion=0.10
    ),
    # Debugging: relevance-heavy but structural helps trace error propagation
    QueryIntent.DEBUGGING: ScoringWeights(
        relevance=0.80, structural=0.10, type_coverage=0.05, cohesion=0.05
    ),
    # CLI lifecycle: favor direct command modules while preserving graph context
    QueryIntent.CLI: ScoringWeights(
        relevance=0.75, structural=0.10, type_coverage=0.05, cohesion=0.10
    ),
    # General: default balanced weights (same as current defaults)
    QueryIntent.GENERAL: ScoringWeights(
        relevance=0.80, structural=0.08, type_coverage=0.04, cohesion=0.08
    ),
}


# Pattern groups for intent detection — order matters, first match wins.
# Each group is (intent, compiled_patterns) where patterns test against
# the lowercased query string.

_DEFINITION_PATTERNS = [
    re.compile(r"\bwhere\s+is\b.*\bdefin"),  # "where is X defined"
    re.compile(r"\bdefini(?:tion|ed)\s+of\b"),  # "definition of X"
    re.compile(r"\bimplement(?:s|ation)\s+of\b"),  # "implementation of X"
    re.compile(r"\bsource\s+(?:code\s+)?(?:of|for)\b"),  # "source of/for X"
]

# Detect identifiers: CamelCase, snake_case, dotted.names, UPPER_CASE
_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"  # CamelCase (2+ parts)
    r"|[a-z]+_[a-z_]+"  # snake_case
    r"|[a-zA-Z]+\.[a-zA-Z]+"  # dotted.name
    r"|[A-Z][A-Z_]{2,}"  # UPPER_CASE (3+ chars)
    r")"
)

_ARCHITECTURE_PATTERNS = [
    # "how does X work" is too broad — matches implementation queries like
    # "How does pydantic chain validators?" which are relevance-heavy.
    # Only match when combined with architecture-specific nouns.
    re.compile(r"\bhow\s+does\b.*\b(?:pipeline|middleware|routing|system|architecture)\b"),
    re.compile(r"\barchitectur"),  # "architecture"
    re.compile(r"\bpipeline\b"),  # "pipeline"
    re.compile(r"\bmiddleware\b"),  # "middleware"
    re.compile(r"\bsystem\s+design\b"),  # "system design"
    re.compile(r"\boverview\b"),  # "overview"
    re.compile(r"\bflow\b"),  # "flow"
    re.compile(r"\blifecycle\b"),  # "lifecycle"
]

_USAGE_PATTERNS = [
    re.compile(r"\bwho\s+calls\b"),  # "who calls X"
    re.compile(r"\bwhere\s+is\b.*\bused\b"),  # "where is X used"
    re.compile(r"\bcallers?\s+of\b"),  # "callers of X"
    re.compile(r"\busage\s+of\b"),  # "usage of X"
    re.compile(r"\bwhat\s+(?:calls|uses|imports)\b"),  # "what calls/uses/imports X"
    re.compile(r"\breferences?\s+to\b"),  # "references to X"
]

_DEBUGGING_PATTERNS = [
    re.compile(r"\berror\b"),  # "error"
    re.compile(r"\bbug\b"),  # "bug"
    re.compile(r"exception"),  # "exception" or "NullPointerException"
    re.compile(r"\bfail(?:s|ing|ure)?\b"),  # "fails", "failing", "failure"
    re.compile(r"\bcrash"),  # "crash", "crashes"
    re.compile(r"\bwhy\s+does\b.*\b(?:throw|raise|return)\b"),  # "why does X throw/raise"
    re.compile(r"\btraceback\b"),  # "traceback"
    re.compile(r"\bstack\s*trace\b"),  # "stack trace"
]

_CLI_PATTERNS = [
    re.compile(r"\barchex\s+(?:init|reset|status|index|config|cache|delta)\b"),
]


def classify_intent(question: str) -> QueryIntent:
    """Classify a search query into an intent bucket for scoring weight routing.

    Uses pattern matching against the lowercased query. The classification
    priority order is: definition_lookup (explicit patterns) > debugging >
    usage_search > cli > definition_lookup (identifier heuristic) >
    architecture_broad > general.

    Explicit definition patterns (e.g. "where is X defined") take highest
    priority. Debugging and usage patterns are checked before the identifier
    heuristic so that "why does X fail?" and "where is X used?" are not
    incorrectly pulled into definition lookup by the identifier signal.

    Args:
        question: The search query to classify.

    Returns:
        The detected QueryIntent enum value.
    """
    q = question.lower()

    # Definition lookup: explicit definition patterns (highest specificity)
    for pat in _DEFINITION_PATTERNS:
        if pat.search(q):
            return QueryIntent.DEFINITION_LOOKUP

    # Debugging: error/exception/failure patterns — check before identifier heuristic
    # so that "why does X fail?" routes to debugging, not definition lookup
    for pat in _DEBUGGING_PATTERNS:
        if pat.search(q):
            return QueryIntent.DEBUGGING

    # Usage search: who calls, where used — check before identifier heuristic
    # so that "where is X used?" routes to usage, not definition lookup
    for pat in _USAGE_PATTERNS:
        if pat.search(q):
            return QueryIntent.USAGE_SEARCH

    # CLI lifecycle: repo-local command lifecycle verbs such as reset/status/cache
    for pat in _CLI_PATTERNS:
        if pat.search(q):
            return QueryIntent.CLI

    # Identifier heuristic: classify as definition lookup when query is dominated
    # by CamelCase/snake_case/dotted identifiers (symbol-targeted queries with no
    # other structural signals)
    identifiers = _IDENTIFIER_RE.findall(question)  # case-sensitive search
    words = question.split()
    if identifiers and len(identifiers) >= len(words) * 0.20:
        return QueryIntent.DEFINITION_LOOKUP

    # Architecture broad: how does, pipeline, middleware, overview
    for pat in _ARCHITECTURE_PATTERNS:
        if pat.search(q):
            return QueryIntent.ARCHITECTURE_BROAD

    return QueryIntent.GENERAL


def weights_for_query(question: str) -> ScoringWeights:
    """Return scoring weights appropriate for the query's detected intent.

    Convenience function combining classify_intent() with INTENT_WEIGHTS lookup.
    """
    intent = classify_intent(question)
    return INTENT_WEIGHTS[intent]
