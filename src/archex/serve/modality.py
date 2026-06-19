"""Deterministic query-modality and budget-tier classification.

Cheap, local signals that route benchmark-only task-aware retrieval. Nothing in
this module is wired into the product query path; it exists so a benchmark
strategy can choose sparse-vs-dense retrieval by query modality and token budget
without any hosted calls, telemetry, or model dependency.

The classifier reuses the existing intent heuristics in :mod:`archex.serve.intent`
and adds cheap lexical signals: identifier density, code fences/snippets,
stack-trace/path evidence, direct path/symbol mentions, and a natural-language
ratio. Budget tiers are anchored on the product token-budget conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from archex.serve.intent import DEFAULT_TOKEN_BUDGET, QueryIntent, classify_intent


class QueryModality(StrEnum):
    """Coarse query modality used to route retrieval behaviour."""

    PL_TO_PL = "pl_to_pl"
    NL_TO_PL = "nl_to_pl"
    MIXED = "mixed"


class BudgetTier(StrEnum):
    """Deterministic token-budget tier used to scale retrieval aggressiveness."""

    TIGHT = "tight"
    STANDARD = "standard"
    LARGE = "large"


# Budget-tier cutoffs anchored on the product budget conventions in
# ``archex.serve.intent``. Symbol-lookup-sized budgets are tight; the product
# default ceiling (``DEFAULT_TOKEN_BUDGET``) is the top of standard; anything
# above it has room to consider whole-file or enclosing-block preservation.
TIGHT_MAX_TOKENS = 3072
STANDARD_MAX_TOKENS = DEFAULT_TOKEN_BUDGET

# Modality thresholds. Kept conservative so a single weak signal does not flip a
# clearly natural-language question into a code lane.
_PL_IDENTIFIER_DENSITY = 0.30
_MIXED_NL_WORD_COUNT = 8

# Identifier shapes: dotted.name, CamelCase, snake_case, UPPER_CASE.
_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]+"  # dotted.name (2+ char tail)
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"  # CamelCase (2+ parts)
    r"|[a-z0-9]+_[a-z0-9_]+"  # snake_case
    r"|[A-Z][A-Z0-9_]{2,}"  # UPPER_CASE (3+ chars)
    r")"
)

# File-path mentions: a slash-bearing path with an extension, or a bare
# ``name.ext`` for a known source extension.
_PATH_RE = re.compile(
    r"[\w.\-/]*[\w\-]+/[\w.\-/]*\.[A-Za-z]{1,6}"  # dir/.../file.ext
    r"|\b[\w\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|cs|rb|cpp|cc|c|h|hpp|swift|php"
    r"|scala|sql|sh|yaml|yml|toml|json|md)\b"  # file.ext
)

# Direct symbol mentions: a dotted reference (``Class.method``) or a call site
# (``name(``). The dotted tail needs 2+ chars and the call form skips ``(s)``/
# ``(es)`` plurals so prose like ``e.g.`` and ``handler(s)`` is not a symbol.
_SYMBOL_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]+"
    r"|[A-Za-z_][A-Za-z0-9_]*\((?!s\)|es\))"
)

# Code fences / inline snippets.
_CODE_FENCE_RE = re.compile(r"```|`[^`]+`")

# Stack-trace / traceback evidence across common ecosystems.
_STACK_TRACE_RE = re.compile(
    r"traceback \(most recent call last\)"
    r'|file "[^"]+", line \d+'
    r"|\bat [\w.$]+\([\w.: ]*\)"  # java/js frame
    r"|(?-i:\b[A-Z]\w*(?:Error|Exception)\b) *:"  # ValueError:, NullPointerException:
    r"|\b[\w/]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|cs|rb|cpp|cc|c|h|hpp|swift|php"
    r"|scala):\d+\b",  # path.ext:line
    re.IGNORECASE,
)

_NL_STRIP_CHARS = ".,;:!?()[]{}\"'`"


def _is_natural_language_word(token: str) -> bool:
    """Return True for a plain English-style word (not a code identifier)."""
    core = token.strip(_NL_STRIP_CHARS)
    if len(core) < 2 or "_" in core or "." in core or "/" in core:
        return False
    if not core.isalpha():
        return False
    # CamelCase / ALLCAPS read as code, not prose.
    return core == core.lower() or core == core.capitalize()


def _strip_code_spans(question: str) -> str:
    """Blank out code fences, stack-trace spans, and file paths for prose counts."""
    stripped = _CODE_FENCE_RE.sub(" ", question)
    stripped = _STACK_TRACE_RE.sub(" ", stripped)
    return _PATH_RE.sub(" ", stripped)


@dataclass(frozen=True)
class ModalitySignals:
    """Cheap deterministic signals extracted from a query string."""

    word_count: int
    identifier_count: int
    identifier_density: float
    has_code_fence: bool
    has_stack_trace: bool
    path_mentions: int
    symbol_mentions: int
    natural_language_word_count: int
    natural_language_ratio: float
    query_intent: QueryIntent


@dataclass(frozen=True)
class ModalityClassification:
    """Final modality + budget-tier decision plus the signals that produced it."""

    modality: QueryModality
    budget_tier: BudgetTier
    token_budget: int
    signals: ModalitySignals
    reasons: tuple[str, ...]

    def to_provenance(self) -> dict[str, str]:
        """Flatten the classification into string-valued provenance entries."""
        signals = self.signals
        return {
            "modality": self.modality.value,
            "budget_tier": self.budget_tier.value,
            "token_budget": str(self.token_budget),
            "query_intent": signals.query_intent.value,
            "identifier_density": f"{signals.identifier_density:.3f}",
            "identifier_count": str(signals.identifier_count),
            "word_count": str(signals.word_count),
            "path_mentions": str(signals.path_mentions),
            "symbol_mentions": str(signals.symbol_mentions),
            "has_code_fence": str(signals.has_code_fence).lower(),
            "has_stack_trace": str(signals.has_stack_trace).lower(),
            "natural_language_ratio": f"{signals.natural_language_ratio:.3f}",
            "reasons": "; ".join(self.reasons),
        }


def extract_signals(question: str) -> ModalitySignals:
    """Compute cheap deterministic modality signals for a query string."""
    word_count = len(question.split())
    identifier_count = len(_IDENTIFIER_RE.findall(question))
    # Count description words from prose outside code fences, stack traces, and
    # file paths so trace boilerplate ("most recent call last", "File", "line")
    # is not mistaken for a natural-language task description.
    prose_tokens = _strip_code_spans(question).split()
    nl_word_count = sum(1 for token in prose_tokens if _is_natural_language_word(token))
    denom = max(word_count, 1)
    return ModalitySignals(
        word_count=word_count,
        identifier_count=identifier_count,
        identifier_density=identifier_count / denom,
        has_code_fence=_CODE_FENCE_RE.search(question) is not None,
        has_stack_trace=_STACK_TRACE_RE.search(question) is not None,
        path_mentions=len(_PATH_RE.findall(question)),
        symbol_mentions=len(_SYMBOL_RE.findall(question)),
        natural_language_word_count=nl_word_count,
        natural_language_ratio=nl_word_count / denom,
        query_intent=classify_intent(question),
    )


def classify_modality(question: str) -> QueryModality:
    """Classify a query into a retrieval modality from cheap lexical signals."""
    return _decide_modality(extract_signals(question))[0]


def _decide_modality(signals: ModalitySignals) -> tuple[QueryModality, tuple[str, ...]]:
    strong_code_block = signals.has_code_fence or signals.has_stack_trace
    direct_target = (
        signals.path_mentions > 0
        or signals.symbol_mentions > 0
        or signals.identifier_density >= _PL_IDENTIFIER_DENSITY
    )
    has_description = signals.natural_language_word_count >= _MIXED_NL_WORD_COUNT

    if strong_code_block and has_description:
        return QueryModality.MIXED, ("code block plus a natural-language description -> mixed",)
    if strong_code_block:
        return QueryModality.PL_TO_PL, ("code fence or stack trace with little prose -> pl_to_pl",)
    if direct_target and has_description:
        return QueryModality.MIXED, ("direct path/symbol signals plus a description -> mixed",)
    if direct_target:
        return QueryModality.PL_TO_PL, ("identifier-dense or direct path/symbol query -> pl_to_pl",)
    return QueryModality.NL_TO_PL, ("natural-language query with few code signals -> nl_to_pl",)


def budget_tier(token_budget: int) -> BudgetTier:
    """Map a requested token budget to a deterministic budget tier."""
    if token_budget <= TIGHT_MAX_TOKENS:
        return BudgetTier.TIGHT
    if token_budget <= STANDARD_MAX_TOKENS:
        return BudgetTier.STANDARD
    return BudgetTier.LARGE


def _budget_reason(tier: BudgetTier, token_budget: int) -> str:
    return {
        BudgetTier.TIGHT: f"budget {token_budget} <= {TIGHT_MAX_TOKENS} -> tight",
        BudgetTier.STANDARD: (
            f"budget {token_budget} in ({TIGHT_MAX_TOKENS}, {STANDARD_MAX_TOKENS}] -> standard"
        ),
        BudgetTier.LARGE: f"budget {token_budget} > {STANDARD_MAX_TOKENS} -> large",
    }[tier]


def classify_query(question: str, token_budget: int) -> ModalityClassification:
    """Classify a query's modality and budget tier with full provenance."""
    signals = extract_signals(question)
    modality, modality_reasons = _decide_modality(signals)
    tier = budget_tier(token_budget)
    reasons = (*modality_reasons, _budget_reason(tier, token_budget))
    return ModalityClassification(
        modality=modality,
        budget_tier=tier,
        token_budget=token_budget,
        signals=signals,
        reasons=reasons,
    )
