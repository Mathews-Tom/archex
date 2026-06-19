"""Deterministic dual-perspective query transformation for the benchmark lane.

Rewrites a user query into two subqueries from complementary perspectives:

* a *structural* subquery that emphasises the code-shaped signals an exact
  retriever keys on — file paths, dotted/CamelCase/snake_case identifiers, call
  sites, class/function names, error/exception types, stack-trace frames, and
  import targets;
* a *behavioural* subquery that preserves the natural-language description — the
  reported symptom, the expected behaviour, and the domain nouns — including the
  domain nouns embedded inside identifiers (``parse_imports`` -> ``parse
  imports``) so prose-leaning retrieval still sees them.

Rule-based only: no hosted LLM calls, no network. Inspired by BLAgent's
dual-perspective query transformation (arXiv:2605.17965). Used exclusively by the
benchmark-only ``archex_query_dual_transform`` lane; product retrieval paths are
unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Code-token shapes. These mirror the structural shapes in
# ``archex.serve.modality``; they are duplicated locally (as
# ``archex.serve.intent`` already duplicates the identifier shape) to keep this
# benchmark lane decoupled from the product classifier's private internals.
_PATH_RE = re.compile(
    r"[\w.\-/]*[\w\-]+/[\w.\-/]*\.[A-Za-z]{1,6}"  # dir/.../file.ext
    r"|\b[\w\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|cs|rb|cpp|cc|c|h|hpp|swift|php"
    r"|scala|sql|sh|yaml|yml|toml|json|md)\b"  # file.ext
)
_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]+"  # dotted.name (2+ char tail)
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"  # CamelCase (2+ parts)
    r"|[a-z0-9]+_[a-z0-9_]+"  # snake_case
    r"|[A-Z][A-Z0-9_]{2,}"  # UPPER_CASE (3+ chars)
    r")"
)
# Call site: ``name(`` excluding the ``(s)``/``(es)`` plural prose forms.
_CALL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\((?!s\)|es\))")
# Error/exception/warning type names, including bare single-word ones.
_ERROR_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)\b")
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
# Import statements: ``from pkg.mod import a, b`` or ``import pkg.mod, other``.
_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+(.+)$")
_IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")

_NL_STRIP_CHARS = ".,;:!?()[]{}\"'`"
# CamelCase split points: lower/digit -> upper, and the ACRONYMWord boundary.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Non-identifier-word separators (underscore included so snake_case splits).
_NONWORD_RE = re.compile(r"[^A-Za-z0-9]+")


def _ordered_unique(tokens: list[str]) -> list[str]:
    """Deduplicate tokens preserving first-occurrence order, dropping blanks."""
    return [token for token in dict.fromkeys(tokens) if token]


def _strip_code_spans(question: str) -> str:
    """Blank out code fences, stack-trace spans, and file paths for prose."""
    stripped = _CODE_FENCE_RE.sub(" ", question)
    stripped = _STACK_TRACE_RE.sub(" ", stripped)
    return _PATH_RE.sub(" ", stripped)


def _is_natural_language_word(token: str) -> bool:
    """Return True for a plain English-style word (not a code identifier)."""
    if len(token) < 2 or "_" in token or "." in token or "/" in token:
        return False
    if not token.isalpha():
        return False
    # CamelCase / ALLCAPS read as code, not prose.
    return token == token.lower() or token == token.capitalize()


def _split_import_names(clause: str) -> list[str]:
    """Split an import clause into target names, dropping aliases and ``*``."""
    names: list[str] = []
    for raw in clause.split(","):
        name = raw.strip().split(" as ", 1)[0].strip()
        if name and name != "*":
            names.append(name)
    return names


def _import_tokens(question: str) -> list[str]:
    """Extract module/symbol names from import statements in the query."""
    tokens: list[str] = []
    for line in question.splitlines():
        from_match = _FROM_IMPORT_RE.match(line)
        if from_match:
            tokens.append(from_match.group(1))
            tokens.extend(_split_import_names(from_match.group(2)))
            continue
        import_match = _IMPORT_RE.match(line)
        if import_match:
            tokens.extend(_split_import_names(import_match.group(1)))
    return tokens


def _split_identifier(identifier: str) -> list[str]:
    """Split an identifier into lowercase domain-noun words (length >= 2)."""
    words: list[str] = []
    for part in _NONWORD_RE.split(identifier):
        words.extend(_CAMEL_BOUNDARY_RE.split(part))
    return [word.lower() for word in words if len(word) >= 2]


def _structural_tokens(question: str) -> list[str]:
    """Code-shaped tokens emphasised by the structural subquery, in query order."""
    tokens: list[str] = []
    tokens.extend(_PATH_RE.findall(question))
    tokens.extend(_ERROR_RE.findall(question))
    tokens.extend(call.rstrip("(") for call in _CALL_RE.findall(question))
    tokens.extend(_IDENTIFIER_RE.findall(question))
    tokens.extend(_import_tokens(question))
    return _ordered_unique(tokens)


def _behavioral_tokens(question: str) -> list[str]:
    """Natural-language words plus identifier-derived domain nouns, in order."""
    prose = _strip_code_spans(question)
    nl_words = [
        core
        for token in prose.split()
        if (core := token.strip(_NL_STRIP_CHARS)) and _is_natural_language_word(core)
    ]
    domain_nouns: list[str] = []
    for identifier in _IDENTIFIER_RE.findall(question):
        domain_nouns.extend(_split_identifier(identifier))
    return _ordered_unique([*nl_words, *domain_nouns])


@dataclass(frozen=True)
class DualQuery:
    """A user query rewritten into structural and behavioural subqueries.

    ``*_from_fallback`` is True when a perspective extracted no tokens and the
    original query string is used verbatim so retrieval always has a query.
    """

    original: str
    structural: str
    behavioral: str
    structural_token_count: int
    behavioral_token_count: int
    structural_from_fallback: bool
    behavioral_from_fallback: bool
    has_stack_trace: bool

    def to_provenance(self) -> dict[str, str]:
        """Flatten the dual query into string-valued provenance entries."""
        return {
            "subquery_structural": self.structural,
            "subquery_behavioral": self.behavioral,
            "structural_token_count": str(self.structural_token_count),
            "behavioral_token_count": str(self.behavioral_token_count),
            "structural_from_fallback": str(self.structural_from_fallback).lower(),
            "behavioral_from_fallback": str(self.behavioral_from_fallback).lower(),
            "has_stack_trace": str(self.has_stack_trace).lower(),
        }


def transform_query(question: str) -> DualQuery:
    """Rewrite a query into deterministic structural and behavioural subqueries."""
    structural_tokens = _structural_tokens(question)
    behavioral_tokens = _behavioral_tokens(question)
    structural = " ".join(structural_tokens)
    behavioral = " ".join(behavioral_tokens)
    return DualQuery(
        original=question,
        structural=structural or question,
        behavioral=behavioral or question,
        structural_token_count=len(structural_tokens),
        behavioral_token_count=len(behavioral_tokens),
        structural_from_fallback=not structural,
        behavioral_from_fallback=not behavioral,
        has_stack_trace=_STACK_TRACE_RE.search(question) is not None,
    )
