"""Deterministic symbolic evidence reranking for the bounded-rerank lane.

Reranks a *compact* candidate set by cheap, local evidence — file-path matches,
symbol-name matches, and query-term overlap — blended with the original
retrieval rank. No model, no network: this symbolic pass is the always-available
bounded rerank and guarantees a deterministic reordering. The benchmark lane may
additionally reuse a local cross-encoder reranker when one is already loaded in
process, but the symbolic pass is what bounds and grounds the result. Inspired by
BLAgent's two-phase, evidence-grounded rerank over a compact candidate set
(arXiv:2605.17965).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import CodeChunk

# Hard caps for the bounded rerank lane. The candidate cap bounds rerank work
# structurally (only the top-N retrieved chunks are ever reranked); the latency
# cap is the wall-clock budget a model rerank pass must stay under to be applied.
DEFAULT_CANDIDATE_CAP = 8
DEFAULT_LATENCY_CAP_MS = 750.0

# Evidence weights (deterministic). A direct file-path match is the strongest
# signal, then a symbol-name hit, then query-term overlap in the body, with the
# original retrieval rank as a small tie-breaking prior.
_PATH_WEIGHT = 2.0
_SYMBOL_WEIGHT = 1.5
_TERM_WEIGHT = 1.0
_RANK_PRIOR_WEIGHT = 0.5

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_PATH_RE = re.compile(
    r"[\w.\-/]*[\w\-]+/[\w.\-/]*\.[A-Za-z]{1,6}"  # dir/.../file.ext
    r"|\b[\w\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|cs|rb|cpp|cc|c|h|hpp|swift|php"
    r"|scala|sql|sh|yaml|yml|toml|json|md)\b"  # file.ext
)


@dataclass(frozen=True)
class RerankCaps:
    """Hard candidate and latency caps for the bounded rerank lane."""

    candidate_cap: int = DEFAULT_CANDIDATE_CAP
    latency_cap_ms: float = DEFAULT_LATENCY_CAP_MS


@dataclass(frozen=True)
class QuerySignals:
    """Cheap deterministic signals extracted from the query for evidence scoring."""

    terms: frozenset[str]
    paths: tuple[str, ...]


def query_signals(question: str) -> QuerySignals:
    """Extract lowercase query terms and file-path mentions for evidence scoring."""
    terms = frozenset(word.lower() for word in _WORD_RE.findall(question) if len(word) >= 2)
    paths = tuple(path.lower() for path in _PATH_RE.findall(question))
    return QuerySignals(terms=terms, paths=paths)


@dataclass(frozen=True)
class EvidenceComponents:
    """Decomposed deterministic evidence for one candidate.

    Carries the raw component signals so a consumer can both score a candidate
    (``score``) and ask whether it has *strong* symbolic evidence — a direct
    file-path match or a symbol-name hit (``has_strong_evidence``) —
    independently of the term-overlap and rank-prior contributions that the
    blended score folds in.
    """

    path_hit: float
    symbol_hit: float
    term_overlap: float
    rank_prior: float

    @property
    def score(self) -> float:
        """Weighted blend of the components (the bounded-rerank evidence score)."""
        return (
            _PATH_WEIGHT * self.path_hit
            + _SYMBOL_WEIGHT * self.symbol_hit
            + _TERM_WEIGHT * self.term_overlap
            + _RANK_PRIOR_WEIGHT * self.rank_prior
        )

    @property
    def has_strong_evidence(self) -> bool:
        """True when a direct path match or symbol-name hit is present.

        Strong evidence is exactly the path/symbol signal a guard protects; it
        excludes the weaker term-overlap and rank-prior contributions, which are
        not on their own an exact-path or exact-symbol match.
        """
        return self.path_hit > 0.0 or self.symbol_hit > 0.0


def evidence_components(
    chunk: CodeChunk, signals: QuerySignals, *, rank: int, total: int
) -> EvidenceComponents:
    """Decompose one candidate's deterministic evidence at a given retrieval rank."""
    path = chunk.file_path.lower()
    path_hit = 1.0 if any(p and (p in path or path.endswith(p)) for p in signals.paths) else 0.0
    symbol_text = " ".join(part for part in (chunk.symbol_name, chunk.qualified_name) if part)
    symbol_text = symbol_text.lower()
    symbol_hit = 1.0 if any(term in symbol_text for term in signals.terms) else 0.0
    content = chunk.content.lower()
    overlap = sum(1 for term in signals.terms if term in content)
    term_overlap = overlap / len(signals.terms) if signals.terms else 0.0
    rank_prior = (total - rank) / total if total else 0.0
    return EvidenceComponents(
        path_hit=path_hit,
        symbol_hit=symbol_hit,
        term_overlap=term_overlap,
        rank_prior=rank_prior,
    )


def evidence_score(chunk: CodeChunk, signals: QuerySignals, *, rank: int, total: int) -> float:
    """Deterministic evidence score for one candidate at a given retrieval rank."""
    return evidence_components(chunk, signals, rank=rank, total=total).score


def symbolic_scores(chunks: list[CodeChunk], signals: QuerySignals) -> list[float]:
    """Evidence scores for the compact candidate set, in candidate order."""
    total = len(chunks)
    return [
        evidence_score(chunk, signals, rank=index, total=total)
        for index, chunk in enumerate(chunks)
    ]
