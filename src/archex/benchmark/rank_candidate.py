"""Direct-evidence file ranking for the M0.3 benchmark candidate.

Reuses only the query/path/symbol/lexical evidence tiers the M0.2 candidate
already computes (`archex.benchmark.coverage_candidate`). Does not read
benchmark task expected-file definitions, change a global score weight, or
force a cross-encoder reranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.coverage_candidate import CoverageSeedDecision
    from archex.models import ContextBundle, RankedChunk

# "identifier" evidence is an explicit CamelCase/snake_case token copied
# verbatim out of the question text -- a rare, highly specific signal.
# "symbol"/"path"/"lexical" evidence is a generic dictionary-word match
# (e.g. "middleware", "error", "request") that is common across many files
# unrelated to the actual answer. Two real-corpus measurements informed this:
#
# 1. Promoting on any evidence tier (identifier/symbol/path/lexical) and
#    reordering the *whole* bundle regressed MRR on 20 of 64 tasks against
#    11 gained: generic-word matches routinely outranked a correctly
#    base-ranked required file.
# 2. Restricting promotion to the identifier tier alone, but still reordering
#    the whole bundle, still regressed MRR net (6 worsened vs 2 gained): an
#    identifier-tier file can itself be a false positive (the term appears in
#    an unrelated file), and -- even when it is not -- promoting it past the
#    base query's *own* top-ranked file is a coin flip the base query's
#    tuned BM25/embedding fusion score is usually better positioned to call.
#
# The only remaining safe use of this evidence: restrict it entirely to the
# candidate-admitted tail (the seed/neighbor files appended after the base
# query's own bundle). A tail file can move toward the front of the tail, but
# can never cross ahead of a base-query file -- so a required file the base
# query already ranked correctly can never be pushed backward by this
# candidate, whatever the evidence says about an unrelated tail file.
_PROMOTABLE_TIER = "identifier"


def evidence_tier_priority(evidence: tuple[str, ...]) -> int:
    """1 when *evidence* carries the promotable (identifier) tier, else 0."""
    return 1 if any(reason.split(":", 1)[0] == _PROMOTABLE_TIER for reason in evidence) else 0


@dataclass(frozen=True)
class DirectEvidenceRerank:
    """A stable, evidence-ordered reorder of one bundle's ranked chunks."""

    bundle: ContextBundle
    promoted_files: tuple[str, ...]


def rerank_by_direct_evidence(
    bundle: ContextBundle,
    decisions: list[CoverageSeedDecision],
    *,
    base_chunk_count: int,
) -> DirectEvidenceRerank:
    """Move identifier-tier-evidenced tail files toward the front of the tail.

    *base_chunk_count* is the number of chunks the base query itself
    contributed, before any seed/neighbor admission ran; `bundle.chunks[:
    base_chunk_count]` is that base query's own ranking. Those chunks are
    **never reordered or displaced** -- only `bundle.chunks[base_chunk_count:]`
    (the candidate-admitted tail) is stable-sorted, by (identifier-tier
    evidence present, evidence score desc, original rank asc). A tail file
    without identifier-tier evidence keeps its exact original relative
    order within the tail.

    Because the base region is frozen and the tail can never cross ahead of
    it, a required file the base query already ranked correctly can never be
    pushed backward by this candidate. This only reorders `bundle.chunks`;
    it never adds or removes a file, so recall/precision/F1 are unaffected.
    """
    best_score_by_file: dict[str, int] = {}
    for decision in decisions:
        if evidence_tier_priority(decision.evidence) == 0:
            continue
        existing = best_score_by_file.get(decision.file)
        if existing is None or decision.score > existing:
            best_score_by_file[decision.file] = decision.score

    base = bundle.chunks[:base_chunk_count]
    tail = list(enumerate(bundle.chunks[base_chunk_count:]))

    def sort_key(pair: tuple[int, RankedChunk]) -> tuple[int, int, int]:
        orig_pos, ranked = pair
        score = best_score_by_file.get(ranked.chunk.file_path)
        return (0, -score, orig_pos) if score is not None else (1, 0, orig_pos)

    ordered_tail = sorted(tail, key=sort_key)
    promoted = tuple(
        chunk.chunk.file_path
        for new_pos, (orig_pos, chunk) in enumerate(ordered_tail)
        if new_pos < orig_pos
    )
    seen: set[str] = set()
    promoted_unique = tuple(f for f in promoted if not (f in seen or seen.add(f)))
    return DirectEvidenceRerank(
        bundle=bundle.model_copy(update={"chunks": [*base, *(chunk for _, chunk in ordered_tail)]}),
        promoted_files=promoted_unique,
    )
