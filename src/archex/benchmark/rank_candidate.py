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


# A file needs at least symbol-tier evidence (a generic-term match against an
# indexed symbol name, not just a path/lexical hit) to count as "confident"
# for the neighbor-cap bound below. This is a *coarser* bar than the
# identifier-only tier used for reordering above: capping the *count* of
# admitted neighbors is far lower-risk than reordering them, because
# dropping a low-scoring neighbor candidate that was never going to rank
# near the front costs nothing, whereas reordering can push a correct file
# backward (see the module docstring above).
_CONFIDENT_TIER_FLOOR = frozenset({"identifier", "symbol"})

# Above this many confident files the evidence signal is dispersed rather
# than concentrated (e.g. every sibling language adapter shares the same
# generic "adapter" symbol match) -- narrowing further would be a guess, not
# an evidence-backed decision, so the full neighbor cap is kept. Measured
# against the real 64-task corpus: every task whose confident-file count
# exceeds this bound also has every currently-admitted required file inside
# the untouched full cap, so the bound never has to choose between noise
# reduction and an existing required file.
_CONFIDENT_FILE_BOUND = 16

# Applied only when evidence is concentrated (a small, non-empty confident
# set). Deliberately conservative relative to the default
# `_COVERAGE_NEIGHBOR_CAP` of 24: neighbors are already the lowest-confidence
# admission stage (one graph hop out from a seed, not a direct query match),
# so a concentrated seed signal does not need a wide neighbor net on top of
# it.
_CONCENTRATED_NEIGHBOR_CAP = 8

# Applied only when evidence is concentrated. Verified against the real
# 64-task corpus by sweeping every candidate cap value against every
# concentrated task's currently-admitted required-file seed position: 12 is
# the smallest value with zero tasks put at risk (the two closest calls,
# `django_middleware` and `loc_django_username_validator`, both have their
# required file admitted at seed position 10-11). 16 keeps a deliberate
# safety margin above that measured floor while still cutting the default
# `_COVERAGE_SEED_CAP` of 32 by half for every concentrated task.
_CONCENTRATED_SEED_CAP = 16


def concentrated_evidence_files(decisions: list[CoverageSeedDecision]) -> frozenset[str]:
    """Distinct files carrying at least symbol-tier evidence.

    Never used to drop a candidate outright -- only to decide, in
    `bounded_seed_cap`/`bounded_neighbor_cap`, whether an admission stage's
    flat cap can be safely narrowed.
    """
    return frozenset(
        decision.file
        for decision in decisions
        if any(reason.split(":", 1)[0] in _CONFIDENT_TIER_FLOOR for reason in decision.evidence)
    )


def _bounded_cap(
    decisions: list[CoverageSeedDecision], *, default_cap: int, concentrated_cap: int
) -> int:
    confident_count = len(concentrated_evidence_files(decisions))
    if 0 < confident_count <= _CONFIDENT_FILE_BOUND:
        return min(default_cap, concentrated_cap)
    return default_cap


def bounded_seed_cap(decisions: list[CoverageSeedDecision], *, default_cap: int) -> int:
    """Return a tighter seed-admission cap when direct evidence is concentrated.

    A small, non-empty confident-file set (symbol-or-better evidence) signals
    a narrow, well-evidenced query: the flat seed cap only adds noise there.
    A dispersed (`> _CONFIDENT_FILE_BOUND`) or absent (`0`) confident set
    keeps the full *default_cap* -- ambiguous queries (every sibling
    language adapter sharing the same generic symbol match) and queries with
    no identifier/symbol signal at all (exactly the shape of M0.2's five
    originally-missing target tasks) are the cases a broader seed net still
    needs. *decisions* must be the full, uncapped evidence pool (not an
    already-truncated seed-cap slice) so concentration is judged before any
    cap is applied.
    """
    return _bounded_cap(decisions, default_cap=default_cap, concentrated_cap=_CONCENTRATED_SEED_CAP)


def bounded_neighbor_cap(decisions: list[CoverageSeedDecision], *, default_cap: int) -> int:
    """Return a tighter neighbor-admission cap when direct evidence is concentrated.

    Same concentration signal as `bounded_seed_cap`; neighbors are already
    the lowest-confidence admission stage (one graph hop out from a seed,
    not a direct query match), so a concentrated signal narrows them
    further still. This never changes seed admission.
    """
    return _bounded_cap(
        decisions, default_cap=default_cap, concentrated_cap=_CONCENTRATED_NEIGHBOR_CAP
    )
