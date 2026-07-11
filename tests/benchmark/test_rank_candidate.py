"""Regression tests for the M0.3 direct-evidence file-ranking candidate."""

from __future__ import annotations

from pathlib import Path

from archex.benchmark.coverage_candidate import CoverageSeedDecision
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.rank_candidate import (
    evidence_tier_priority,
    rerank_by_direct_evidence,
)
from archex.benchmark.strategies import (
    default_strategy_registry,
    run_archex_query_coverage_candidate,
    run_archex_query_rank_candidate,
)
from archex.models import CodeChunk, ContextBundle, RankedChunk


def _chunk(file_path: str, chunk_id: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        file_path=file_path,
        start_line=1,
        end_line=2,
        content="x",
        token_count=1,
        language="python",
    )


def _bundle(*file_paths: str) -> ContextBundle:
    chunks = [
        RankedChunk(chunk=_chunk(path, str(i)), final_score=1.0)
        for i, path in enumerate(file_paths)
    ]
    return ContextBundle(
        query="q",
        chunks=chunks,
        token_count=sum(c.chunk.token_count for c in chunks),
    )


def test_evidence_tier_priority_is_one_only_for_an_identifier_match() -> None:
    assert evidence_tier_priority(("identifier:foo",)) == 1
    assert evidence_tier_priority(("symbol:foo",)) == 0
    assert evidence_tier_priority(("path:foo",)) == 0
    assert evidence_tier_priority(("lexical:foo",)) == 0
    assert evidence_tier_priority(()) == 0
    assert evidence_tier_priority(("symbol:foo", "identifier:foo")) == 1


def test_rerank_promotes_identifier_evidence_within_the_tail() -> None:
    bundle = _bundle("base.py", "decoy_tail.py", "target_tail.py")
    decisions = [
        CoverageSeedDecision(file="target_tail.py", score=24, evidence=("identifier:foo",)),
    ]

    result = rerank_by_direct_evidence(bundle, decisions, base_chunk_count=1)

    assert [c.chunk.file_path for c in result.bundle.chunks] == [
        "base.py",
        "target_tail.py",
        "decoy_tail.py",
    ]
    assert result.promoted_files == ("target_tail.py",)


def test_rerank_never_crosses_an_identifier_hit_ahead_of_the_base_region() -> None:
    # Regression guard: an earlier version of this candidate reordered the
    # *whole* bundle, including the base query's own ranked chunks. Measured
    # on the real 64-task corpus this let a tail file with coincidental
    # identifier-tier evidence (e.g. a class name that also appears in an
    # unrelated file) jump ahead of a base-query file the base query had
    # *already* ranked correctly, regressing MRR on more tasks than it
    # improved (routing_pl_scoring: `context.py`, base-ranked #1 and
    # required, got pushed to #2 by `models.py`, tail-admitted with an
    # unrelated identifier match). The base region must be frozen.
    bundle = _bundle("required_base.py", "other_base.py", "wrong_tail.py")
    decisions = [
        CoverageSeedDecision(file="wrong_tail.py", score=999, evidence=("identifier:rankedchunk",)),
    ]

    result = rerank_by_direct_evidence(bundle, decisions, base_chunk_count=2)

    assert [c.chunk.file_path for c in result.bundle.chunks] == [
        "required_base.py",
        "other_base.py",
        "wrong_tail.py",
    ]


def test_rerank_never_promotes_symbol_path_or_lexical_tier_evidence() -> None:
    # Only an explicit identifier match (a distinctive token copied verbatim
    # out of the question) is trusted to move a file ahead of another within
    # the tail. Symbol/path/lexical matches are common generic-word overlaps
    # (e.g. "middleware", "error") that regressed MRR on the real 64-task
    # corpus when they were allowed to trigger a promotion.
    bundle = _bundle("first_tail.py", "lexical_only.py", "symbol_hit.py", "path_hit.py")
    decisions = [
        CoverageSeedDecision(file="lexical_only.py", score=100, evidence=("lexical:foo",)),
        CoverageSeedDecision(file="symbol_hit.py", score=100, evidence=("symbol:foo",)),
        CoverageSeedDecision(file="path_hit.py", score=100, evidence=("path:foo",)),
    ]

    result = rerank_by_direct_evidence(bundle, decisions, base_chunk_count=0)

    assert [c.chunk.file_path for c in result.bundle.chunks] == [
        "first_tail.py",
        "lexical_only.py",
        "symbol_hit.py",
        "path_hit.py",
    ]
    assert result.promoted_files == ()


def test_rerank_preserves_relative_order_among_evidence_less_tail_files() -> None:
    # No decisions at all -- every tail file falls back to the same key, so
    # the stable sort must leave the original tail order untouched rather
    # than inventing an order from nothing.
    bundle = _bundle("base.py", "a.py", "b.py", "c.py")

    result = rerank_by_direct_evidence(bundle, [], base_chunk_count=1)

    assert [c.chunk.file_path for c in result.bundle.chunks] == ["base.py", "a.py", "b.py", "c.py"]
    assert result.promoted_files == ()


def test_rerank_never_changes_the_returned_file_set() -> None:
    bundle = _bundle("a.py", "b.py", "c.py")
    decisions = [
        CoverageSeedDecision(file="c.py", score=24, evidence=("identifier:foo",)),
        CoverageSeedDecision(file="a.py", score=1, evidence=("lexical:foo",)),
    ]

    result = rerank_by_direct_evidence(bundle, decisions, base_chunk_count=1)

    before = {c.chunk.file_path for c in bundle.chunks}
    after = {c.chunk.file_path for c in result.bundle.chunks}
    assert before == after
    assert len(result.bundle.chunks) == len(bundle.chunks)


def test_candidate_is_available_but_not_default() -> None:
    from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES

    assert (
        default_strategy_registry.get(Strategy.ARCHEX_QUERY_RANK_CANDIDATE)
        is run_archex_query_rank_candidate
    )
    assert Strategy.ARCHEX_QUERY_RANK_CANDIDATE in AVAILABLE_STRATEGIES
    assert Strategy.ARCHEX_QUERY_RANK_CANDIDATE not in DEFAULT_STRATEGIES


def test_candidate_returned_files_do_not_depend_on_expected_files(
    python_simple_repo: Path,
) -> None:
    first = run_archex_query_rank_candidate(
        BenchmarkTask(
            task_id="generic_rank_candidate",
            repo="test/python_simple",
            commit="abc",
            question="Where is the AuthService class?",
            expected_files=["models.py"],
            token_budget=1024,
        ),
        python_simple_repo,
    )
    second = run_archex_query_rank_candidate(
        BenchmarkTask(
            task_id="generic_rank_candidate",
            repo="test/python_simple",
            commit="abc",
            question="Where is the AuthService class?",
            expected_files=["utils.py"],
            token_budget=1024,
        ),
        python_simple_repo,
    )

    assert first.result_files == second.result_files
    assert first.provenance == second.provenance


def test_candidate_reorders_without_changing_the_admitted_file_set(
    python_simple_repo: Path,
) -> None:
    task = BenchmarkTask(
        task_id="rank_vs_coverage_file_set",
        repo="test/python_simple",
        commit="abc",
        question="Where does the AuthService verify a session token?",
        expected_files=["services/auth.py"],
        token_budget=1024,
    )

    coverage = run_archex_query_coverage_candidate(task, python_simple_repo)
    ranked = run_archex_query_rank_candidate(task, python_simple_repo)

    # PR-2 reuses the exact same seed/neighbor admission as the M0.2
    # candidate -- only order can change, never the returned file set.
    assert set(ranked.result_files) == set(coverage.result_files)
    assert ranked.recall == coverage.recall
    assert ranked.precision == coverage.precision
