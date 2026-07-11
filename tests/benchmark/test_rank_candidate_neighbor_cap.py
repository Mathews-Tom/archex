"""Regression tests for the M0.3 localization-aware seed/neighbor-cap bounds."""

from __future__ import annotations

from pathlib import Path

from archex.benchmark.coverage_candidate import CoverageSeedDecision
from archex.benchmark.models import BenchmarkTask
from archex.benchmark.rank_candidate import (
    bounded_neighbor_cap,
    bounded_seed_cap,
    concentrated_evidence_files,
)
from archex.benchmark.strategies import (
    run_archex_query_coverage_candidate,
    run_archex_query_rank_candidate,
)


def _decision(file: str, evidence: tuple[str, ...]) -> CoverageSeedDecision:
    return CoverageSeedDecision(file=file, score=1, evidence=evidence)


def test_concentrated_evidence_files_requires_symbol_or_identifier_tier() -> None:
    decisions = [
        _decision("symbol.py", ("symbol:foo",)),
        _decision("identifier.py", ("identifier:foo",)),
        _decision("path.py", ("path:foo",)),
        _decision("lexical.py", ("lexical:foo",)),
    ]

    result = concentrated_evidence_files(decisions)

    assert result == {"symbol.py", "identifier.py"}


def test_bounded_neighbor_cap_narrows_for_a_small_confident_set() -> None:
    decisions = [_decision(f"f{i}.py", ("symbol:foo",)) for i in range(3)]

    assert bounded_neighbor_cap(decisions, default_cap=24) < 24


def test_bounded_neighbor_cap_keeps_default_when_confident_set_is_dispersed() -> None:
    # More than the concentration bound -- this is the "every sibling
    # language adapter shares the same generic symbol match" ambiguity case:
    # narrowing further would be a guess, not an evidence-backed decision.
    decisions = [_decision(f"f{i}.py", ("symbol:foo",)) for i in range(20)]

    assert bounded_neighbor_cap(decisions, default_cap=24) == 24


def test_bounded_neighbor_cap_keeps_default_when_no_confident_evidence_exists() -> None:
    # Zero symbol/identifier-tier files -- a query with no strong lexical
    # overlap at all is exactly the case M0.2's five originally-missing
    # target tasks needed the full net for; never narrow here.
    decisions = [_decision("f.py", ("lexical:foo",)), _decision("g.py", ("path:foo",))]

    assert bounded_neighbor_cap(decisions, default_cap=24) == 24


def test_bounded_neighbor_cap_never_exceeds_the_default_cap() -> None:
    # A default_cap smaller than the concentrated cap must not be widened.
    decisions = [_decision("f.py", ("symbol:foo",))]

    assert bounded_neighbor_cap(decisions, default_cap=3) == 3


def test_bounded_seed_cap_narrows_for_a_small_confident_set() -> None:
    decisions = [_decision(f"f{i}.py", ("symbol:foo",)) for i in range(3)]

    assert bounded_seed_cap(decisions, default_cap=32) < 32


def test_bounded_seed_cap_keeps_default_when_confident_set_is_dispersed() -> None:
    decisions = [_decision(f"f{i}.py", ("symbol:foo",)) for i in range(20)]

    assert bounded_seed_cap(decisions, default_cap=32) == 32


def test_bounded_seed_cap_keeps_default_when_no_confident_evidence_exists() -> None:
    decisions = [_decision("f.py", ("lexical:foo",)), _decision("g.py", ("path:foo",))]

    assert bounded_seed_cap(decisions, default_cap=32) == 32


def test_bounded_seed_cap_never_exceeds_the_default_cap() -> None:
    decisions = [_decision("f.py", ("symbol:foo",))]

    assert bounded_seed_cap(decisions, default_cap=3) == 3


def test_bounded_seed_cap_stays_well_above_measured_required_file_positions() -> None:
    # Regression guard: swept against the real 64-task corpus, the tightest
    # safe seed cap for every concentrated task was 12 (the closest calls,
    # `django_middleware` and `loc_django_username_validator`, admit their
    # required file at seed position 10-11). This asserts the deployed
    # constant keeps a safety margin above that measured floor rather than
    # silently drifting back down to an unsafe value.
    decisions = [_decision(f"f{i}.py", ("symbol:foo",)) for i in range(3)]

    assert bounded_seed_cap(decisions, default_cap=32) >= 16


def test_candidate_neighbor_admission_never_exceeds_the_coverage_candidate(
    python_simple_repo: Path,
) -> None:
    task = BenchmarkTask(
        task_id="rank_neighbor_cap_bound",
        repo="test/python_simple",
        commit="abc",
        question="Where does the AuthService verify a session token?",
        expected_files=["services/auth.py"],
        token_budget=1024,
    )

    coverage = run_archex_query_coverage_candidate(task, python_simple_repo)
    ranked = run_archex_query_rank_candidate(task, python_simple_repo)

    # The seed/neighbor caps can only narrow admission (never widen it
    # beyond what M0.2's candidate already admits), so the rank candidate's
    # file set is always a subset of the coverage candidate's on the same
    # task.
    assert set(ranked.result_files) <= set(coverage.result_files)
    assert ranked.recall <= coverage.recall + 1e-9
