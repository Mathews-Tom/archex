"""Tests for the S0 cluster-bootstrap replication analysis."""

from __future__ import annotations

import pytest

from archex.benchmark.replication import ReplicationVerdict, derive_verdict
from archex.benchmark.replication_analysis import (
    ArmOutcomes,
    ClusterBootstrapResult,
    ReplicationAnalysisError,
    cluster_bootstrap,
    group_by_cluster,
)

REPOS = ("repo_a", "repo_b", "repo_c", "repo_d")


def _arm(arm_id: str, hits_per_repo: dict[str, int], *, per_repo: int = 25) -> ArmOutcomes:
    outcomes: dict[str, bool] = {}
    for repo in REPOS:
        hits = hits_per_repo[repo]
        for index in range(per_repo):
            outcomes[f"{repo}/{index}"] = index < hits
    return ArmOutcomes(arm_id=arm_id, exact_match=outcomes)


def test_cluster_grouping_uses_the_repository_prefix() -> None:
    clusters = group_by_cluster(["repo_a/1", "repo_a/2", "repo_b/1"])
    assert clusters == {"repo_a": ["repo_a/1", "repo_a/2"], "repo_b": ["repo_b/1"]}


def test_task_id_without_a_cluster_is_rejected() -> None:
    with pytest.raises(ReplicationAnalysisError, match="repository cluster"):
        group_by_cluster(["ungrouped"])


def test_rates_and_delta_are_percentage_points() -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    treatment = _arm("treatment", dict.fromkeys(REPOS, 10))
    result = cluster_bootstrap(control, treatment, resamples=200, seed=7)
    assert result.control_rate == pytest.approx(20.0)  # pyright: ignore[reportUnknownMemberType]
    assert result.treatment_rate == pytest.approx(40.0)  # pyright: ignore[reportUnknownMemberType]
    assert result.delta == pytest.approx(20.0)  # pyright: ignore[reportUnknownMemberType]


def test_mismatched_task_sets_are_rejected() -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    partial = dict(control.exact_match)
    partial.pop("repo_a/0")
    with pytest.raises(ReplicationAnalysisError, match="different task sets"):
        cluster_bootstrap(control, ArmOutcomes("treatment", partial), resamples=10, seed=1)


def test_a_single_cluster_cannot_be_bootstrapped() -> None:
    outcomes = {f"solo/{i}": i < 3 for i in range(10)}
    with pytest.raises(ReplicationAnalysisError, match="at least two clusters"):
        cluster_bootstrap(
            ArmOutcomes("control", outcomes),
            ArmOutcomes("treatment", outcomes),
            resamples=10,
            seed=1,
        )


def test_bootstrap_is_deterministic_under_a_fixed_seed() -> None:
    control = _arm("control", {"repo_a": 5, "repo_b": 7, "repo_c": 4, "repo_d": 9})
    treatment = _arm("treatment", {"repo_a": 9, "repo_b": 12, "repo_c": 5, "repo_d": 14})
    first = cluster_bootstrap(control, treatment, resamples=500, seed=20260727)
    second = cluster_bootstrap(control, treatment, resamples=500, seed=20260727)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_the_point_delta_does_not_depend_on_the_seed() -> None:
    """Only the interval is resampled; the reported delta is the observed one."""
    control = _arm("control", {"repo_a": 5, "repo_b": 7, "repo_c": 4, "repo_d": 9})
    treatment = _arm("treatment", {"repo_a": 9, "repo_b": 12, "repo_c": 5, "repo_d": 14})
    first = cluster_bootstrap(control, treatment, resamples=500, seed=1)
    second = cluster_bootstrap(control, treatment, resamples=500, seed=2)
    assert first.delta == pytest.approx(second.delta)  # pyright: ignore[reportUnknownMemberType]
    assert (first.seed, first.resamples) == (1, 500)
    assert first.clusters == ("repo_a", "repo_b", "repo_c", "repo_d")


def test_a_consistent_effect_yields_an_interval_clear_of_zero() -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    treatment = _arm("treatment", dict.fromkeys(REPOS, 10))
    result = cluster_bootstrap(control, treatment, resamples=2000, seed=20260727)
    assert result.ci_low > 0.0


def test_an_effect_carried_by_one_repository_does_not_clear_zero() -> None:
    """One repository doing all the work is what clustering is meant to expose."""
    control = _arm("control", dict.fromkeys(REPOS, 5))
    treatment = _arm("treatment", {"repo_a": 25, "repo_b": 5, "repo_c": 5, "repo_d": 5})
    result = cluster_bootstrap(control, treatment, resamples=2000, seed=20260727)
    assert result.delta > 0.0
    assert result.ci_low == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]
    assert result.verdict(-100.0, 100.0) is ReplicationVerdict.INCONCLUSIVE


def test_verdict_outside_the_band_is_a_fail_however_tight_the_interval() -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    treatment = _arm("treatment", dict.fromkeys(REPOS, 10))
    result = cluster_bootstrap(control, treatment, resamples=2000, seed=20260727)
    assert result.ci_low > 0.0
    assert result.verdict(2.88, 6.88) is ReplicationVerdict.FAIL


def test_verdict_inside_the_band_with_a_clear_interval_is_a_pass() -> None:
    control = _arm("control", dict.fromkeys(REPOS, 40), per_repo=400)
    treatment = _arm("treatment", dict.fromkeys(REPOS, 60), per_repo=400)
    result = cluster_bootstrap(control, treatment, resamples=2000, seed=20260727)
    assert result.delta == pytest.approx(5.0)  # pyright: ignore[reportUnknownMemberType]
    assert result.verdict(2.88, 6.88) is ReplicationVerdict.PASS


def test_per_cluster_deltas_are_reported() -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    treatment = _arm("treatment", {"repo_a": 25, "repo_b": 5, "repo_c": 5, "repo_d": 5})
    result = cluster_bootstrap(control, treatment, resamples=100, seed=1)
    assert result.per_cluster_delta["repo_a"] == pytest.approx(80.0)  # pyright: ignore[reportUnknownMemberType]
    assert result.per_cluster_delta["repo_b"] == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]


def test_empty_arm_is_rejected() -> None:
    with pytest.raises(ReplicationAnalysisError, match="no tasks"):
        ArmOutcomes("empty", {}).rate()


@pytest.mark.parametrize("resamples", [0, -1])
def test_nonpositive_resamples_are_rejected(resamples: int) -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    with pytest.raises(ReplicationAnalysisError, match="resamples must be positive"):
        cluster_bootstrap(control, control, resamples=resamples, seed=1)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_confidence_outside_the_unit_interval_is_rejected(confidence: float) -> None:
    control = _arm("control", dict.fromkeys(REPOS, 5))
    with pytest.raises(ReplicationAnalysisError, match="confidence must lie"):
        cluster_bootstrap(control, control, resamples=10, seed=1, confidence=confidence)


@pytest.mark.parametrize(
    ("delta", "ci_low", "ci_high", "expected"),
    [
        (2.88, 1.0, 4.0, ReplicationVerdict.PASS),
        (6.88, 1.0, 4.0, ReplicationVerdict.PASS),
        (2.8799, 1.0, 4.0, ReplicationVerdict.FAIL),
        (6.8801, 1.0, 4.0, ReplicationVerdict.FAIL),
        (4.0, 0.0, 4.0, ReplicationVerdict.INCONCLUSIVE),
        (4.0, -4.0, 0.0, ReplicationVerdict.INCONCLUSIVE),
        (4.0, -4.0, -0.0, ReplicationVerdict.INCONCLUSIVE),
        (4.0, 0.0001, 4.0, ReplicationVerdict.PASS),
    ],
)
def test_band_is_closed_and_the_zero_test_is_open(
    delta: float, ci_low: float, ci_high: float, expected: ReplicationVerdict
) -> None:
    """The gate turned on 0.0675 points, so both boundaries are pinned exactly."""
    result = ClusterBootstrapResult(
        control_rate=0.0,
        treatment_rate=delta,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        resamples=1,
        seed=0,
        clusters=("a", "b"),
        per_cluster_delta={},
    )
    assert result.verdict(2.88, 6.88) is expected


def test_the_analysis_and_the_schema_share_one_verdict_rule() -> None:
    """Two copies of the rule could drift; one function cannot."""
    for delta in (-5.0, -0.0, 0.0, 2.8799, 2.88, 4.88, 6.88, 6.8801, 9.9):
        for ci_low, ci_high in ((-5.0, -0.0), (0.0, 5.0), (-1.0, 1.0), (0.5, 5.0)):
            result = ClusterBootstrapResult(
                control_rate=0.0,
                treatment_rate=delta,
                delta=delta,
                ci_low=ci_low,
                ci_high=ci_high,
                resamples=1,
                seed=0,
                clusters=("a", "b"),
                per_cluster_delta={},
            )
            assert result.verdict(2.88, 6.88) is derive_verdict(
                delta=delta, band_low=2.88, band_high=6.88, ci_low=ci_low, ci_high=ci_high
            )
