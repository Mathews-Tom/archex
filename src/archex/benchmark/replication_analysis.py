"""Cluster-bootstrap analysis for the S0 external replication gate.

The inputs are two arms' per-task exact-match outcomes over the *same* task set,
each task belonging to exactly one repository cluster. The output is the paired
delta in percentage points, a cluster-bootstrap interval over repositories, and
the verdict implied by the pre-registered equivalence band.

The verdict is derived here rather than asserted by the caller, so a miss cannot
be written up as a hit. :mod:`archex.benchmark.replication` independently
re-derives it when the artifact is validated.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from archex.benchmark.replication import ReplicationVerdict, derive_verdict

PERCENT = 100.0


class ReplicationAnalysisError(ValueError):
    """Raised when two arms cannot be compared as a paired measurement."""


@dataclass(frozen=True, slots=True)
class ArmOutcomes:
    """One arm's per-task exact-match outcomes, keyed by task ID."""

    arm_id: str
    exact_match: Mapping[str, bool]

    def rate(self) -> float:
        if not self.exact_match:
            msg = f"arm {self.arm_id!r} has no tasks"
            raise ReplicationAnalysisError(msg)
        return sum(self.exact_match.values()) / len(self.exact_match) * PERCENT


@dataclass(frozen=True, slots=True)
class ClusterBootstrapResult:
    """The paired delta with its cluster-bootstrap interval."""

    control_rate: float
    treatment_rate: float
    delta: float
    ci_low: float
    ci_high: float
    resamples: int
    seed: int
    clusters: tuple[str, ...]
    per_cluster_delta: Mapping[str, float]

    def verdict(self, band_low: float, band_high: float) -> ReplicationVerdict:
        """Derive the pre-registered verdict. Inside the band is not enough.

        Delegates to :func:`archex.benchmark.replication.derive_verdict`, the one
        place the rule lives, so this can never disagree with what the schema
        re-derives when the artifact is validated.
        """
        return derive_verdict(
            delta=self.delta,
            band_low=band_low,
            band_high=band_high,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
        )


def _cluster_of(task_id: str) -> str:
    """Repository cluster for a RepoEval task ID of the form ``<repo>/<index>``."""
    repo, _, remainder = task_id.partition("/")
    if not repo or not remainder:
        msg = f"task ID {task_id!r} does not carry a repository cluster"
        raise ReplicationAnalysisError(msg)
    return repo


def group_by_cluster(task_ids: Sequence[str]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    for task_id in task_ids:
        clusters.setdefault(_cluster_of(task_id), []).append(task_id)
    return clusters


def cluster_bootstrap(
    control: ArmOutcomes,
    treatment: ArmOutcomes,
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> ClusterBootstrapResult:
    """Compare two arms on an identical task set, resampling repositories.

    Both arms must cover exactly the same tasks. A partial arm is an error
    rather than a silently smaller comparison, because dropping the tasks one
    arm failed on would bias the delta toward whichever arm crashed less.
    """
    if resamples < 1:
        msg = f"resamples must be positive, got {resamples}"
        raise ReplicationAnalysisError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie in (0, 1), got {confidence}"
        raise ReplicationAnalysisError(msg)

    control_ids = set(control.exact_match)
    treatment_ids = set(treatment.exact_match)
    if control_ids != treatment_ids:
        missing = sorted(control_ids - treatment_ids)[:5]
        unexpected = sorted(treatment_ids - control_ids)[:5]
        msg = (
            f"arms cover different task sets: {len(control_ids)} vs {len(treatment_ids)}; "
            f"absent from {treatment.arm_id!r}={missing}, "
            f"absent from {control.arm_id!r}={unexpected}"
        )
        raise ReplicationAnalysisError(msg)

    clusters = group_by_cluster(sorted(control_ids))
    cluster_names = sorted(clusters)
    if len(cluster_names) < 2:
        msg = f"a cluster bootstrap needs at least two clusters, got {cluster_names}"
        raise ReplicationAnalysisError(msg)

    def delta_over(task_ids: Sequence[str]) -> float:
        total = len(task_ids)
        control_hits = sum(control.exact_match[task_id] for task_id in task_ids)
        treatment_hits = sum(treatment.exact_match[task_id] for task_id in task_ids)
        return (treatment_hits - control_hits) / total * PERCENT

    all_ids = [task_id for name in cluster_names for task_id in clusters[name]]
    point_delta = delta_over(all_ids)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        drawn = [rng.choice(cluster_names) for _ in cluster_names]
        resampled = [task_id for name in drawn for task_id in clusters[name]]
        draws.append(delta_over(resampled))
    draws.sort()

    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(draws) - 1, int(tail * len(draws))))
    high_index = max(0, min(len(draws) - 1, int((1.0 - tail) * len(draws)) - 1))

    return ClusterBootstrapResult(
        control_rate=control.rate(),
        treatment_rate=treatment.rate(),
        delta=point_delta,
        ci_low=draws[low_index],
        ci_high=draws[high_index],
        resamples=resamples,
        seed=seed,
        clusters=tuple(cluster_names),
        per_cluster_delta={name: delta_over(clusters[name]) for name in cluster_names},
    )
