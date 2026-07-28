"""Measure prompt-prefix cache economics without changing retrieval ordering.

The harness models the input-side cache ledger from byte-identical rendered
prefixes. It does not call a hosted model, score retrieval quality, or alter
archex's retrieval path. A seeded comparator is reproducible for evidence while
representing the per-turn ordering instability that an ANN implementation can
expose in production.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from archex.reporting import count_tokens

EVIDENCE_VERSION = 1
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1
BASE_INPUT_USD_PER_MILLION = 5.0
BOOTSTRAP_CONFIDENCE = 0.95


class DeterminismEconomicsError(ValueError):
    """Raised when deterministic-economics evidence is incomplete or incoherent."""


def validate_preregistration_commit(
    repository: Path, preregistration_commit: str, source_revision: str
) -> None:
    """Require a pre-registration commit to exist before the measured source."""
    commands = (
        ("cat-file", "-e", f"{preregistration_commit}^{{commit}}"),
        ("merge-base", "--is-ancestor", preregistration_commit, source_revision),
    )
    for command in commands:
        result = subprocess.run(
            ("git", *command),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DeterminismEconomicsError(
                "S7 pre-registration commit must exist and precede the measured source revision"
            )


class OrderingArm(StrEnum):
    """The fixed, pre-registered ordering arms."""

    DETERMINISTIC = "deterministic"
    PERTURBED = "perturbed"
    ANN_BASELINE = "ann_baseline"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionTurn(_Model):
    """One user turn and its selected code-context candidates."""

    question: str = Field(min_length=1)
    contexts: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def _distinct_contexts(self) -> SessionTurn:
        if len(self.contexts) != len(set(self.contexts)):
            msg = "a turn must not repeat a context candidate"
            raise ValueError(msg)
        return self


class SessionFixture(_Model):
    """A repeated coding session belonging to one repository cluster."""

    session_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    resolved: bool
    turns: list[SessionTurn] = Field(min_length=2)


class PricingSchedule(_Model):
    """Published first-party input pricing used for the offline cost ledger."""

    model: str = "claude-opus-5"
    base_input_usd_per_million: float = BASE_INPUT_USD_PER_MILLION
    cache_write_multiplier: float = CACHE_WRITE_MULTIPLIER
    cache_read_multiplier: float = CACHE_READ_MULTIPLIER
    minimum_cacheable_tokens: Literal[512] = 512
    cache_ttl: Literal["5m"] = "5m"
    source_url: str = "https://platform.claude.com/docs/en/about-claude/pricing"
    retrieved_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _published_multipliers(self) -> PricingSchedule:
        if self.base_input_usd_per_million <= 0.0:
            raise ValueError("base input price must be positive")
        if self.cache_write_multiplier != CACHE_WRITE_MULTIPLIER:
            raise ValueError("R6 uses the published five-minute 1.25x cache-write multiplier")
        if self.cache_read_multiplier != CACHE_READ_MULTIPLIER:
            raise ValueError("R6 uses the published 0.1x cache-read multiplier")
        return self


class SessionLedger(_Model):
    """Measured cache accounting for one arm/session pair."""

    session_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    resolved: bool
    cacheable_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    uncached_input_tokens: int = Field(ge=0)
    rendered_prefixes: list[str] = Field(min_length=1)
    prefix_sha256: list[str] = Field(min_length=1)
    input_cost_usd: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _accounting_is_coherent(self) -> SessionLedger:
        if self.cache_read_tokens + self.cache_write_tokens != self.cacheable_tokens:
            raise ValueError("cacheable tokens must equal cache reads plus writes")
        if len(self.rendered_prefixes) != len(self.prefix_sha256):
            raise ValueError("each rendered prefix must have one SHA-256 identity")
        for prefix, digest in zip(self.rendered_prefixes, self.prefix_sha256, strict=True):
            if hashlib.sha256(prefix.encode()).hexdigest() != digest:
                raise ValueError("prefix SHA-256 does not match its rendered prefix")
        return self


class MetricInterval(_Model):
    """A repository-clustered percentile interval for one arm-level metric."""

    point_estimate: float
    low: float
    high: float
    resamples: int = Field(ge=20)
    seed: int
    confidence: float = BOOTSTRAP_CONFIDENCE

    @model_validator(mode="after")
    def _ordered(self) -> MetricInterval:
        if self.confidence != BOOTSTRAP_CONFIDENCE:
            raise ValueError("R6 intervals use the fixed 95% confidence level")
        if self.low > self.high:
            raise ValueError("bootstrap interval low must not exceed high")
        if not self.low <= self.point_estimate <= self.high:
            raise ValueError("point estimate must lie inside its bootstrap interval")
        return self


class ArmEvidence(_Model):
    """All per-session ledgers and pooled economics for one ordering arm."""

    arm: OrderingArm
    sessions: list[SessionLedger] = Field(min_length=1)

    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    cache_hit_rate_interval: MetricInterval
    input_cost_usd_per_resolved_task: float = Field(ge=0.0)
    input_cost_usd_per_resolved_task_interval: MetricInterval

    @model_validator(mode="after")
    def _summary_matches_ledgers(self) -> ArmEvidence:
        ids = [ledger.session_id for ledger in self.sessions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"arm {self.arm.value!r} contains duplicate sessions")
        cacheable = sum(ledger.cacheable_tokens for ledger in self.sessions)
        reads = sum(ledger.cache_read_tokens for ledger in self.sessions)
        resolved = sum(ledger.resolved for ledger in self.sessions)
        if resolved == 0:
            raise ValueError(f"arm {self.arm.value!r} has no resolved sessions")
        expected_hit_rate = reads / cacheable if cacheable else 0.0
        if abs(self.cache_hit_rate - expected_hit_rate) > 1e-12:
            raise ValueError(f"arm {self.arm.value!r} cache hit rate does not match its ledgers")
        expected_cost = sum(ledger.input_cost_usd for ledger in self.sessions) / resolved
        if abs(self.input_cost_usd_per_resolved_task - expected_cost) > 1e-12:
            raise ValueError(
                f"arm {self.arm.value!r} cost per resolved task does not match ledgers"
            )
        if abs(self.cache_hit_rate_interval.point_estimate - self.cache_hit_rate) > 1e-12:
            raise ValueError(f"arm {self.arm.value!r} cache interval does not match its summary")
        if (
            abs(
                self.input_cost_usd_per_resolved_task_interval.point_estimate
                - self.input_cost_usd_per_resolved_task
            )
            > 1e-12
        ):
            raise ValueError(f"arm {self.arm.value!r} cost interval does not match its summary")
        return self


class BootstrapInterval(_Model):
    """A repository-clustered percentile interval for one comparator."""

    comparator: OrderingArm
    point_estimate_percent: float
    low_percent: float
    high_percent: float
    resamples: int = Field(ge=20)
    seed: int
    confidence: float = BOOTSTRAP_CONFIDENCE

    @model_validator(mode="after")
    def _ordered(self) -> BootstrapInterval:
        if self.confidence != BOOTSTRAP_CONFIDENCE:
            raise ValueError("R6 intervals use the fixed 95% confidence level")
        if self.low_percent > self.high_percent:
            raise ValueError("bootstrap interval low must not exceed high")
        if not self.low_percent <= self.point_estimate_percent <= self.high_percent:
            raise ValueError("point estimate must lie inside its bootstrap interval")
        return self


class DeterminismEconomicsArtifact(_Model):
    """Validated, standalone R6 evidence artifact."""

    evidence_version: Literal[1] = EVIDENCE_VERSION
    spike_id: Literal["S7"] = "S7"
    preregistration: Literal["benchmarks/preregistrations/S7-determinism-economics.md"]
    preregistration_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    session_fixture: str = Field(min_length=1)
    session_fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measurement_command: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    tokenizer: Literal["cl100k_base"] = "cl100k_base"
    pricing: PricingSchedule
    arms: list[ArmEvidence] = Field(min_length=3)
    intervals: list[BootstrapInterval] = Field(min_length=2)

    @model_validator(mode="after")
    def _fixed_matrix(self) -> DeterminismEconomicsArtifact:
        arms = {arm.arm for arm in self.arms}
        expected = set(OrderingArm)
        if arms != expected or len(self.arms) != len(expected):
            raise ValueError(f"R6 arms must be exactly {[arm.value for arm in OrderingArm]}")
        ledgers = {arm.arm: arm.sessions for arm in self.arms}
        control = ledgers[OrderingArm.DETERMINISTIC]
        control_ids = [ledger.session_id for ledger in control]
        control_resolution = {ledger.session_id: ledger.resolved for ledger in control}
        for arm, values in ledgers.items():
            if [ledger.session_id for ledger in values] != control_ids:
                raise ValueError(f"arm {arm.value!r} does not cover the control sessions in order")
            if {ledger.session_id: ledger.resolved for ledger in values} != control_resolution:
                raise ValueError(f"arm {arm.value!r} changes fixed resolution labels")
        comparators = {interval.comparator for interval in self.intervals}
        expected_comparators = {OrderingArm.PERTURBED, OrderingArm.ANN_BASELINE}
        if comparators != expected_comparators or len(self.intervals) != len(expected_comparators):
            raise ValueError("R6 intervals must cover perturbed and ann_baseline exactly once")
        for values in ledgers.values():
            for ledger in values:
                expected_cost = _cost_usd(
                    reads=ledger.cache_read_tokens,
                    writes=ledger.cache_write_tokens,
                    uncached=ledger.uncached_input_tokens,
                    pricing=self.pricing,
                )
                if abs(ledger.input_cost_usd - expected_cost) > 1e-12:
                    raise ValueError("ledger cost does not match pricing and token accounting")
        interval_by_comparator = {interval.comparator: interval for interval in self.intervals}
        for comparator, interval in interval_by_comparator.items():
            point = relative_cost_reduction(control, ledgers[comparator])
            if abs(interval.point_estimate_percent - point) > 1e-12:
                raise ValueError("comparison interval does not match its ledgers")
        return self


def fixture_digest(sessions: Sequence[SessionFixture]) -> str:
    """Return a canonical digest for a session fixture."""
    payload = [session.model_dump(mode="json") for session in sessions]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_sessions(path: Path) -> list[SessionFixture]:
    """Load a frozen session fixture and reject malformed or duplicate sessions."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeterminismEconomicsError(f"cannot read session fixture {path}: {exc}") from exc
    try:
        sessions = [SessionFixture.model_validate(item) for item in raw]
    except (TypeError, ValidationError) as exc:
        raise DeterminismEconomicsError(f"invalid session fixture {path}: {exc}") from exc
    if len(sessions) < 2:
        raise DeterminismEconomicsError("R6 needs sessions from at least two repository clusters")
    ids = [session.session_id for session in sessions]
    if len(ids) != len(set(ids)):
        raise DeterminismEconomicsError("session fixture contains duplicate session IDs")
    if len({session.repository for session in sessions}) < 2:
        raise DeterminismEconomicsError("R6 needs at least two repository clusters")
    return sessions


def _order_contexts(
    contexts: Sequence[str],
    arm: OrderingArm,
    session_id: str,
    turn_index: int,
) -> list[str]:
    ordered = list(contexts)
    if arm is OrderingArm.DETERMINISTIC:
        return ordered
    seed_material = f"{arm.value}:{session_id}:{turn_index}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    if arm is OrderingArm.PERTURBED:
        offset = 1 + seed % (len(ordered) - 1)
        return ordered[offset:] + ordered[:offset]
    random.Random(seed).shuffle(ordered)
    if ordered == list(contexts):
        return ordered[1:] + ordered[:1]
    return ordered


def _cache_prefix(contexts: Sequence[str]) -> str:
    return "SYSTEM\nUse the selected code context.\n\nCONTEXT\n" + "\n\n".join(contexts) + "\n\n"


def _turn_suffix(question: str, history: Sequence[str]) -> str:
    return "HISTORY\n" + "\n".join(history) + "\n\nQUESTION\n" + question


def _cost_usd(*, reads: int, writes: int, uncached: int, pricing: PricingSchedule) -> float:
    weighted_tokens = (
        reads * pricing.cache_read_multiplier + writes * pricing.cache_write_multiplier + uncached
    )
    return weighted_tokens * pricing.base_input_usd_per_million / 1_000_000


def _measure_session(
    session: SessionFixture,
    arm: OrderingArm,
    pricing: PricingSchedule,
) -> SessionLedger:
    cached_prefixes: set[str] = set()
    reads = 0
    writes = 0
    uncached = 0
    rendered_prefixes: list[str] = []
    history: list[str] = []
    for index, turn in enumerate(session.turns):
        contexts = _order_contexts(turn.contexts, arm, session.session_id, index)
        prefix = _cache_prefix(contexts)
        rendered_prefixes.append(prefix)
        tokens = count_tokens(prefix)
        if tokens < pricing.minimum_cacheable_tokens:
            uncached += tokens
        elif prefix in cached_prefixes:
            reads += tokens
        else:
            writes += tokens
            cached_prefixes.add(prefix)
        suffix = _turn_suffix(turn.question, history)
        uncached += count_tokens(suffix)
        history.extend((turn.question, "assistant response"))
    return SessionLedger(
        session_id=session.session_id,
        repository=session.repository,
        resolved=session.resolved,
        cacheable_tokens=reads + writes,
        cache_read_tokens=reads,
        cache_write_tokens=writes,
        uncached_input_tokens=uncached,
        rendered_prefixes=rendered_prefixes,
        prefix_sha256=[hashlib.sha256(prefix.encode()).hexdigest() for prefix in rendered_prefixes],
        input_cost_usd=_cost_usd(
            reads=reads,
            writes=writes,
            uncached=uncached,
            pricing=pricing,
        ),
    )


def _cache_hit_rate(ledgers: Sequence[SessionLedger]) -> float:
    cacheable = sum(ledger.cacheable_tokens for ledger in ledgers)
    if cacheable == 0:
        return 0.0
    return sum(ledger.cache_read_tokens for ledger in ledgers) / cacheable


def _cost_per_resolved_task(ledgers: Sequence[SessionLedger]) -> float:
    resolved = sum(ledger.resolved for ledger in ledgers)
    if resolved == 0:
        raise DeterminismEconomicsError(
            "cannot calculate cost per resolved task without resolved sessions"
        )
    return sum(ledger.input_cost_usd for ledger in ledgers) / resolved


def _arm_metric_interval(
    ledgers: Sequence[SessionLedger],
    *,
    metric: Literal["cache_hit_rate", "input_cost_usd_per_resolved_task"],
    resamples: int,
    seed: int,
) -> MetricInterval:
    by_repo: dict[str, list[SessionLedger]] = {}
    for ledger in ledgers:
        by_repo.setdefault(ledger.repository, []).append(ledger)
    repositories = sorted(by_repo)
    if len(repositories) < 2:
        raise DeterminismEconomicsError("R6 cluster bootstrap requires at least two repositories")
    measure = _cache_hit_rate if metric == "cache_hit_rate" else _cost_per_resolved_task
    point = measure(ledgers)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        drawn = [rng.choice(repositories) for _ in repositories]
        samples.append(measure([ledger for repo in drawn for ledger in by_repo[repo]]))
    samples.sort()
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return MetricInterval(
        point_estimate=point,
        low=samples[int(tail * len(samples))],
        high=samples[max(0, int((1.0 - tail) * len(samples)) - 1)],
        resamples=resamples,
        seed=seed,
        confidence=BOOTSTRAP_CONFIDENCE,
    )


def _summarize(
    arm: OrderingArm,
    sessions: Sequence[SessionFixture],
    pricing: PricingSchedule,
    *,
    resamples: int,
    seed: int,
) -> ArmEvidence:
    ledgers = [_measure_session(session, arm, pricing) for session in sessions]
    return ArmEvidence(
        arm=arm,
        sessions=ledgers,
        cache_hit_rate=_cache_hit_rate(ledgers),
        cache_hit_rate_interval=_arm_metric_interval(
            ledgers,
            metric="cache_hit_rate",
            resamples=resamples,
            seed=seed,
        ),
        input_cost_usd_per_resolved_task=_cost_per_resolved_task(ledgers),
        input_cost_usd_per_resolved_task_interval=_arm_metric_interval(
            ledgers,
            metric="input_cost_usd_per_resolved_task",
            resamples=resamples,
            seed=seed + 1,
        ),
    )


def relative_cost_reduction(
    control: Iterable[SessionLedger],
    comparator: Iterable[SessionLedger],
) -> float:
    control_ledgers = list(control)
    comparator_ledgers = list(comparator)
    if len(control_ledgers) != len(comparator_ledgers):
        raise DeterminismEconomicsError("arms do not cover identical session IDs")
    for control_ledger, comparator_ledger in zip(control_ledgers, comparator_ledgers, strict=True):
        if (
            control_ledger.session_id != comparator_ledger.session_id
            or control_ledger.repository != comparator_ledger.repository
            or control_ledger.resolved != comparator_ledger.resolved
        ):
            raise DeterminismEconomicsError("arms do not cover identical session IDs")
    resolved = sum(ledger.resolved for ledger in control_ledgers)
    if resolved == 0:
        raise DeterminismEconomicsError(
            "cannot calculate cost per resolved task without resolved sessions"
        )
    control_cost = sum(ledger.input_cost_usd for ledger in control_ledgers) / resolved
    comparator_cost = sum(ledger.input_cost_usd for ledger in comparator_ledgers) / resolved
    if comparator_cost <= 0.0:
        raise DeterminismEconomicsError("comparator cost must be positive")
    return (comparator_cost - control_cost) / comparator_cost * 100.0


def _bootstrap(
    control: Sequence[SessionLedger],
    comparator: Sequence[SessionLedger],
    *,
    comparator_arm: OrderingArm,
    resamples: int,
    seed: int,
) -> BootstrapInterval:
    control_by_repo: dict[str, list[SessionLedger]] = {}
    comparator_by_repo: dict[str, list[SessionLedger]] = {}
    for ledger in control:
        control_by_repo.setdefault(ledger.repository, []).append(ledger)
    for ledger in comparator:
        comparator_by_repo.setdefault(ledger.repository, []).append(ledger)
    if set(control_by_repo) != set(comparator_by_repo):
        raise DeterminismEconomicsError("arms do not cover identical repository clusters")
    repositories = sorted(control_by_repo)
    if len(repositories) < 2:
        raise DeterminismEconomicsError("R6 cluster bootstrap requires at least two repositories")
    point = relative_cost_reduction(control, comparator)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        drawn = [rng.choice(repositories) for _ in repositories]
        sampled_control = [ledger for repo in drawn for ledger in control_by_repo[repo]]
        sampled_comparator = [ledger for repo in drawn for ledger in comparator_by_repo[repo]]
        samples.append(relative_cost_reduction(sampled_control, sampled_comparator))
    samples.sort()
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return BootstrapInterval(
        comparator=comparator_arm,
        point_estimate_percent=point,
        low_percent=samples[int(tail * len(samples))],
        high_percent=samples[max(0, int((1.0 - tail) * len(samples)) - 1)],
        resamples=resamples,
        seed=seed,
    )


def measure_economics(
    sessions: Sequence[SessionFixture],
    *,
    preregistration_commit: str,
    source_revision: str,
    generated_at: str,
    session_fixture: str,
    measurement_command: str,
    resamples: int = 10_000,
    seed: int = 20_260_729,
    pricing: PricingSchedule | None = None,
) -> DeterminismEconomicsArtifact:
    """Measure all three frozen arms over exactly the same session fixture."""
    if resamples < 20:
        raise DeterminismEconomicsError("R6 requires at least 20 bootstrap resamples")
    active_pricing = pricing or PricingSchedule(retrieved_at=generated_at)
    arms = [
        _summarize(arm, sessions, active_pricing, resamples=resamples, seed=seed + index * 10)
        for index, arm in enumerate(OrderingArm)
    ]
    by_arm = {arm.arm: arm for arm in arms}
    control = by_arm[OrderingArm.DETERMINISTIC].sessions
    intervals = [
        _bootstrap(
            control,
            by_arm[comparator].sessions,
            comparator_arm=comparator,
            resamples=resamples,
            seed=seed,
        )
        for comparator in (OrderingArm.PERTURBED, OrderingArm.ANN_BASELINE)
    ]
    return DeterminismEconomicsArtifact(
        preregistration="benchmarks/preregistrations/S7-determinism-economics.md",
        preregistration_commit=preregistration_commit,
        source_revision=source_revision,
        session_fixture=session_fixture,
        session_fixture_sha256=fixture_digest(sessions),
        measurement_command=measurement_command,
        generated_at=generated_at,
        pricing=active_pricing,
        arms=arms,
        intervals=intervals,
    )


def validate_determinism_economics_artifact(path: Path) -> DeterminismEconomicsArtifact:
    """Load standalone R6 evidence and reject malformed or incoherent records."""
    if not path.is_file():
        raise DeterminismEconomicsError(f"determinism-economics artifact is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeterminismEconomicsError(
            f"cannot read determinism-economics artifact {path}: {exc}"
        ) from exc
    try:
        artifact = DeterminismEconomicsArtifact.model_validate(payload)
        sessions = load_sessions(Path(artifact.session_fixture))
        validate_preregistration_commit(
            Path.cwd(), artifact.preregistration_commit, artifact.source_revision
        )
        if fixture_digest(sessions) != artifact.session_fixture_sha256:
            raise DeterminismEconomicsError(
                f"determinism-economics fixture digest does not match {artifact.session_fixture}"
            )
        expected = measure_economics(
            sessions,
            preregistration_commit=artifact.preregistration_commit,
            source_revision=artifact.source_revision,
            generated_at=artifact.generated_at,
            session_fixture=artifact.session_fixture,
            measurement_command=artifact.measurement_command,
            resamples=artifact.arms[0].cache_hit_rate_interval.resamples,
            seed=artifact.arms[0].cache_hit_rate_interval.seed,
            pricing=artifact.pricing,
        )
    except (DeterminismEconomicsError, ValidationError) as exc:
        raise DeterminismEconomicsError(
            f"determinism-economics artifact failed validation: {path}: {exc}"
        ) from exc
    if expected.model_dump(mode="json") != artifact.model_dump(mode="json"):
        raise DeterminismEconomicsError(
            "determinism-economics artifact does not reproduce from its fixture"
        )
    return artifact
