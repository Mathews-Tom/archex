"""Measure what the benchmark corpus is capable of detecting.

This module measures the corpus; it never changes it. Nothing here fixes a
defect, deletes a task, or touches retrieval. Every function is deterministic
given the same task files, so the audit reruns to the same figures.

Three questions, in order of how badly a wrong answer misleads everything else:

* **Leakage** -- does a task hand the answer to the retriever in its own
  question or keywords? A gold symbol or path quoted verbatim in the query makes
  that task measure string matching rather than retrieval.
* **Clustering** -- tasks drawn from one repository share a codebase, an API
  surface, and a style, so they are not independent observations. The number of
  *independent* clusters, not the number of tasks, is what inference can spend.
* **Resolution** -- given the measured cluster structure, how large must an
  effect be before this corpus can distinguish it from zero?
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkTask

PERCENT = 100.0
SELF_REPO = "."
#: Fragments too generic to count as a leak on their own. A task asking about
#: "the parser" has not leaked `parser.py`; one naming `AdapterRegistry` has.
_GENERIC_TOKENS = frozenset(
    {
        "api",
        "app",
        "base",
        "cache",
        "client",
        "config",
        "core",
        "data",
        "error",
        "get",
        "index",
        "init",
        "main",
        "model",
        "models",
        "parser",
        "query",
        "run",
        "server",
        "set",
        "src",
        "test",
        "tests",
        "type",
        "types",
        "utils",
    }
)
_MIN_TOKEN_LENGTH = 4
_IDENTIFIER_SHAPED = re.compile(r"[A-Z]|_")
_MIN_WORD_SYMBOL_LENGTH = 10


class CorpusAuditError(ValueError):
    """Raised when the corpus cannot be audited as given."""


@dataclass(frozen=True, slots=True)
class LeakSignal:
    """One gold identifier found verbatim in a task's own query surface.

    ``kind`` separates two strengths of evidence, and they are never pooled into
    a single headline. A ``symbol`` match means the query quotes a gold symbol
    name, which the retriever can match lexically without understanding
    anything. A ``path_stem`` match means the query quotes a gold file's stem;
    that is weaker, because a question about adapters legitimately says
    "adapter" whether or not `adapters.py` is gold.
    """

    task_id: str
    kind: str
    value: str
    surface: str


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Per-task leakage over the whole corpus."""

    total_tasks: int
    leaked_task_ids: tuple[str, ...]
    symbol_leaked_task_ids: tuple[str, ...]
    signals: tuple[LeakSignal, ...]
    by_kind: dict[str, int] = field(default_factory=lambda: {})
    by_family: dict[str, int] = field(default_factory=lambda: {})

    def _rate(self, count: int) -> float:
        if not self.total_tasks:
            msg = "cannot compute a leak rate over zero tasks"
            raise CorpusAuditError(msg)
        return count / self.total_tasks

    @property
    def leak_rate(self) -> float:
        """Any-evidence rate. Reported with the strong rate, never instead of it."""
        return self._rate(len(self.leaked_task_ids))

    @property
    def symbol_leak_rate(self) -> float:
        """The defensible headline: a gold symbol quoted verbatim in the query."""
        return self._rate(len(self.symbol_leaked_task_ids))


@dataclass(frozen=True, slots=True)
class ClusterReport:
    """The corpus viewed as clusters rather than as independent tasks."""

    total_tasks: int
    cluster_sizes: dict[str, int]
    largest_cluster: str
    self_repo_tasks: int

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_sizes)

    @property
    def largest_cluster_share(self) -> float:
        return self.cluster_sizes[self.largest_cluster] / self.total_tasks

    @property
    def self_repo_share(self) -> float:
        return self.self_repo_tasks / self.total_tasks

    @property
    def weighted_mean_cluster_size(self) -> float:
        """``sum(m^2) / sum(m)``, the cluster size the design effect depends on.

        Not the arithmetic mean. With unequal clusters the design effect is
        driven by the *size-weighted* mean, because a large cluster contributes
        its correlated observations in proportion to its own size. This corpus
        makes the difference stark: sizes of 24 and fifteen clusters of 2 to 4
        give an arithmetic mean of 4.0 but a weighted mean of 10.8, so using the
        arithmetic mean would overstate the usable sample size by about twofold.
        """
        return sum(size * size for size in self.cluster_sizes.values()) / self.total_tasks

    def effective_sample_size(self, icc: float) -> float:
        """Design-effect-corrected N.

        With clustered observations the usable sample size is
        ``N / (1 + (m_A - 1) * ICC)`` where ``m_A`` is the size-weighted mean
        cluster size. At ICC 0 the clusters carry no shared signal and N is
        unchanged; at ICC 1 each cluster contributes about one observation.
        """
        if not 0.0 <= icc <= 1.0:
            msg = f"icc must lie in [0, 1], got {icc}"
            raise CorpusAuditError(msg)
        design_effect = 1.0 + (self.weighted_mean_cluster_size - 1.0) * icc
        return self.total_tasks / design_effect


@dataclass(frozen=True, slots=True)
class HeldOutReport:
    """Whether the declared held-out set is actually held out."""

    declared: tuple[str, ...]
    also_in_task_corpus: tuple[str, ...]
    enforced_by_code: bool

    @property
    def leak_rate(self) -> float:
        if not self.declared:
            msg = "cannot compute a held-out leak rate over an empty declaration"
            raise CorpusAuditError(msg)
        return len(self.also_in_task_corpus) / len(self.declared)


def _normalise(text: str) -> str:
    """Collapse punctuation to spaces, for matching prose against prose."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def _raw(text: str) -> str:
    """Lowercase while preserving `_` and `.`, for matching identifiers literally.

    Both surfaces are needed. Matching only the normalised form lets `_merge`
    match the ordinary word "merge" and `block_on` match "block on", which
    readmits the false-positive class the tiering exists to keep out. Matching
    only the raw form loses nothing real, so the raw form is what an identifier
    is tested against.
    """
    return re.sub(r"[^a-z0-9_.]+", " ", text.lower())


def _stem_of_path(path: str, *, exclude: frozenset[str]) -> str | None:
    """The gold file's stem, if it is distinctive enough to count as evidence.

    Only the stem is considered. Directory components were tried first and are
    deliberately excluded: on a self-repo task every gold path begins with the
    project's own name, so counting directories reported an 89% leak rate driven
    entirely by questions about archex containing the word "archex". That is a
    property of the detector, not of the corpus.
    """
    stem = PurePosixPath(path).stem
    if stem.lower() in exclude or not _is_distinctive(stem):
        return None
    return stem


def _repo_tokens(repo: str) -> frozenset[str]:
    """Tokens naming the repository itself, which carry no information."""
    if repo == SELF_REPO:
        return frozenset({"archex"})
    return frozenset(part.lower() for part in re.split(r"[/_-]", repo) if part)


def _symbol_kind(symbol: str) -> str:  # noqa: D401
    """Separate identifier-shaped symbols from gold symbols that are plain words.

    `PythonAdapter` and `default_adapter_registry` cannot appear in a question by
    coincidence. `retry`, `filter`, `next`, and `abort` are gold symbol names in
    this corpus *and* the ordinary English word for what the task asks about, so
    counting them as leakage would inflate the headline with the detector's own
    ambiguity. They are reported as `symbol_word` and excluded from the strong
    tier without being hidden.
    """
    if symbol.lower() in _GENERIC_TOKENS:
        # Generic words are reported in the weak tier rather than dropped, so the
        # count stays auditable. `_GENERIC_TOKENS` still drops path stems, where
        # a generic word carries no information at all.
        return "symbol_word"
    if _IDENTIFIER_SHAPED.search(symbol) or len(symbol) >= _MIN_WORD_SYMBOL_LENGTH:
        return "symbol"
    return "symbol_word"


def _is_distinctive(token: str) -> bool:
    return (
        len(token) >= _MIN_TOKEN_LENGTH
        and token.lower() not in _GENERIC_TOKENS
        and not token.startswith(".")
    )


def _contains_token(haystack: str, token: str) -> bool:
    """Whole-token containment of *token* in an already-lowercased *haystack*.

    Boundaries are checked against alphanumerics, underscore, and dot, so
    `merge` does not match inside `_merge` and `parse` does not match `parsed`.
    """
    needle = token.lower().strip()
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9_.]){re.escape(needle)}(?![a-z0-9_.])"
    return re.search(pattern, haystack) is not None


def score_task_leakage(task: BenchmarkTask) -> tuple[LeakSignal, ...]:
    """Find gold identifiers quoted verbatim in a task's question or keywords.

    Both surfaces are checked because either is visible to the retriever. A
    symbol match is scored on the raw symbol; a path match is scored on its
    distinctive fragments, since no question quotes a full path.
    """
    keywords = " ".join(task.keywords)
    # Identifiers are matched literally against a surface that keeps `_` and `.`;
    # path stems are prose-like and matched against the normalised surface.
    raw_surfaces = {"question": _raw(task.question), "keywords": _raw(keywords)}
    prose_surfaces = {
        "question": _normalise(task.question),
        "keywords": _normalise(keywords),
    }
    exclude = _repo_tokens(task.repo)
    signals: list[LeakSignal] = []
    for symbol in task.expected_symbols:
        if symbol.lower() in exclude or len(symbol) < _MIN_TOKEN_LENGTH:
            continue
        kind = _symbol_kind(symbol)
        for surface, text in raw_surfaces.items():
            if _contains_token(text, symbol):
                signals.append(
                    LeakSignal(task_id=task.task_id, kind=kind, value=symbol, surface=surface)
                )
    seen_stems: set[tuple[str, str]] = set()
    for path in task.expected_files:
        stem = _stem_of_path(path, exclude=exclude)
        if stem is None:
            continue
        for surface, text in prose_surfaces.items():
            if (stem, surface) in seen_stems:
                continue
            if _contains_token(text, stem):
                seen_stems.add((stem, surface))
                signals.append(
                    LeakSignal(task_id=task.task_id, kind="path_stem", value=stem, surface=surface)
                )
    return tuple(signals)


def score_corpus_leakage(tasks: Sequence[BenchmarkTask]) -> LeakageReport:
    if not tasks:
        msg = "cannot audit an empty corpus"
        raise CorpusAuditError(msg)
    signals: list[LeakSignal] = []
    leaked: list[str] = []
    symbol_leaked: list[str] = []
    by_kind: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for task in sorted(tasks, key=lambda item: item.task_id):
        task_signals = score_task_leakage(task)
        if not task_signals:
            continue
        leaked.append(task.task_id)
        if any(signal.kind == "symbol" for signal in task_signals):
            symbol_leaked.append(task.task_id)
        signals.extend(task_signals)
        family = task.family.value
        by_family[family] = by_family.get(family, 0) + 1
        for signal in task_signals:
            by_kind[signal.kind] = by_kind.get(signal.kind, 0) + 1
    return LeakageReport(
        total_tasks=len(tasks),
        leaked_task_ids=tuple(leaked),
        symbol_leaked_task_ids=tuple(symbol_leaked),
        signals=tuple(signals),
        by_kind=by_kind,
        by_family=by_family,
    )


def describe_clusters(tasks: Sequence[BenchmarkTask]) -> ClusterReport:
    """Cluster the corpus by repository, which is the unit inference resamples."""
    if not tasks:
        msg = "cannot audit an empty corpus"
        raise CorpusAuditError(msg)
    sizes: dict[str, int] = {}
    for task in tasks:
        sizes[task.repo] = sizes.get(task.repo, 0) + 1
    largest = max(sorted(sizes), key=lambda repo: sizes[repo])
    return ClusterReport(
        total_tasks=len(tasks),
        cluster_sizes=dict(sorted(sizes.items())),
        largest_cluster=largest,
        self_repo_tasks=sizes.get(SELF_REPO, 0),
    )


def audit_held_out(
    declared: Sequence[str],
    tasks: Sequence[BenchmarkTask],
    *,
    enforced_by_code: bool,
) -> HeldOutReport:
    """Check a declared held-out set against the corpus it is held out from.

    ``enforced_by_code`` is supplied by the caller rather than inferred: whether
    any code path excludes these IDs from a run is a fact about the repository,
    not about the task files, and guessing it would be worse than passing it in.
    """
    if not declared:
        msg = "held-out declaration is empty"
        raise CorpusAuditError(msg)
    task_ids = {task.task_id for task in tasks}
    return HeldOutReport(
        declared=tuple(declared),
        also_in_task_corpus=tuple(sorted(set(declared) & task_ids)),
        enforced_by_code=enforced_by_code,
    )


@dataclass(frozen=True, slots=True)
class PowerResult:
    """Power and interval width for one true effect on one cluster structure."""

    effect_points: float
    power: float
    mean_ci_width: float
    simulations: int
    seed: int

    @property
    def monte_carlo_se(self) -> float:
        """Standard error of the power estimate itself.

        Reported because a power estimate read against a 0.80 target is only
        meaningful if its own noise is small against the distance to that target.
        """
        return math.sqrt(max(0.0, self.power * (1.0 - self.power)) / self.simulations)

    def clears(self, target_power: float) -> bool:
        """Whether power clears *target* by more than two of its own standard errors."""
        return self.power - 2.0 * self.monte_carlo_se >= target_power


def _cluster_probabilities(
    sizes: Sequence[int], base_rate: float, cluster_sd: float, rng: random.Random
) -> list[float]:
    """Per-cluster success rates, dispersed to induce between-cluster variance."""
    return [min(0.99, max(0.01, rng.gauss(base_rate, cluster_sd))) for _ in sizes]


#: Below this many clusters the percentile bootstrap cannot approximate 95%
#: coverage: with k clusters there are only k**k distinct resamples, and the
#: interval collapses, reporting *higher* power for a worse design. Projections
#: below this threshold are refused rather than quietly reported.
MIN_VALID_CLUSTERS = 8


def simulate_power(
    cluster_sizes: Sequence[int],
    *,
    effect_points: float,
    base_rate: float,
    cluster_sd: float,
    simulations: int,
    resamples: int,
    seed: int,
    effect_sd: float = 0.0,
) -> PowerResult:
    """Estimate power to detect *effect_points* on a given cluster structure.

    Generative model, stated rather than hidden. Each cluster draws a success
    rate from ``Normal(base_rate, cluster_sd)``, which is what makes observations
    within a repository correlated. Every task in that cluster then draws a
    control outcome at the cluster rate and a treatment outcome at the cluster
    rate plus that cluster's effect. The paired delta is the difference of the two
    pooled means, and inference is the same cluster bootstrap over repositories
    that R3 used, so a power figure here is comparable to an interval measured
    there.

    ``effect_sd`` is the **between-cluster standard deviation of the treatment
    effect**, and it is the only parameter under which the number of
    repositories -- as opposed to the number of tasks -- limits power. At
    ``effect_sd=0`` every repository responds identically, the shared cluster
    rate cancels in the paired delta, and power is governed by total task count.
    Do not set it above what data supports: estimated from R3's eight measured
    per-repository deltas it is indistinguishable from zero, because their spread
    is smaller than the within-repository sampling noise at 200 tasks each.

    Power is the fraction of simulations whose 95% cluster-bootstrap interval
    excludes zero -- the same rule the pre-registered Gate A decision applied.
    """
    if not cluster_sizes:
        msg = "cannot simulate power over zero clusters"
        raise CorpusAuditError(msg)
    if len(cluster_sizes) < MIN_VALID_CLUSTERS:
        msg = (
            f"a percentile cluster bootstrap cannot hold 95% coverage with "
            f"{len(cluster_sizes)} clusters; at least {MIN_VALID_CLUSTERS} are required, "
            "because below that the interval narrows artificially and reports higher "
            "power for a worse design"
        )
        raise CorpusAuditError(msg)
    if effect_sd < 0.0:
        msg = f"effect_sd must be non-negative, got {effect_sd}"
        raise CorpusAuditError(msg)
    if simulations < 1 or resamples < 1:
        msg = f"simulations and resamples must be positive, got {simulations} and {resamples}"
        raise CorpusAuditError(msg)

    if base_rate + effect_points / PERCENT > 1.0:
        msg = (
            f"base rate {base_rate} plus effect {effect_points} points saturates the "
            "treatment arm; every task would succeed and the result is not a power "
            "calculation"
        )
        raise CorpusAuditError(msg)

    # Descending, so the estimator does not depend on the alphabetical order of
    # repository names, which is what `describe_clusters` happens to return.
    ordered = tuple(sorted(cluster_sizes, reverse=True))
    effect = effect_points / PERCENT
    detected = 0
    widths: list[float] = []

    for index in range(simulations):
        # One stream per simulation, so `simulations` and `resamples` are
        # independent knobs. Sharing a single interleaved stream made a +/-1
        # change to `resamples` reshuffle every later simulation.
        rng = random.Random(f"{seed}:{index}")  # noqa: S311
        rates = _cluster_probabilities(ordered, base_rate, cluster_sd, rng)
        per_cluster: list[tuple[int, int, int]] = []
        for size, rate in zip(ordered, rates, strict=True):
            cluster_effect = effect if effect_sd == 0.0 else rng.gauss(effect, effect_sd / PERCENT)
            treated_rate = min(1.0, max(0.0, rate + cluster_effect))
            control_hits = sum(1 for _ in range(size) if rng.random() < rate)
            treatment_hits = sum(1 for _ in range(size) if rng.random() < treated_rate)
            per_cluster.append((size, control_hits, treatment_hits))

        draws: list[float] = []
        for _ in range(resamples):
            drawn = [rng.choice(per_cluster) for _ in per_cluster]
            total = sum(size for size, _, _ in drawn)
            delta = sum(t - c for _, c, t in drawn) / total * PERCENT
            draws.append(delta)
        draws.sort()
        low = draws[max(0, int(0.025 * len(draws)))]
        high = draws[min(len(draws) - 1, int(0.975 * len(draws)) - 1)]
        widths.append(high - low)
        if low > 0.0 or high < 0.0:
            detected += 1

    return PowerResult(
        effect_points=effect_points,
        power=detected / simulations,
        mean_ci_width=sum(widths) / len(widths),
        simulations=simulations,
        seed=seed,
    )


def minimum_detectable_effect(
    cluster_sizes: Sequence[int],
    *,
    base_rate: float,
    cluster_sd: float,
    target_power: float,
    candidates: Sequence[float],
    simulations: int,
    resamples: int,
    seed: int,
    effect_sd: float = 0.0,
) -> tuple[float | None, tuple[PowerResult, ...]]:
    """Smallest candidate effect reaching *target_power*, or None if none does.

    ``None`` is a real answer, not an error: it means no effect in the searched
    range is detectable on this corpus, which is the finding R4 exists to
    establish.
    """
    if not 0.0 < target_power < 1.0:
        msg = f"target_power must lie in (0, 1), got {target_power}"
        raise CorpusAuditError(msg)
    curve = tuple(
        simulate_power(
            cluster_sizes,
            effect_points=effect,
            base_rate=base_rate,
            cluster_sd=cluster_sd,
            simulations=simulations,
            resamples=resamples,
            seed=seed,
            effect_sd=effect_sd,
        )
        for effect in sorted(candidates)
    )
    for result in curve:
        if result.clears(target_power):
            return result.effect_points, curve
    return None, curve


@dataclass(frozen=True, slots=True)
class DetectionBracket:
    """The smallest and largest candidate effects consistent with *target_power*.

    A single "minimum detectable effect" is a false precision when the estimator's
    own noise is comparable to the grid spacing: the smallest effect whose power
    clears the target by two standard errors, and the largest that has not yet
    clearly cleared it, bracket the honest answer.
    """

    lower: float | None
    upper: float | None
    target_power: float

    def describe(self) -> str:
        if self.lower is None:
            return f"no searched effect reaches {self.target_power:.0%} power"
        if self.upper is None or self.upper >= self.lower:
            return f"{self.lower:g} points"
        return f"between {self.upper:g} and {self.lower:g} points"


def detection_bracket(curve: Sequence[PowerResult], *, target_power: float) -> DetectionBracket:
    """Bracket the detectable effect, accounting for Monte Carlo noise."""
    lower = next((item.effect_points for item in curve if item.clears(target_power)), None)
    upper = next(
        (
            item.effect_points
            for item in reversed(list(curve))
            if lower is not None and item.effect_points < lower and item.power < target_power
        ),
        None,
    )
    return DetectionBracket(lower=lower, upper=upper, target_power=target_power)


def _as_float(block: dict[str, object], key: str) -> float:
    """Read a numeric field, refusing anything that is not a real number."""
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"expected a number for {key!r}, got {value!r}"
        raise ValueError(msg)
    return float(value)


class CorpusAuditArtifact(BaseModel):
    """The checked-in record of one corpus validity audit."""

    model_config = ConfigDict(extra="forbid")

    corpus_audit_version: Literal[1] = 1
    milestone: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tasks_dir: str = Field(min_length=1)
    task_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_tasks: int = Field(gt=0)
    leakage: dict[str, object] = Field(min_length=1)
    clustering: dict[str, object] = Field(min_length=1)
    held_out: dict[str, object] = Field(min_length=1)
    power: dict[str, object] = Field(min_length=1)
    calibration: dict[str, object] = Field(min_length=1)
    verdict: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_calibration(self) -> CorpusAuditArtifact:
        """Recompute the calibration rather than trusting its conclusion field.

        R4's whole claim is a projection from a simulation, so the artifact may
        not omit the check of that simulation against a real measured interval,
        may not record a check it failed, and may not assert a verdict its own
        recorded numbers contradict. Checking only ``within_tolerance`` would
        validate the one field a miscomputing producer would still set to true.
        """
        required = ("reference", "measured_ci_width", "simulated_ci_width", "within_tolerance")
        for key in required:
            if key not in self.calibration:
                msg = f"calibration must record {key!r}"
                raise ValueError(msg)
        measured = _as_float(self.calibration, "measured_ci_width")
        simulated = _as_float(self.calibration, "simulated_ci_width")
        tolerance = _as_float(self.calibration, "tolerance")
        if measured <= 0.0:
            msg = f"calibration measured_ci_width must be positive, got {measured}"
            raise ValueError(msg)
        implied = abs(simulated - measured) / measured <= tolerance
        if self.calibration["within_tolerance"] is not implied:
            msg = (
                f"calibration records within_tolerance="
                f"{self.calibration['within_tolerance']!r}, but a simulated width of "
                f"{simulated} against a measured {measured} at tolerance {tolerance} "
                f"implies {implied!r}"
            )
            raise ValueError(msg)
        if not implied:
            msg = (
                "calibration did not reproduce the reference interval, so no power "
                "projection in this artifact may be trusted"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> CorpusAuditArtifact:
        """Cross-check the blocks against each other and against total_tasks."""
        raw_sizes = self.clustering.get("cluster_sizes")
        if not isinstance(raw_sizes, dict) or not raw_sizes:
            msg = "clustering must record a non-empty cluster_sizes mapping"
            raise ValueError(msg)
        sizes: dict[str, int] = {str(key): int(value) for key, value in raw_sizes.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        counted = sum(sizes.values())
        if counted != self.total_tasks:
            msg = (
                f"cluster sizes sum to {counted} but total_tasks is {self.total_tasks}; "
                "the audit and the corpus it reports on disagree"
            )
            raise ValueError(msg)
        recorded_clusters = self.clustering.get("cluster_count")
        if isinstance(recorded_clusters, int) and recorded_clusters != len(sizes):
            msg = f"cluster_count {recorded_clusters} does not match {len(sizes)} cluster sizes"
            raise ValueError(msg)
        for key in ("symbol_leak_rate", "any_leak_rate"):
            rate = _as_float(self.leakage, key)
            if not 0.0 <= rate <= 1.0:
                msg = f"leakage {key} must lie in [0, 1], got {rate}"
                raise ValueError(msg)
        for field_name, ids_key, rate_key in (
            ("symbol", "symbol_leaked_tasks", "symbol_leak_rate"),
            ("any", "any_leaked_tasks", "any_leak_rate"),
        ):
            raw_ids = self.leakage.get(ids_key)
            if not isinstance(raw_ids, list):
                msg = f"leakage must record {ids_key} as a list"
                raise ValueError(msg)
            ids: list[str] = [str(item) for item in raw_ids]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            expected = round(len(ids) / self.total_tasks, 4)
            if abs(_as_float(self.leakage, rate_key) - expected) > 1e-6:
                msg = (
                    f"leakage {rate_key} does not match its own {field_name}-tier task "
                    f"list: {len(ids)}/{self.total_tasks} implies {expected}"
                )
                raise ValueError(msg)
        return self


def validate_corpus_audit_artifact(path: Path) -> CorpusAuditArtifact:
    """Load and validate one corpus-audit artifact, failing loudly."""
    if not path.is_file():
        msg = f"Corpus audit artifact is not a file: {path}"
        raise CorpusAuditError(msg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Corpus audit artifact is not readable JSON: {path}: {exc}"
        raise CorpusAuditError(msg) from exc
    try:
        return CorpusAuditArtifact.model_validate(payload)
    except ValidationError as exc:
        msg = f"Corpus audit artifact failed validation: {path}: {exc}"
        raise CorpusAuditError(msg) from exc


def estimate_effect_heterogeneity(
    per_cluster_deltas: Sequence[float], *, tasks_per_cluster: int, base_rate: float
) -> float:
    """Between-cluster effect SD, with within-cluster sampling noise removed.

    The raw spread of measured per-repository deltas is *not* heterogeneity: most
    or all of it can be binomial noise from finitely many tasks per repository.
    Subtracting the expected within-cluster variance is what separates the two,
    and the result is floored at zero because a negative variance estimate means
    the data show no heterogeneity at all rather than a negative amount of it.
    """
    if len(per_cluster_deltas) < 2:
        msg = f"need at least two clusters to estimate heterogeneity, got {len(per_cluster_deltas)}"
        raise CorpusAuditError(msg)
    if tasks_per_cluster < 1:
        msg = f"tasks_per_cluster must be positive, got {tasks_per_cluster}"
        raise CorpusAuditError(msg)
    mean = sum(per_cluster_deltas) / len(per_cluster_deltas)
    observed_variance = sum((value - mean) ** 2 for value in per_cluster_deltas) / (
        len(per_cluster_deltas) - 1
    )
    within_variance = 2.0 * base_rate * (1.0 - base_rate) / tasks_per_cluster * PERCENT * PERCENT
    return math.sqrt(max(0.0, observed_variance - within_variance))
