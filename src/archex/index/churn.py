"""Per-file recency/churn priors for benchmark-only ranking candidates.

Provides a deterministic, optional, artifact-backed churn/recency prior. The
prior is a small per-file multiplier (>= ``1.0``) that benchmark candidate lanes
may apply to the final ranking score; the product default path never builds it.

Two sources are supported, resolved in order:

1. A checked-in precomputed per-file churn fixture (``archex.churn.v1`` JSON),
   for benchmark lanes whose clones are shallow.
2. Real git history from a full local clone.

When neither is available (a shallow benchmark clone with no fixture) every
prior is neutral (multiplier ``1.0``) so the candidate's ranking is bit-identical
to ``archex_query``. The prior is therefore never a source of nondeterminism:
given a fixed commit and a fixed fixture it is reproducible, and given no usable
history it is the identity.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

CHURN_SOURCE_HISTORY = "history"
CHURN_SOURCE_FIXTURE = "fixture"
CHURN_SOURCE_NEUTRAL = "neutral_fallback"

# Schema tag every precomputed churn fixture must carry.
CHURN_FIXTURE_SCHEMA = "archex.churn.v1"

# Maximum multiplicative boost a maximally-churned file can receive. Small by
# design: the prior may break ties and nudge ordering but must never dominate the
# relevance signal. Applied as ``final *= 1.0 + max_boost * churn_score`` with
# ``churn_score`` in ``[0, 1]``.
DEFAULT_CHURN_MAX_BOOST = 0.05

# Bound the history scan so a deep repository cannot make a benchmark lane
# unbounded. A fixed HEAD plus a fixed cap stays deterministic.
_MAX_HISTORY_COMMITS = 5000
_GIT_TIMEOUT_SECONDS = 60

# Blend weights for the churn score: commit frequency vs last-touched recency.
_FREQUENCY_WEIGHT = 0.5
_RECENCY_WEIGHT = 0.5

# Marks a commit-timestamp line in the parsed ``git log`` stream; no file path
# begins with this control byte, so it cannot collide with a name-only entry.
_COMMIT_MARKER = "\x01"


class ChurnError(Exception):
    """Raised when a churn fixture is malformed."""


@dataclass(frozen=True)
class FileChurnMetrics:
    """Raw per-file churn inputs.

    ``commits`` is the number of commits that touched the file; ``recency`` is in
    ``[0, 1]`` where ``1.0`` is the most recently touched file in the set.
    """

    commits: int
    recency: float


def _empty_priors() -> dict[str, float]:
    """Typed default factory for :attr:`ChurnPriors.priors`."""
    return {}


@dataclass(frozen=True)
class ChurnPriors:
    """Per-file ranking priors plus the source they were derived from.

    ``priors`` maps a repo-relative file path to a multiplier in
    ``[1.0, 1.0 + max_boost]``. Files absent from the map are neutral (``1.0``);
    a neutral-fallback result carries an empty map so every lookup returns
    ``1.0``.
    """

    source: str
    commit: str
    max_boost: float
    priors: dict[str, float] = field(default_factory=_empty_priors)

    def prior_for(self, file_path: str) -> float:
        """Return the bounded multiplier for ``file_path`` (``1.0`` when unknown)."""
        return self.priors.get(file_path, 1.0)


def load_churn_priors(
    repo_path: Path,
    *,
    fixture_path: Path | None = None,
    max_boost: float = DEFAULT_CHURN_MAX_BOOST,
) -> ChurnPriors:
    """Resolve per-file churn priors for ``repo_path``.

    Resolution order:

    1. ``fixture_path`` when it exists — a checked-in precomputed per-file churn
       fixture (deterministic for benchmark lanes).
    2. Real git history from a full local clone.
    3. Neutral fallback (every prior ``1.0``) when neither is available — e.g. a
       shallow benchmark clone with no fixture. This guarantees the candidate's
       ranking is identical to ``archex_query``.
    """
    if fixture_path is not None and fixture_path.exists():
        metrics, commit = _load_fixture(fixture_path)
        return ChurnPriors(
            source=CHURN_SOURCE_FIXTURE,
            commit=commit,
            max_boost=max_boost,
            priors=_priors_from_metrics(metrics, max_boost),
        )
    metrics, commit = _read_history(repo_path)
    if not metrics:
        return ChurnPriors(source=CHURN_SOURCE_NEUTRAL, commit="", max_boost=max_boost, priors={})
    return ChurnPriors(
        source=CHURN_SOURCE_HISTORY,
        commit=commit,
        max_boost=max_boost,
        priors=_priors_from_metrics(metrics, max_boost),
    )


def _priors_from_metrics(
    metrics: dict[str, FileChurnMetrics], max_boost: float
) -> dict[str, float]:
    """Normalize raw per-file metrics into bounded multipliers (``> 1.0``).

    Frequency is normalized by the maximum commit count in the set; recency is
    already in ``[0, 1]``. Files whose blended score is ``0`` are omitted so they
    stay neutral via :meth:`ChurnPriors.prior_for`.
    """
    if not metrics or max_boost <= 0.0:
        return {}
    max_commits = max(metric.commits for metric in metrics.values())
    priors: dict[str, float] = {}
    for path, metric in metrics.items():
        frequency = (metric.commits / max_commits) if max_commits > 0 else 0.0
        recency = min(max(metric.recency, 0.0), 1.0)
        score = _FREQUENCY_WEIGHT * frequency + _RECENCY_WEIGHT * recency
        multiplier = 1.0 + max_boost * score
        if multiplier > 1.0:
            priors[path] = multiplier
    return priors


def _load_fixture(fixture_path: Path) -> tuple[dict[str, FileChurnMetrics], str]:
    """Parse a precomputed ``archex.churn.v1`` fixture into raw metrics."""
    try:
        decoded: object = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChurnError(f"failed to read churn fixture {fixture_path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ChurnError(f"churn fixture {fixture_path} must be a JSON object")
    raw = cast("dict[str, object]", decoded)
    schema = raw.get("schema")
    if schema != CHURN_FIXTURE_SCHEMA:
        raise ChurnError(
            f"churn fixture {fixture_path} has schema {schema!r}, expected {CHURN_FIXTURE_SCHEMA!r}"
        )
    files_obj = raw.get("files")
    if not isinstance(files_obj, dict):
        raise ChurnError(f"churn fixture {fixture_path} must carry a 'files' object")
    files = cast("dict[str, object]", files_obj)
    commit = str(raw.get("commit", ""))
    metrics: dict[str, FileChurnMetrics] = {}
    for path, entry in files.items():
        metrics[path] = _fixture_entry_metrics(path, entry)
    return metrics, commit


def _fixture_entry_metrics(path: str, entry: object) -> FileChurnMetrics:
    """Validate and convert a single fixture file entry into metrics."""
    if not isinstance(entry, dict):
        raise ChurnError(f"churn fixture entry for {path!r} must be an object")
    fields = cast("dict[str, object]", entry)
    commits = fields.get("commits", 0)
    recency = fields.get("recency", 0.0)
    if isinstance(commits, bool) or not isinstance(commits, int) or commits < 0:
        raise ChurnError(f"churn fixture entry {path!r} has invalid 'commits': {commits!r}")
    if (
        isinstance(recency, bool)
        or not isinstance(recency, (int, float))
        or not math.isfinite(recency)
    ):
        raise ChurnError(f"churn fixture entry {path!r} has invalid 'recency': {recency!r}")
    return FileChurnMetrics(commits=commits, recency=float(recency))


def _read_history(repo_path: Path) -> tuple[dict[str, FileChurnMetrics], str]:
    """Read per-file churn from a full local clone.

    Returns ``({}, "")`` when history is unavailable: no ``.git``, a shallow
    clone, a single-commit repository, or any git failure. Each of those yields
    the neutral fallback in :func:`load_churn_priors`.
    """
    if not (repo_path / ".git").exists():
        return {}, ""
    if _is_shallow(repo_path):
        return {}, ""
    if _commit_count(repo_path) <= 1:
        return {}, ""
    commit = _git_output(repo_path, ["rev-parse", "HEAD"]) or ""
    if not commit:
        return {}, ""
    log = _git_output(
        repo_path,
        [
            "log",
            f"--max-count={_MAX_HISTORY_COMMITS}",
            "--no-merges",
            "--no-renames",
            "--name-only",
            f"--format=format:{_COMMIT_MARKER}%ct",
        ],
    )
    if log is None:
        return {}, ""
    metrics = _parse_git_log(log)
    if not metrics:
        return {}, ""
    return metrics, commit


def _parse_git_log(log: str) -> dict[str, FileChurnMetrics]:
    """Parse a marked ``git log --name-only`` stream into per-file metrics."""
    commit_counts: dict[str, int] = {}
    last_touched: dict[str, int] = {}
    current_ts = 0
    for raw in log.splitlines():
        if not raw:
            continue
        if raw[0] == _COMMIT_MARKER:
            try:
                current_ts = int(raw[1:])
            except ValueError:
                current_ts = 0
            continue
        commit_counts[raw] = commit_counts.get(raw, 0) + 1
        if current_ts > last_touched.get(raw, 0):
            last_touched[raw] = current_ts
    if not commit_counts or not last_touched:
        return {}
    min_ts = min(last_touched.values())
    max_ts = max(last_touched.values())
    span = max_ts - min_ts
    metrics: dict[str, FileChurnMetrics] = {}
    for path, count in commit_counts.items():
        touched = last_touched.get(path, min_ts)
        recency = ((touched - min_ts) / span) if span > 0 else 1.0
        metrics[path] = FileChurnMetrics(commits=count, recency=recency)
    return metrics


def _is_shallow(repo_path: Path) -> bool:
    """Whether ``repo_path`` is a shallow clone (missing representative history)."""
    return _git_output(repo_path, ["rev-parse", "--is-shallow-repository"]) == "true"


def _commit_count(repo_path: Path) -> int:
    """Number of commits reachable from HEAD, or ``0`` when git fails."""
    out = _git_output(repo_path, ["rev-list", "--count", "HEAD"])
    if out is None:
        return 0
    try:
        return int(out)
    except ValueError:
        return 0


def _git_output(repo_path: Path, args: list[str]) -> str | None:
    """Run a read-only git command, returning stripped stdout or ``None``."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
