"""State-management dimension: signals for shared state, persistence, and caching."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import ArchProfile, DimensionComparison
from archex.serve.compare._base import (
    classify_delta,
    module_excerpts,
    modules_matching,
    pattern_excerpts,
    patterns_matching,
)

_PATTERN_KEYWORDS = {"singleton", "repository", "registry", "cache", "store", "state", "session"}
_MODULE_KEYWORDS = {"store", "cache", "state", "repository", "session", "registry"}
_PERSISTENCE_DEPS = {
    "redis",
    "memcache",
    "memcached",
    "sqlite",
    "postgres",
    "psycopg",
    "mysql",
    "pymysql",
    "mongo",
    "pymongo",
    "sqlalchemy",
    "tortoise",
    "peewee",
    "djongo",
}


@dataclass
class StateManagementEvidence:
    """Quantitative signals for stateful design."""

    repo: str
    pattern_count: int
    state_modules: int
    persistence_deps: list[str]
    pattern_samples: list[str]
    module_samples: list[str]


def extract(profile: ArchProfile) -> StateManagementEvidence:
    matched_patterns = patterns_matching(profile.pattern_catalog, _PATTERN_KEYWORDS)
    matched_modules = modules_matching(profile.module_map, _MODULE_KEYWORDS)
    persistence: set[str] = set()
    for mod in profile.module_map:
        for dep in mod.external_deps:
            normalized = dep.lower().split(".")[0].replace("-", "_")
            if normalized in _PERSISTENCE_DEPS:
                persistence.add(normalized)
    return StateManagementEvidence(
        repo=profile.repo.url or profile.repo.local_path or "repo",
        pattern_count=len(matched_patterns),
        state_modules=len(matched_modules),
        persistence_deps=sorted(persistence),
        pattern_samples=pattern_excerpts(matched_patterns),
        module_samples=module_excerpts(matched_modules),
    )


def _approach(ev: StateManagementEvidence) -> str:
    if ev.pattern_count + ev.state_modules + len(ev.persistence_deps) == 0:
        return "No explicit state-management structure detected"
    parts: list[str] = []
    if ev.pattern_count:
        parts.append(f"{ev.pattern_count} state pattern(s)")
    if ev.state_modules:
        parts.append(f"{ev.state_modules} state module(s)")
    if ev.persistence_deps:
        parts.append(f"persistence via {', '.join(ev.persistence_deps)}")
    return "; ".join(parts)


def _evidence_lines(ev: StateManagementEvidence) -> list[str]:
    lines = [
        f"State patterns (singleton/repo/cache/...): {ev.pattern_count}",
        f"State modules: {ev.state_modules}",
        f"Persistence libraries detected: {len(ev.persistence_deps)}",
    ]
    if ev.persistence_deps:
        lines.append(f"Persistence backends: {', '.join(ev.persistence_deps)}")
    lines.extend(ev.pattern_samples)
    lines.extend(ev.module_samples)
    return lines


def compare(a: StateManagementEvidence, b: StateManagementEvidence) -> DimensionComparison:
    trade_offs = [
        classify_delta("state-pattern", a.pattern_count, b.pattern_count),
    ]
    a_only = set(a.persistence_deps) - set(b.persistence_deps)
    b_only = set(b.persistence_deps) - set(a.persistence_deps)
    if a_only:
        trade_offs.append(f"`{a.repo}` uses {', '.join(sorted(a_only))} not seen in `{b.repo}`.")
    if b_only:
        trade_offs.append(f"`{b.repo}` uses {', '.join(sorted(b_only))} not seen in `{a.repo}`.")
    return DimensionComparison(
        dimension="state_management",
        repo_a_approach=_approach(a),
        repo_b_approach=_approach(b),
        evidence_a=_evidence_lines(a),
        evidence_b=_evidence_lines(b),
        trade_offs=trade_offs,
    )
