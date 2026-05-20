"""Concurrency dimension: async signatures, threading/multiproc deps, sync primitives."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import ArchProfile, DimensionComparison
from archex.serve.compare._base import (
    classify_delta,
    interface_excerpts,
    pattern_excerpts,
    patterns_matching,
)

_PATTERN_KEYWORDS = {"async", "thread", "lock", "queue", "pool", "concurrent", "parallel"}
_CONCURRENCY_DEPS = {
    "asyncio",
    "trio",
    "anyio",
    "threading",
    "multiprocessing",
    "concurrent",
    "celery",
    "rq",
    "aiohttp",
    "uvloop",
}
_ASYNC_SIG_MARKERS = ("async def", "async (", "await ")


@dataclass
class ConcurrencyEvidence:
    """Quantitative signals for concurrency approach."""

    repo: str
    async_interfaces: int
    total_interfaces: int
    pattern_count: int
    concurrency_deps: list[str]
    pattern_samples: list[str]
    interface_samples: list[str]


def extract(profile: ArchProfile) -> ConcurrencyEvidence:
    interfaces = profile.interface_surface
    async_ifaces = [
        i for i in interfaces if any(marker in i.signature for marker in _ASYNC_SIG_MARKERS)
    ]
    patterns = patterns_matching(profile.pattern_catalog, _PATTERN_KEYWORDS)
    deps: set[str] = set()
    for mod in profile.module_map:
        for dep in mod.external_deps:
            normalized = dep.lower().split(".")[0].replace("-", "_")
            if normalized in _CONCURRENCY_DEPS:
                deps.add(normalized)
    return ConcurrencyEvidence(
        repo=profile.repo.url or profile.repo.local_path or "repo",
        async_interfaces=len(async_ifaces),
        total_interfaces=len(interfaces),
        pattern_count=len(patterns),
        concurrency_deps=sorted(deps),
        pattern_samples=pattern_excerpts(patterns),
        interface_samples=interface_excerpts(async_ifaces),
    )


def _async_ratio(ev: ConcurrencyEvidence) -> float:
    if ev.total_interfaces == 0:
        return 0.0
    return ev.async_interfaces / ev.total_interfaces


def _approach(ev: ConcurrencyEvidence) -> str:
    if ev.async_interfaces + ev.pattern_count + len(ev.concurrency_deps) == 0:
        return "Predominantly synchronous; no concurrency primitives detected"
    parts: list[str] = []
    if ev.async_interfaces:
        parts.append(f"{ev.async_interfaces} async interface(s) ({_async_ratio(ev):.0%})")
    if ev.pattern_count:
        parts.append(f"{ev.pattern_count} concurrency pattern(s)")
    if ev.concurrency_deps:
        parts.append(f"deps: {', '.join(ev.concurrency_deps)}")
    return "; ".join(parts)


def _evidence_lines(ev: ConcurrencyEvidence) -> list[str]:
    async_summary = (
        f"Async interfaces: {ev.async_interfaces} of {ev.total_interfaces} ({_async_ratio(ev):.1%})"
    )
    lines = [
        async_summary,
        f"Concurrency patterns: {ev.pattern_count}",
        f"Concurrency libraries: {len(ev.concurrency_deps)}",
    ]
    if ev.concurrency_deps:
        lines.append(f"Libraries: {', '.join(ev.concurrency_deps)}")
    lines.extend(ev.pattern_samples)
    lines.extend(ev.interface_samples)
    return lines


def compare(a: ConcurrencyEvidence, b: ConcurrencyEvidence) -> DimensionComparison:
    trade_offs = [classify_delta("concurrency-pattern", a.pattern_count, b.pattern_count)]
    ratio_a = _async_ratio(a)
    ratio_b = _async_ratio(b)
    if ratio_a >= ratio_b + 0.15:
        trade_offs.append(
            f"`{a.repo}` is markedly more async-heavy ({ratio_a:.0%} vs {ratio_b:.0%})."
        )
    elif ratio_b >= ratio_a + 0.15:
        trade_offs.append(
            f"`{b.repo}` is markedly more async-heavy ({ratio_b:.0%} vs {ratio_a:.0%})."
        )
    a_only = set(a.concurrency_deps) - set(b.concurrency_deps)
    b_only = set(b.concurrency_deps) - set(a.concurrency_deps)
    if a_only:
        trade_offs.append(f"`{a.repo}` adopts {', '.join(sorted(a_only))} not seen in `{b.repo}`.")
    if b_only:
        trade_offs.append(f"`{b.repo}` adopts {', '.join(sorted(b_only))} not seen in `{a.repo}`.")
    return DimensionComparison(
        dimension="concurrency",
        repo_a_approach=_approach(a),
        repo_b_approach=_approach(b),
        evidence_a=_evidence_lines(a),
        evidence_b=_evidence_lines(b),
        trade_offs=trade_offs,
    )
