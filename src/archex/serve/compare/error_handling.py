"""Error-handling dimension: count exception-related patterns, modules, and interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import ArchProfile, DimensionComparison
from archex.serve.compare._base import (
    classify_delta,
    interface_excerpts,
    interfaces_matching,
    module_excerpts,
    modules_matching,
    pattern_excerpts,
    patterns_matching,
)

_PATTERN_KEYWORDS = {"error_handler", "custom_exception", "exception", "error", "result_type"}
_INTERFACE_KEYWORDS = {"raise", "error", "exception", "fail"}
_MODULE_KEYWORDS = {"error", "exception", "errors", "exceptions"}


@dataclass
class ErrorHandlingEvidence:
    """Quantitative signals for error-handling design in a repository."""

    repo: str
    pattern_count: int
    interface_count: int
    module_count: int
    pattern_samples: list[str]
    interface_samples: list[str]
    module_samples: list[str]


def extract(profile: ArchProfile) -> ErrorHandlingEvidence:
    matched_patterns = patterns_matching(profile.pattern_catalog, _PATTERN_KEYWORDS)
    matched_interfaces = interfaces_matching(profile.interface_surface, _INTERFACE_KEYWORDS)
    matched_modules = modules_matching(profile.module_map, _MODULE_KEYWORDS)
    return ErrorHandlingEvidence(
        repo=profile.repo.url or profile.repo.local_path or "repo",
        pattern_count=len(matched_patterns),
        interface_count=len(matched_interfaces),
        module_count=len(matched_modules),
        pattern_samples=pattern_excerpts(matched_patterns),
        interface_samples=interface_excerpts(matched_interfaces),
        module_samples=module_excerpts(matched_modules),
    )


def _approach(ev: ErrorHandlingEvidence) -> str:
    total = ev.pattern_count + ev.interface_count + ev.module_count
    if total == 0:
        return "No explicit error-handling structure detected"
    parts: list[str] = []
    if ev.pattern_count:
        parts.append(f"{ev.pattern_count} pattern(s)")
    if ev.module_count:
        parts.append(f"{ev.module_count} dedicated module(s)")
    if ev.interface_count:
        parts.append(f"{ev.interface_count} raise/error interface(s)")
    return f"Surface includes {', '.join(parts)}"


def _evidence_lines(ev: ErrorHandlingEvidence) -> list[str]:
    lines = [
        f"Exception-related patterns: {ev.pattern_count}",
        f"Error/raise interfaces: {ev.interface_count}",
        f"Dedicated error modules: {ev.module_count}",
    ]
    lines.extend(ev.pattern_samples)
    lines.extend(ev.module_samples)
    lines.extend(ev.interface_samples)
    return lines


def compare(a: ErrorHandlingEvidence, b: ErrorHandlingEvidence) -> DimensionComparison:
    trade_offs = [
        classify_delta("exception-pattern", a.pattern_count, b.pattern_count),
        classify_delta("error-module", a.module_count, b.module_count),
    ]
    if a.pattern_count + a.module_count == 0 and b.pattern_count + b.module_count > 0:
        trade_offs.append(f"`{b.repo}` codifies error handling as structure; `{a.repo}` does not.")
    elif b.pattern_count + b.module_count == 0 and a.pattern_count + a.module_count > 0:
        trade_offs.append(f"`{a.repo}` codifies error handling as structure; `{b.repo}` does not.")
    return DimensionComparison(
        dimension="error_handling",
        repo_a_approach=_approach(a),
        repo_b_approach=_approach(b),
        evidence_a=_evidence_lines(a),
        evidence_b=_evidence_lines(b),
        trade_offs=trade_offs,
    )
