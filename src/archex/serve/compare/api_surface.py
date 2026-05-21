"""API-surface dimension: public-interface count, annotation coverage, framework signals."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import ArchProfile, DimensionComparison
from archex.serve.compare._base import (
    classify_delta,
    interface_excerpts,
    interfaces_matching,
    pattern_excerpts,
    patterns_matching,
    repo_label,
)

_API_PATTERN_KEYWORDS = {"api", "endpoint", "route", "handler", "controller", "view", "resource"}
_API_PATH_KEYWORDS = {"api", "routes", "endpoints", "views", "handlers"}


@dataclass
class ApiSurfaceEvidence:
    """Quantitative signals for public API design."""

    repo: str
    total_interfaces: int
    api_named_interfaces: int
    return_type_coverage: float  # 0.0 - 1.0
    docstring_coverage: float  # 0.0 - 1.0
    avg_parameters: float
    framework_patterns: int
    interface_samples: list[str]
    pattern_samples: list[str]


def extract(profile: ArchProfile) -> ApiSurfaceEvidence:
    interfaces = profile.interface_surface
    total = len(interfaces)
    if total > 0:
        return_annotated = sum(1 for i in interfaces if i.return_type)
        documented = sum(1 for i in interfaces if i.docstring)
        param_total = sum(len(i.parameters) for i in interfaces)
        return_type_cov = return_annotated / total
        docstring_cov = documented / total
        avg_params = param_total / total
    else:
        return_type_cov = 0.0
        docstring_cov = 0.0
        avg_params = 0.0

    api_named = interfaces_matching(interfaces, _API_PATH_KEYWORDS)
    framework_patterns = patterns_matching(profile.pattern_catalog, _API_PATTERN_KEYWORDS)

    return ApiSurfaceEvidence(
        repo=repo_label(profile),
        total_interfaces=total,
        api_named_interfaces=len(api_named),
        return_type_coverage=return_type_cov,
        docstring_coverage=docstring_cov,
        avg_parameters=avg_params,
        framework_patterns=len(framework_patterns),
        interface_samples=interface_excerpts(api_named),
        pattern_samples=pattern_excerpts(framework_patterns),
    )


def _approach(ev: ApiSurfaceEvidence) -> str:
    if ev.total_interfaces == 0:
        return "No public interfaces detected"
    return (
        f"{ev.total_interfaces} public interface(s); "
        f"return-type coverage {ev.return_type_coverage:.0%}, "
        f"docstring coverage {ev.docstring_coverage:.0%}"
    )


def _evidence_lines(ev: ApiSurfaceEvidence) -> list[str]:
    lines = [
        f"Total public interfaces: {ev.total_interfaces}",
        f"API-pathed interfaces: {ev.api_named_interfaces}",
        f"Return-type annotation coverage: {ev.return_type_coverage:.1%}",
        f"Docstring coverage: {ev.docstring_coverage:.1%}",
        f"Average parameters per interface: {ev.avg_parameters:.1f}",
        f"Framework patterns (api/route/handler): {ev.framework_patterns}",
    ]
    lines.extend(ev.interface_samples)
    lines.extend(ev.pattern_samples)
    return lines


def compare(a: ApiSurfaceEvidence, b: ApiSurfaceEvidence) -> DimensionComparison:
    trade_offs = [
        classify_delta("public-interface", a.total_interfaces, b.total_interfaces),
    ]
    if a.total_interfaces and b.total_interfaces:
        if a.return_type_coverage >= b.return_type_coverage + 0.15:
            trade_offs.append(
                f"`{a.repo}` has stronger return-type discipline "
                f"({a.return_type_coverage:.0%} vs {b.return_type_coverage:.0%})."
            )
        elif b.return_type_coverage >= a.return_type_coverage + 0.15:
            trade_offs.append(
                f"`{b.repo}` has stronger return-type discipline "
                f"({b.return_type_coverage:.0%} vs {a.return_type_coverage:.0%})."
            )
        if a.docstring_coverage >= b.docstring_coverage + 0.15:
            trade_offs.append(
                f"`{a.repo}` documents interfaces more thoroughly "
                f"({a.docstring_coverage:.0%} vs {b.docstring_coverage:.0%})."
            )
        elif b.docstring_coverage >= a.docstring_coverage + 0.15:
            trade_offs.append(
                f"`{b.repo}` documents interfaces more thoroughly "
                f"({b.docstring_coverage:.0%} vs {a.docstring_coverage:.0%})."
            )
    return DimensionComparison(
        dimension="api_surface",
        repo_a_approach=_approach(a),
        repo_b_approach=_approach(b),
        evidence_a=_evidence_lines(a),
        evidence_b=_evidence_lines(b),
        trade_offs=trade_offs,
    )
