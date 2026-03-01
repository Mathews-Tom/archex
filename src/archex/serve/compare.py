"""Comparison assembly: diff two ArchProfiles across dimensions into a ComparisonResult."""

from __future__ import annotations

from archex.models import (
    ArchProfile,
    ComparisonResult,
    DetectedPattern,
    DimensionComparison,
    Interface,
    Module,
)

SUPPORTED_DIMENSIONS = frozenset(
    {
        "error_handling",
        "api_surface",
        "state_management",
        "concurrency",
        "testing",
        "configuration",
    }
)

_DEFAULT_DIMENSIONS = sorted(SUPPORTED_DIMENSIONS)


def validate_dimensions(dims: list[str]) -> None:
    """Raise ValueError if any dimension is not supported."""
    unsupported = set(dims) - SUPPORTED_DIMENSIONS
    if unsupported:
        raise ValueError(
            f"Unsupported dimensions: {', '.join(sorted(unsupported))}. "
            f"Supported: {', '.join(sorted(SUPPORTED_DIMENSIONS))}"
        )


# ---------------------------------------------------------------------------
# Per-dimension evidence extractors
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = {"error_handler", "custom_exception", "exception", "error", "result_type"}
_CONCURRENCY_PATTERNS = {"async", "thread", "lock", "queue", "pool", "concurrent", "parallel"}
_STATE_PATTERNS = {
    "singleton",
    "repository",
    "registry",
    "cache",
    "store",
    "state",
    "session",
}
_CONFIG_PATTERNS = {"config", "settings", "env", "environment", "options", "configuration"}
_TESTING_PATTERNS = {"test", "fixture", "mock", "assert", "spec", "setup", "teardown"}


def _patterns_matching(
    patterns: list[DetectedPattern],
    keywords: set[str],
) -> list[str]:
    """Return evidence strings for patterns whose name or description contains keywords."""
    evidence: list[str] = []
    for pat in patterns:
        name_lower = pat.name.lower()
        desc_lower = pat.description.lower()
        if any(kw in name_lower or kw in desc_lower for kw in keywords):
            locations = ", ".join(e.file_path for e in pat.evidence[:3])
            evidence.append(
                f"Pattern '{pat.display_name}' ({pat.category.value}, "
                f"confidence={pat.confidence:.0%})" + (f" in {locations}" if locations else "")
            )
    return evidence


def _interfaces_matching(
    interfaces: list[Interface],
    keywords: set[str],
) -> list[str]:
    """Return evidence strings for interfaces whose name/signature matches keywords."""
    evidence: list[str] = []
    for iface in interfaces:
        sig_lower = iface.signature.lower()
        name_lower = iface.symbol.name.lower()
        path_lower = iface.symbol.file_path.lower()
        if any(kw in sig_lower or kw in name_lower or kw in path_lower for kw in keywords):
            evidence.append(f"Interface '{iface.signature}' ({iface.symbol.file_path})")
    return evidence


def _modules_matching(
    modules: list[Module],
    keywords: set[str],
) -> list[str]:
    """Return evidence strings for modules whose name or deps match keywords."""
    evidence: list[str] = []
    for mod in modules:
        name_lower = mod.name.lower()
        if any(kw in name_lower for kw in keywords):
            evidence.append(f"Module '{mod.name}' ({mod.file_count} files, {mod.line_count} lines)")
        for dep in mod.external_deps:
            if any(kw in dep.lower() for kw in keywords):
                evidence.append(f"Module '{mod.name}' depends on '{dep}'")
    return evidence


def _summarize_approach(evidence: list[str], dimension: str) -> str:
    """Build a short approach description from evidence."""
    if not evidence:
        return f"No {dimension.replace('_', ' ')} evidence found"
    return f"{len(evidence)} indicator(s) detected"


def _extract_dimension(
    profile: ArchProfile,
    dimension: str,
) -> list[str]:
    """Extract evidence for a single dimension from an ArchProfile."""
    keyword_map: dict[str, set[str]] = {
        "error_handling": _ERROR_PATTERNS,
        "api_surface": {"api", "endpoint", "route", "handler", "controller", "view", "resource"},
        "state_management": _STATE_PATTERNS,
        "concurrency": _CONCURRENCY_PATTERNS,
        "testing": _TESTING_PATTERNS,
        "configuration": _CONFIG_PATTERNS,
    }
    keywords = keyword_map.get(dimension, {dimension.replace("_", " ")})

    evidence: list[str] = []
    evidence.extend(_patterns_matching(profile.pattern_catalog, keywords))
    evidence.extend(_interfaces_matching(profile.interface_surface, keywords))
    evidence.extend(_modules_matching(profile.module_map, keywords))

    # Decisions as evidence
    for dec in profile.decision_log:
        dec_lower = dec.decision.lower()
        if any(kw in dec_lower for kw in keywords):
            evidence.append(f"Decision: {dec.decision} ({dec.source})")

    # Stats-based evidence for specific dimensions
    if dimension == "api_surface":
        pub_count = len(profile.interface_surface)
        if pub_count > 0:
            evidence.append(f"Total public interfaces: {pub_count}")

    if dimension == "testing":
        for lang, ls in profile.stats.languages.items():
            if "test" in lang.lower():
                evidence.append(f"Test language stats: {ls.files} files, {ls.lines} lines")

    return evidence


def _compute_trade_offs(
    dimension: str,
    evidence_a: list[str],
    evidence_b: list[str],
) -> list[str]:
    """Derive trade-off observations from comparing evidence counts and types."""
    trade_offs: list[str] = []
    len_a = len(evidence_a)
    len_b = len(evidence_b)

    dim_label = dimension.replace("_", " ")

    if len_a == 0 and len_b == 0:
        trade_offs.append(f"Neither repository shows significant {dim_label} patterns")
    elif len_a == 0:
        trade_offs.append(f"Repo A lacks {dim_label} evidence; Repo B has {len_b} indicator(s)")
    elif len_b == 0:
        trade_offs.append(f"Repo A has {len_a} indicator(s); Repo B lacks {dim_label} evidence")
    elif len_a > len_b * 2:
        trade_offs.append(
            f"Repo A has significantly more {dim_label} coverage ({len_a} vs {len_b} indicators)"
        )
    elif len_b > len_a * 2:
        trade_offs.append(
            f"Repo B has significantly more {dim_label} coverage ({len_b} vs {len_a} indicators)"
        )
    else:
        trade_offs.append(
            f"Both repositories show comparable {dim_label} coverage "
            f"({len_a} vs {len_b} indicators)"
        )

    return trade_offs


def _build_summary(
    profile_a: ArchProfile,
    profile_b: ArchProfile,
    dimensions: list[DimensionComparison],
) -> str:
    """Build a textual summary of the comparison."""
    name_a = profile_a.repo.url or profile_a.repo.local_path or "Repo A"
    name_b = profile_b.repo.url or profile_b.repo.local_path or "Repo B"

    lines: list[str] = [
        f"Comparison of {name_a} ({profile_a.stats.total_files} files, "
        f"{profile_a.stats.total_lines} lines) vs "
        f"{name_b} ({profile_b.stats.total_files} files, "
        f"{profile_b.stats.total_lines} lines).",
    ]

    for dim in dimensions:
        total_a = len(dim.evidence_a)
        total_b = len(dim.evidence_b)
        lines.append(f"  {dim.dimension}: A={total_a} indicator(s), B={total_b} indicator(s)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_repos(
    profile_a: ArchProfile,
    profile_b: ArchProfile,
    dimensions: list[str] | None = None,
) -> ComparisonResult:
    """Compare two ArchProfiles across specified dimensions.

    Args:
        profile_a: First repository's architecture profile.
        profile_b: Second repository's architecture profile.
        dimensions: List of dimension names to compare. If None, all supported
            dimensions are used.

    Returns:
        A ComparisonResult with per-dimension comparisons and a summary.

    Raises:
        ValueError: If any requested dimension is not supported.
    """
    dims = _DEFAULT_DIMENSIONS if dimensions is None else dimensions
    validate_dimensions(dims)

    comparisons: list[DimensionComparison] = []
    for dim in dims:
        evidence_a = _extract_dimension(profile_a, dim)
        evidence_b = _extract_dimension(profile_b, dim)
        trade_offs = _compute_trade_offs(dim, evidence_a, evidence_b)

        comparisons.append(
            DimensionComparison(
                dimension=dim,
                repo_a_approach=_summarize_approach(evidence_a, dim),
                repo_b_approach=_summarize_approach(evidence_b, dim),
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                trade_offs=trade_offs,
            )
        )

    summary = _build_summary(profile_a, profile_b, comparisons)

    return ComparisonResult(
        repo_a=profile_a.repo,
        repo_b=profile_b.repo,
        dimensions=comparisons,
        summary=summary,
    )
