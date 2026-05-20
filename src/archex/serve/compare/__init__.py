"""Cross-repo comparison: dispatch per-dimension templates to produce a ComparisonResult.

Each dimension is implemented as a sibling module with `extract(profile)` and
`compare(a, b)` functions plus a per-dimension Evidence dataclass. The public
entry point `compare_repos` validates the requested dimension list and walks
each template in the user-provided order.

Output is fully deterministic; no LLM call is made. The contract surface
(`DimensionComparison`, `ComparisonResult`) is unchanged from prior versions,
so existing consumers (CLI, MCP, JSON renderers) require no edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.models import ArchProfile, ComparisonResult, DimensionComparison
from archex.serve.compare import (
    api_surface,
    concurrency,
    configuration,
    error_handling,
    state_management,
    testing,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_DISPATCH: dict[str, Callable[[ArchProfile, ArchProfile], DimensionComparison]] = {
    "error_handling": lambda a, b: error_handling.compare(
        error_handling.extract(a), error_handling.extract(b)
    ),
    "api_surface": lambda a, b: api_surface.compare(api_surface.extract(a), api_surface.extract(b)),
    "state_management": lambda a, b: state_management.compare(
        state_management.extract(a), state_management.extract(b)
    ),
    "concurrency": lambda a, b: concurrency.compare(concurrency.extract(a), concurrency.extract(b)),
    "testing": lambda a, b: testing.compare(testing.extract(a), testing.extract(b)),
    "configuration": lambda a, b: configuration.compare(
        configuration.extract(a), configuration.extract(b)
    ),
}

SUPPORTED_DIMENSIONS: frozenset[str] = frozenset(_DISPATCH)

_DEFAULT_DIMENSIONS: list[str] = sorted(_DISPATCH)


def validate_dimensions(dims: list[str]) -> None:
    """Raise ValueError if any dimension is not supported."""
    unsupported = set(dims) - SUPPORTED_DIMENSIONS
    if unsupported:
        raise ValueError(
            f"Unsupported dimensions: {', '.join(sorted(unsupported))}. "
            f"Supported: {', '.join(sorted(SUPPORTED_DIMENSIONS))}"
        )


def _build_summary(
    profile_a: ArchProfile,
    profile_b: ArchProfile,
    comparisons: list[DimensionComparison],
) -> str:
    name_a = profile_a.repo.url or profile_a.repo.local_path or "Repo A"
    name_b = profile_b.repo.url or profile_b.repo.local_path or "Repo B"
    lines: list[str] = [
        f"Comparison of {name_a} ({profile_a.stats.total_files} files, "
        f"{profile_a.stats.total_lines} lines) vs "
        f"{name_b} ({profile_b.stats.total_files} files, "
        f"{profile_b.stats.total_lines} lines).",
    ]
    for comp in comparisons:
        lines.append(f"  {comp.dimension}: A — {comp.repo_a_approach}; B — {comp.repo_b_approach}")
    return "\n".join(lines)


def compare_repos(
    profile_a: ArchProfile,
    profile_b: ArchProfile,
    dimensions: list[str] | None = None,
) -> ComparisonResult:
    """Compare two ArchProfiles across the requested dimensions.

    Args:
        profile_a: First repository's architecture profile.
        profile_b: Second repository's architecture profile.
        dimensions: Names from SUPPORTED_DIMENSIONS to evaluate. If None, all
            dimensions are evaluated in their default sorted order.

    Returns:
        ComparisonResult with one DimensionComparison per requested dimension,
        plus a textual summary header.

    Raises:
        ValueError: If any requested dimension is not in SUPPORTED_DIMENSIONS.
    """
    dims = list(_DEFAULT_DIMENSIONS) if dimensions is None else list(dimensions)
    validate_dimensions(dims)
    comparisons = [_DISPATCH[dim](profile_a, profile_b) for dim in dims]
    return ComparisonResult(
        repo_a=profile_a.repo,
        repo_b=profile_b.repo,
        dimensions=comparisons,
        summary=_build_summary(profile_a, profile_b, comparisons),
    )


__all__ = ["SUPPORTED_DIMENSIONS", "compare_repos", "validate_dimensions"]
