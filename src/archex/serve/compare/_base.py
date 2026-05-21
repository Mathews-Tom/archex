"""Shared helpers for per-dimension comparison templates.

Each per-dimension template (error_handling, api_surface, etc.) consumes the
same ArchProfile inputs and emits DimensionComparison. The helpers here
provide the small set of utilities every template reuses: keyword matching,
file-path excerpt extraction, and delta classification across paired counts.

The split keeps each per-dimension file focused on its own evidence schema
and rendering, not on string-search plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import ArchProfile, DetectedPattern, Interface, Module


def repo_label(profile: ArchProfile) -> str:
    """Return a stable, human-readable identifier for a repository profile."""
    return profile.repo.url or profile.repo.local_path or "repo"


def matched_deps(modules: list[Module], known: set[str]) -> list[str]:
    """Return external deps from any module that match the known set.

    Normalises deps via lowercase + split-on-dot (top package) + hyphen→underscore
    so that "python-dotenv", "Python.DotEnv", "python_dotenv" all collapse to
    the same key before set membership.
    """
    found: set[str] = set()
    for mod in modules:
        for dep in mod.external_deps:
            normalized = dep.lower().split(".")[0].replace("-", "_")
            if normalized in known:
                found.add(normalized)
    return sorted(found)


def library_diff_trade_offs(
    a_repo: str,
    b_repo: str,
    a_deps: list[str],
    b_deps: list[str],
    verb: str = "uses",
) -> list[str]:
    """Build trade-off lines for libraries present in one repo and not the other."""
    out: list[str] = []
    a_only = set(a_deps) - set(b_deps)
    b_only = set(b_deps) - set(a_deps)
    if a_only:
        out.append(f"`{a_repo}` {verb} {', '.join(sorted(a_only))} not seen in `{b_repo}`.")
    if b_only:
        out.append(f"`{b_repo}` {verb} {', '.join(sorted(b_only))} not seen in `{a_repo}`.")
    return out


def patterns_matching(
    patterns: list[DetectedPattern],
    keywords: set[str],
) -> list[DetectedPattern]:
    """Return patterns whose name or description contains any of the keywords."""
    matched: list[DetectedPattern] = []
    for pat in patterns:
        haystack = f"{pat.name.lower()} {pat.description.lower()}"
        if any(kw in haystack for kw in keywords):
            matched.append(pat)
    return matched


def interfaces_matching(
    interfaces: list[Interface],
    keywords: set[str],
) -> list[Interface]:
    """Return interfaces whose name/signature/path contains any of the keywords."""
    matched: list[Interface] = []
    for iface in interfaces:
        haystack = (
            f"{iface.signature.lower()} "
            f"{iface.symbol.name.lower()} "
            f"{iface.symbol.file_path.lower()}"
        )
        if any(kw in haystack for kw in keywords):
            matched.append(iface)
    return matched


def modules_matching(
    modules: list[Module],
    keywords: set[str],
) -> list[Module]:
    """Return modules whose name or external deps contain any of the keywords."""
    matched: list[Module] = []
    for mod in modules:
        if any(kw in mod.name.lower() for kw in keywords):
            matched.append(mod)
            continue
        for dep in mod.external_deps:
            if any(kw in dep.lower() for kw in keywords):
                matched.append(mod)
                break
    return matched


def pattern_excerpts(patterns: list[DetectedPattern], limit: int = 3) -> list[str]:
    """Return short evidence strings for the top-N patterns."""
    out: list[str] = []
    for pat in patterns[:limit]:
        files = ", ".join(e.file_path for e in pat.evidence[:2])
        suffix = f" at {files}" if files else ""
        out.append(f"Pattern '{pat.display_name}' (confidence={pat.confidence:.0%}){suffix}")
    return out


def interface_excerpts(interfaces: list[Interface], limit: int = 3) -> list[str]:
    """Return short signature excerpts for the top-N interfaces."""
    return [
        f"Interface `{iface.symbol.name}` in {iface.symbol.file_path}"
        for iface in interfaces[:limit]
    ]


def module_excerpts(modules: list[Module], limit: int = 3) -> list[str]:
    """Return short summary lines for the top-N modules."""
    return [
        f"Module '{mod.name}' ({mod.file_count} files, {mod.line_count} lines)"
        for mod in modules[:limit]
    ]


def classify_delta(label: str, count_a: int, count_b: int) -> str:
    """Build a one-line trade-off observation from two counts.

    Used by templates when they want a generic "A has 2x more than B" line.
    Templates may also emit dimension-specific trade-offs of their own.
    """
    if count_a == 0 and count_b == 0:
        return f"Neither repository shows {label} signals."
    if count_a == 0:
        return f"Repo A has no {label} signals; Repo B has {count_b}."
    if count_b == 0:
        return f"Repo A has {count_a} {label} signals; Repo B has none."
    if count_a >= 2 * count_b:
        return f"Repo A has substantially more {label} signals ({count_a} vs {count_b})."
    if count_b >= 2 * count_a:
        return f"Repo B has substantially more {label} signals ({count_b} vs {count_a})."
    return f"Both repositories show comparable {label} coverage ({count_a} vs {count_b})."
