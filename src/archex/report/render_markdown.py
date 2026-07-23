"""Markdown rendering for AnalysisArtifactV1.

A pure projection: every value rendered here already exists on the
artifact. This module adds no analysis, no source text, and no network
calls -- only markdown formatting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.report.artifact import AnalysisArtifactV1


def render_markdown(artifact: AnalysisArtifactV1) -> str:
    lines: list[str] = [
        f"# Diff Review: {artifact.source_identity}",
        "",
        *_provenance_section(artifact),
        "",
        *_summary_section(artifact),
        "",
        *_changed_files_section(artifact),
        "",
        *_symbol_candidates_section(artifact),
        "",
        *_path_list_section(
            "Affected Public Interfaces",
            [candidate.path for candidate in artifact.diff.affected_interfaces],
            artifact.diff.affected_interfaces_total,
        ),
        "",
        *_path_list_section(
            "Test Candidates",
            [candidate.path for candidate in artifact.diff.test_candidates],
            artifact.diff.test_candidates_total,
        ),
        "",
        *_path_list_section(
            "Unsupported Files",
            [item.path for item in artifact.diff.unsupported_files],
            artifact.diff.unsupported_files_total,
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _provenance_section(artifact: AnalysisArtifactV1) -> list[str]:
    return [
        "## Provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Schema version | `{artifact.schema_version.value}` |",
        f"| archex version | `{artifact.archex_version}` |",
        f"| Base ref | `{artifact.diff.base_ref}` "
        f"(`{artifact.diff.base_resolved_sha or 'unresolved'}`) |",
        f"| Head revision | `{artifact.diff.head_ref or 'unknown'}` |",
        f"| Index generation | `{artifact.index_generation or 'unknown'}` |",
        f"| Freshness | `{artifact.freshness.value}` |",
        f"| Completeness | `{artifact.completeness.value}` |",
        f"| Confidence | `{artifact.confidence.value}` |",
        f"| Redaction | `{artifact.redaction_mode.value}` |",
        f"| Generated at | `{artifact.generated_at}` |",
    ]


def _summary_section(artifact: AnalysisArtifactV1) -> list[str]:
    diff = artifact.diff
    lines = [
        "## Summary",
        "",
        f"- **Risk:** `{diff.risk_level.value}`",
        f"- **Changed files:** {diff.changed_files_total}",
        f"- **Symbol candidates:** {diff.symbol_candidates_total}",
        f"- **Affected interfaces:** {diff.affected_interfaces_total}",
        f"- **Test candidates:** {diff.test_candidates_total}",
        f"- **Unsupported files:** {diff.unsupported_files_total}",
    ]
    if diff.risk_reasons:
        lines.append(
            f"- **Risk reasons:** {', '.join(f'`{reason}`' for reason in diff.risk_reasons)}"
        )
    if artifact.excluded_counts:
        excluded = ", ".join(
            f"{key}={value}" for key, value in sorted(artifact.excluded_counts.items())
        )
        lines.append(f"- **Excluded:** {excluded}")
    if artifact.unknown_counts:
        unknown = ", ".join(
            f"{key}={value}" for key, value in sorted(artifact.unknown_counts.items())
        )
        lines.append(f"- **Unknown:** {unknown}")
    return lines


def _changed_files_section(artifact: AnalysisArtifactV1) -> list[str]:
    lines = [
        "## Changed Files",
        "",
        "| Status | Path | Hunks | Handle |",
        "| --- | --- | --- | --- |",
    ]
    for change in artifact.diff.changed_files:
        hunks = ", ".join(f"{hunk.start_line}-{hunk.end_line}" for hunk in change.hunks) or "-"
        lines.append(f"| `{change.status}` | `{change.path}` | {hunks} | `{change.handle}` |")
    if not artifact.diff.changed_files:
        lines.append("| - | _none_ | - | - |")
    if artifact.diff.changed_files_total > len(artifact.diff.changed_files):
        remaining = artifact.diff.changed_files_total - len(artifact.diff.changed_files)
        lines.append("")
        lines.append(f"_{remaining} additional changed file(s) omitted; see the JSON artifact._")
    return lines


def _symbol_candidates_section(artifact: AnalysisArtifactV1) -> list[str]:
    lines = [
        "## Symbol Risk Candidates",
        "",
        "| Risk | Confidence | Symbol | File | Lines | Handle |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in artifact.diff.symbol_candidates:
        label = candidate.qualified_name or candidate.symbol_name or "<unnamed>"
        lines.append(
            f"| `{candidate.risk_level}` | `{candidate.confidence.value}` | `{label}` | "
            f"`{candidate.file_path}` | {candidate.start_line}-{candidate.end_line} | "
            f"`{candidate.handle}` |"
        )
    if not artifact.diff.symbol_candidates:
        lines.append("| - | - | _none_ | - | - | - |")
    if artifact.diff.symbol_candidates_total > len(artifact.diff.symbol_candidates):
        remaining = artifact.diff.symbol_candidates_total - len(artifact.diff.symbol_candidates)
        lines.append("")
        lines.append(
            f"_{remaining} additional symbol candidate(s) omitted; see the JSON artifact._"
        )
    return lines


def _path_list_section(title: str, paths: list[str], total: int) -> list[str]:
    lines = [f"## {title}", ""]
    if not paths:
        lines.append("- None")
        return lines
    lines.extend(f"- `{path}`" for path in paths)
    if total > len(paths):
        lines.append(
            f"- _{total - len(paths)} additional "
            f"entr{'y' if total - len(paths) == 1 else 'ies'} omitted_"
        )
    return lines
