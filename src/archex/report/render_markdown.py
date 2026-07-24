"""Markdown + Mermaid rendering for AnalysisArtifactV1.

A pure projection: every value rendered here already exists on the
artifact. This module adds no analysis, no source text, and no network
calls -- only markdown formatting and a bounded Mermaid flowchart sketch
of the diff's changed-file/symbol-candidate structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.report.artifact import AnalysisArtifactV1, SymbolCandidate

# Independent of the artifact's own MAX_* bounds: a Mermaid diagram stays
# readable only at a much smaller scale than the canonical JSON payload.
MAX_MERMAID_FILE_NODES = 20
MAX_MERMAID_SYMBOL_NODES = 40

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def render_markdown(artifact: AnalysisArtifactV1) -> str:
    lines: list[str] = [
        f"# Diff Review: {artifact.source_identity}",
        "",
        *_provenance_section(artifact),
        "",
        *_summary_section(artifact),
        "",
        *_mermaid_section(artifact),
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
        *_test_candidates_section(artifact),
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
        "| Risk | Confidence | Symbol | File | Lines | Handle | Runtime evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in artifact.diff.symbol_candidates:
        label = candidate.qualified_name or candidate.symbol_name or "<unnamed>"
        lines.append(
            f"| `{candidate.risk_level}` | `{candidate.confidence.value}` | `{label}` | "
            f"`{candidate.file_path}` | {candidate.start_line}-{candidate.end_line} | "
            f"`{candidate.handle}` | {_runtime_evidence_label(candidate)} |"
        )
    if not artifact.diff.symbol_candidates:
        lines.append("| - | - | _none_ | - | - | - | - |")
    if artifact.diff.symbol_candidates_total > len(artifact.diff.symbol_candidates):
        remaining = artifact.diff.symbol_candidates_total - len(artifact.diff.symbol_candidates)
        lines.append("")
        lines.append(
            f"_{remaining} additional symbol candidate(s) omitted; see the JSON artifact._"
        )
    return lines


def _runtime_evidence_label(candidate: SymbolCandidate) -> str:
    if candidate.runtime_sample_count is None:
        return "-"
    revision = (candidate.runtime_revision or "unknown")[:8]
    stale = " **STALE**" if candidate.runtime_stale else ""
    return f"{candidate.runtime_sample_count} samples @ `{revision}`{stale}"


def _test_candidates_section(artifact: AnalysisArtifactV1) -> list[str]:
    lines = [
        "## Test Candidates",
        "",
        "| Path | Handle | Coverage | Revision | Stale |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate in artifact.diff.test_candidates:
        coverage = "-"
        if candidate.coverage_line_rate is not None:
            coverage = f"{candidate.coverage_line_rate:.2f}"
        revision = f"`{candidate.coverage_revision[:8]}`" if candidate.coverage_revision else "-"
        stale = "yes" if candidate.coverage_stale else "no"
        lines.append(
            f"| `{candidate.path}` | `{candidate.handle}` | {coverage} | {revision} | {stale} |"
        )
    if not artifact.diff.test_candidates:
        lines.append("| _none_ | - | - | - | - |")
    if artifact.diff.test_candidates_total > len(artifact.diff.test_candidates):
        remaining = artifact.diff.test_candidates_total - len(artifact.diff.test_candidates)
        lines.append("")
        lines.append(f"_{remaining} additional test candidate(s) omitted; see the JSON artifact._")
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


def _mermaid_section(artifact: AnalysisArtifactV1) -> list[str]:
    diagram = render_mermaid(artifact)
    if diagram is None:
        return ["## Structure", "", "_No changed files to diagram._"]
    return ["## Structure", "", "```mermaid", diagram, "```"]


def render_mermaid(artifact: AnalysisArtifactV1) -> str | None:
    """Render a bounded flowchart of changed files and their symbol risk candidates.

    A pure sketch of data already on the artifact -- no graph edges are
    constructed here; `-->` links only mirror each symbol candidate's
    already-known `file_path`.
    """
    changed_files = artifact.diff.changed_files[:MAX_MERMAID_FILE_NODES]
    if not changed_files:
        return None

    candidates_by_file: dict[str, list[SymbolCandidate]] = {}
    for candidate in artifact.diff.symbol_candidates:
        candidates_by_file.setdefault(candidate.file_path, []).append(candidate)

    lines = [
        "flowchart TD",
        "    classDef high fill:#f66,stroke:#900,color:#200",
        "    classDef medium fill:#fc6,stroke:#960,color:#320",
        "    classDef low fill:#9c6,stroke:#360,color:#130",
        "    classDef unknown fill:#eee,stroke:#999,color:#333",
    ]

    node_ids: dict[str, str] = {}
    symbol_node_count = 0

    def _node_id(key: str) -> str:
        if key not in node_ids:
            node_ids[key] = f"n{len(node_ids)}"
        return node_ids[key]

    for change in changed_files:
        file_key = f"file:{change.path}"
        file_id = _node_id(file_key)
        file_risk = _max_risk(candidates_by_file.get(change.path, []))
        lines.append(f'    {file_id}["{_escape_label(change.path)}"]:::{file_risk}')

        for candidate in candidates_by_file.get(change.path, []):
            if symbol_node_count >= MAX_MERMAID_SYMBOL_NODES:
                break
            symbol_key = f"symbol:{candidate.handle}"
            symbol_id = _node_id(symbol_key)
            label = candidate.qualified_name or candidate.symbol_name or "<unnamed>"
            lines.append(f'    {symbol_id}["{_escape_label(label)}"]:::{candidate.risk_level}')
            lines.append(f"    {file_id} --> {symbol_id}")
            symbol_node_count += 1

    omitted_files = artifact.diff.changed_files_total - len(changed_files)
    if omitted_files > 0:
        note_id = _node_id("note:omitted")
        lines.append(f'    {note_id}["+{omitted_files} more changed file(s)"]:::unknown')

    return "\n".join(lines)


def _max_risk(candidates: list[SymbolCandidate]) -> str:
    if not candidates:
        return "unknown"
    return max(
        (candidate.risk_level for candidate in candidates),
        key=lambda level: _RISK_ORDER.get(level, 0),
    )


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "&quot;").replace("\n", " ")
