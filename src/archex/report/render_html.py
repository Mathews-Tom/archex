"""Static, offline HTML rendering for AnalysisArtifactV1.

A pure projection, like the Markdown renderer: every value rendered here
already exists on the artifact. No source parsing, no analysis, no network
calls -- and, unlike Markdown, no client-side script either. The Mermaid
diagram is embedded as its plain-text source inside a `<pre>` block rather
than rendered as an interactive diagram: rendering it client-side would
require vendoring or fetching the Mermaid JS library, which conflicts with
staying offline and script-free. The page is a single self-contained HTML
document: inline CSS only, zero `<script>` tags, zero remote references
(no CDN stylesheets, fonts, or images), so it opens correctly from a local
file:// URL with no network access.

Rows are capped independently of (and tighter than) the artifact's own
MAX_* bounds and the Markdown renderer's full-list display: a static page
meant to be opened directly needs a much smaller size budget than the
canonical JSON payload.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING
from urllib.parse import quote

from archex.report.render_markdown import render_mermaid

if TYPE_CHECKING:
    from archex.report.artifact import AnalysisArtifactV1, DiffFileChange, SymbolCandidate

MAX_HTML_ROWS = 50

_STYLE = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 2rem; color: #1a1a1a; background: #fff; }
    h1 { font-size: 1.4rem; }
    h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd;
         padding-bottom: .25rem; }
    table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .85rem; }
    th, td { border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; }
    th { background: #f4f4f4; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    code { background: #f4f4f4; padding: .1rem .3rem; border-radius: 3px; }
    pre { background: #f7f7f7; padding: .75rem; overflow-x: auto; border-radius: 4px; }
    .risk-high { color: #900; font-weight: 600; }
    .risk-medium, .risk-moderate { color: #960; font-weight: 600; }
    .risk-low { color: #360; }
    .note { color: #666; font-style: italic; font-size: .85rem; }
    .file-link { color: #0b57d0; text-decoration: none; }
    .file-link:hover { text-decoration: underline; }
""".strip()


def render_html(artifact: AnalysisArtifactV1) -> str:
    title = f"Diff Review: {escape(artifact.source_identity)}"
    body = [
        f"<h1>{title}</h1>",
        _provenance_table(artifact),
        _summary_list(artifact),
        _structure_block(artifact),
        _changed_files_table(artifact),
        _symbol_candidates_table(artifact),
        _path_list(
            "Affected Public Interfaces",
            [c.path for c in artifact.diff.affected_interfaces],
            artifact.diff.affected_interfaces_total,
            artifact.source_root,
        ),
        _path_list(
            "Test Candidates",
            [c.path for c in artifact.diff.test_candidates],
            artifact.diff.test_candidates_total,
            artifact.source_root,
        ),
        _path_list(
            "Unsupported Files",
            [item.path for item in artifact.diff.unsupported_files],
            artifact.diff.unsupported_files_total,
            artifact.source_root,
        ),
    ]
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n" + "\n".join(body) + "\n</body>\n</html>\n"
    )


def _provenance_table(artifact: AnalysisArtifactV1) -> str:
    rows = [
        ("Schema version", artifact.schema_version.value),
        ("archex version", artifact.archex_version),
        (
            "Base ref",
            f"{artifact.diff.base_ref} ({artifact.diff.base_resolved_sha or 'unresolved'})",
        ),
        ("Head revision", artifact.diff.head_ref or "unknown"),
        ("Index generation", artifact.index_generation or "unknown"),
        ("Freshness", artifact.freshness.value),
        ("Completeness", artifact.completeness.value),
        ("Confidence", artifact.confidence.value),
        ("Redaction", artifact.redaction_mode.value),
        ("Generated at", artifact.generated_at),
    ]
    body_rows = "".join(
        f"<tr><td>{escape(k)}</td><td><code>{escape(v)}</code></td></tr>" for k, v in rows
    )
    return f"<h2>Provenance</h2>\n<table>{body_rows}</table>"


def _summary_list(artifact: AnalysisArtifactV1) -> str:
    diff = artifact.diff
    items = [
        f"Risk: <span class='risk-{escape(diff.risk_level.value)}'>"
        f"{escape(diff.risk_level.value)}</span>",
        f"Changed files: {diff.changed_files_total}",
        f"Symbol candidates: {diff.symbol_candidates_total}",
        f"Affected interfaces: {diff.affected_interfaces_total}",
        f"Test candidates: {diff.test_candidates_total}",
        f"Unsupported files: {diff.unsupported_files_total}",
    ]
    if diff.risk_reasons:
        reasons = ", ".join(f"<code>{escape(r)}</code>" for r in diff.risk_reasons)
        items.append(f"Risk reasons: {reasons}")
    if artifact.excluded_counts:
        excluded = ", ".join(
            f"{escape(k)}={v}" for k, v in sorted(artifact.excluded_counts.items())
        )
        items.append(f"Excluded: {excluded}")
    if artifact.unknown_counts:
        unknown = ", ".join(f"{escape(k)}={v}" for k, v in sorted(artifact.unknown_counts.items()))
        items.append(f"Unknown: {unknown}")
    list_items = "".join(f"<li>{item}</li>" for item in items)
    return f"<h2>Summary</h2>\n<ul>{list_items}</ul>"


def _structure_block(artifact: AnalysisArtifactV1) -> str:
    diagram = render_mermaid(artifact)
    if diagram is None:
        return "<h2>Structure</h2>\n<p class='note'>No changed files to diagram.</p>"
    return f"<h2>Structure</h2>\n<pre>{escape(diagram)}</pre>"


def _changed_files_table(artifact: AnalysisArtifactV1) -> str:
    changes = artifact.diff.changed_files[:MAX_HTML_ROWS]
    header = "<tr><th>Status</th><th>Path</th><th>Hunks</th><th>Handle</th></tr>"
    rows = (
        "".join(_changed_file_row(change, artifact.source_root) for change in changes)
        or "<tr><td colspan='4'><em>None</em></td></tr>"
    )
    note = _truncation_note(artifact.diff.changed_files_total, len(changes))
    return f"<h2>Changed Files</h2>\n<table>{header}{rows}</table>{note}"


def _changed_file_row(change: DiffFileChange, source_root: str) -> str:
    hunks = ", ".join(f"{h.start_line}-{h.end_line}" for h in change.hunks) or "-"
    first_line = change.hunks[0].start_line if change.hunks else None
    path_link = _editor_link(source_root, change.path, first_line, escape(change.path))
    return (
        f"<tr><td><code>{escape(change.status)}</code></td>"
        f"<td><code>{path_link}</code></td>"
        f"<td>{escape(hunks)}</td>"
        f"<td><code>{escape(change.handle)}</code></td></tr>"
    )


def _symbol_candidates_table(artifact: AnalysisArtifactV1) -> str:
    candidates = artifact.diff.symbol_candidates[:MAX_HTML_ROWS]
    header = (
        "<tr><th>Risk</th><th>Confidence</th><th>Symbol</th>"
        "<th>File</th><th>Lines</th><th>Handle</th></tr>"
    )
    rows = (
        "".join(_symbol_candidate_row(candidate, artifact.source_root) for candidate in candidates)
        or "<tr><td colspan='6'><em>None</em></td></tr>"
    )
    note = _truncation_note(artifact.diff.symbol_candidates_total, len(candidates))
    return f"<h2>Symbol Risk Candidates</h2>\n<table>{header}{rows}</table>{note}"


def _symbol_candidate_row(candidate: SymbolCandidate, source_root: str) -> str:
    label = candidate.qualified_name or candidate.symbol_name or "<unnamed>"
    risk = escape(candidate.risk_level)
    file_link = _editor_link(
        source_root, candidate.file_path, candidate.start_line, escape(candidate.file_path)
    )
    return (
        f"<tr><td class='risk-{risk}'>{risk}</td>"
        f"<td>{escape(candidate.confidence.value)}</td>"
        f"<td><code>{escape(label)}</code></td>"
        f"<td><code>{file_link}</code></td>"
        f"<td>{candidate.start_line}-{candidate.end_line}</td>"
        f"<td><code>{escape(candidate.handle)}</code></td></tr>"
    )


def _path_list(title: str, paths: list[str], total: int, source_root: str) -> str:
    shown = paths[:MAX_HTML_ROWS]
    if not shown:
        return f"<h2>{escape(title)}</h2>\n<p class='note'>None</p>"
    items = "".join(
        f"<li><code>{_editor_link(source_root, path, None, escape(path))}</code></li>"
        for path in shown
    )
    note = _truncation_note(total, len(shown))
    return f"<h2>{escape(title)}</h2>\n<ul>{items}</ul>{note}"


def _editor_link(source_root: str, path: str, line: int | None, inner_html: str) -> str:
    """Build an offline, local-only `vscode://file/` link for `path` (optionally `:line`).

    Purely a client-side URI scheme resolved by the local machine's editor,
    if any is registered for it; never a network request. Preserves the
    same path/line identity every renderer already carries -- clicking it
    requires nothing from archex itself.
    """
    absolute = f"{source_root.rstrip('/')}/{path}"
    location = f"{absolute}:{line}" if line is not None else absolute
    href = f"vscode://file/{quote(location, safe='/:')}"
    return f'<a class="file-link" href="{href}">{inner_html}</a>'


def _truncation_note(total: int, shown: int) -> str:
    if total <= shown:
        return ""
    remaining = total - shown
    return (
        f"<p class='note'>{remaining} additional "
        f"entr{'y' if remaining == 1 else 'ies'} omitted; see the JSON artifact.</p>"
    )
