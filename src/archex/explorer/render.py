"""Server-rendered HTML for the local explorer.

Every page is a single self-contained HTML document with an inline
`<style>` block -- no external stylesheet, script, font, or CDN request, and
no client-side JavaScript at all (navigation and the neighborhood search use
plain `<a href>` links and `<form method="get">` submissions). This keeps
every response fully offline and lets the Content-Security-Policy the server
attaches (see `archex.explorer.security`) block `script-src` entirely.

Every value written into a page is escaped with `html.escape`; nothing here
ever emits raw source text (`AnalysisArtifactV1` and `ArchGraph` never carry
any).
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.explorer.viewmodel import (
        DiffView,
        HealthView,
        ManifestView,
        ModuleMapView,
        NeighborhoodView,
        ReceiptView,
    )

NAV_ITEMS: list[tuple[str, str]] = [
    ("Module Map", "/view/modules"),
    ("Diff Review", "/view/diff"),
    ("Target Neighborhood", "/view/neighborhood"),
    ("Receipt Inspector", "/view/receipt"),
    ("Index Health", "/view/health"),
]

_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 0; color: #1a1a1a; background: #fafafa; }
header.banner { background: #1a1a2e; color: #eee; padding: 0.75rem 1.25rem; }
header.banner h1 { margin: 0 0 0.25rem 0; font-size: 1.1rem; }
header.banner .meta { font-size: 0.8rem; color: #b8b8d0; }
nav { background: #16213e; padding: 0.5rem 1.25rem; }
nav a { color: #e0e0f0; text-decoration: none; margin-right: 1.25rem; font-size: 0.85rem; }
nav a:hover { text-decoration: underline; }
main { padding: 1.25rem; max-width: 960px; margin: 0 auto; }
h2 { font-size: 1.05rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.85rem; }
th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #e5e5e5; }
th { background: #f0f0f5; }
.badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px;
         font-size: 0.75rem; background: #e0e0e0; }
.badge-high, .badge-critical { background: #f8d7da; color: #842029; }
.badge-medium { background: #fff3cd; color: #664d03; }
.badge-low { background: #d1e7dd; color: #0f5132; }
.note { color: #666; font-size: 0.8rem; font-style: italic; }
.empty { color: #666; font-style: italic; }
""".strip()


def render_page(
    title: str,
    manifest: ManifestView,
    body: str,
    *,
    nav: list[tuple[str, str]] | None = None,
) -> str:
    nav_items = NAV_ITEMS if nav is None else nav
    nav_html = " ".join(
        f'<a href="{escape(path)}">{escape(label)}</a>' for label, path in nav_items
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n<title>{escape(title)}</title>\n'
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"{_manifest_banner(manifest)}\n"
        f"<nav>{nav_html}</nav>\n"
        f"<main>\n{body}\n</main>\n"
        "</body>\n</html>\n"
    )


def _manifest_banner(manifest: ManifestView) -> str:
    return (
        '<header class="banner">'
        f"<h1>{escape(manifest.source_identity)} @ {escape(manifest.source_revision)}</h1>"
        '<div class="meta">'
        f"generated {escape(manifest.generated_at)} by archex {escape(manifest.archex_version)} "
        f"(schema {escape(manifest.schema_version)}) &middot; "
        f"freshness: {escape(manifest.freshness)} &middot; "
        f"completeness: {escape(manifest.completeness)} &middot; "
        f"confidence: {escape(manifest.confidence)} &middot; "
        f"redaction: {escape(manifest.redaction_mode)} &middot; "
        f"evidence: {manifest.evidence_count} &middot; "
        f"excluded: {manifest.excluded_total} &middot; "
        f"unknown: {manifest.unknown_total} &middot; "
        f"graph: {'attached' if manifest.has_graph else 'not provided'}"
        "</div></header>"
    )


def render_diff_page(manifest: ManifestView, view: DiffView) -> str:
    body = "".join(
        [
            _diff_summary(view),
            _changed_files_table(view),
            _symbol_candidates_table(view),
            _interface_candidates_table(view),
            _test_candidates_table(view),
            _unsupported_files_list(view),
        ]
    )
    return render_page(f"Diff Review: {view.base_ref}", manifest, body)


def _risk_badge(level: str) -> str:
    return f'<span class="badge badge-{escape(level.lower())}">{escape(level)}</span>'


def _diff_summary(view: DiffView) -> str:
    reasons = (
        "".join(f"<li>{escape(reason)}</li>" for reason in view.risk_reasons)
        or '<li class="empty">no risk reasons recorded</li>'
    )
    return (
        "<h2>Summary</h2>"
        "<ul>"
        f"<li>Base: <code>{escape(view.base_ref)}</code>"
        f" ({escape(view.base_resolved_sha) or 'unresolved'})</li>"
        f"<li>Head: <code>{escape(view.head_ref) or 'working tree'}</code></li>"
        f"<li>Risk level: {_risk_badge(view.risk_level)}</li>"
        f"<li>Changed files: {view.changed_files_total}</li>"
        f"<li>Symbol candidates: {view.symbol_candidates_total}</li>"
        f"<li>Affected interfaces: {view.affected_interfaces_total}</li>"
        f"<li>Test candidates: {view.test_candidates_total}</li>"
        f"<li>Unsupported files: {view.unsupported_files_total}</li>"
        "</ul>"
        f"<h2>Risk reasons</h2><ul>{reasons}</ul>"
    )


def _changed_files_table(view: DiffView) -> str:
    if not view.changed_files:
        return '<h2>Changed files</h2><p class="empty">none</p>'
    rows = "".join(
        "<tr>"
        f"<td>{escape(row.path)}</td><td>{escape(row.status)}</td>"
        f"<td>{escape(row.old_path or '-')}</td>"
        f"<td>{', '.join(f'{h.start_line}-{h.end_line}' for h in row.hunks) or '-'}</td>"
        f"<td><code>{escape(row.handle)}</code></td>"
        "</tr>"
        for row in view.changed_files
    )
    note = _truncation_note(view.changed_files_total, len(view.changed_files))
    return (
        "<h2>Changed files</h2>"
        "<table><tr><th>Path</th><th>Status</th><th>Old path</th>"
        "<th>Hunks</th><th>Handle</th></tr>" + rows + "</table>" + note
    )


def _symbol_candidates_table(view: DiffView) -> str:
    if not view.symbol_candidates:
        return '<h2>Symbol risk candidates</h2><p class="empty">none</p>'
    rows = "".join(
        "<tr>"
        f"<td>{escape(row.label)}</td><td>{escape(row.file_path)}:{row.start_line}-{row.end_line}</td>"
        f"<td>{_risk_badge(row.risk_level)}</td><td>{escape(row.confidence)}</td>"
        f"<td>{', '.join(escape(s) for s in row.signals) or '-'}</td>"
        f"<td><code>{escape(row.handle)}</code></td>"
        "</tr>"
        for row in view.symbol_candidates
    )
    note = _truncation_note(view.symbol_candidates_total, len(view.symbol_candidates))
    return (
        "<h2>Symbol risk candidates</h2>"
        "<table><tr><th>Symbol</th><th>Location</th><th>Risk</th>"
        "<th>Confidence</th><th>Signals</th><th>Handle</th></tr>" + rows + "</table>" + note
    )


def _interface_candidates_table(view: DiffView) -> str:
    if not view.affected_interfaces:
        return '<h2>Affected interfaces</h2><p class="empty">none</p>'
    rows = "".join(
        f"<tr><td>{escape(row.path)}</td><td>{escape(row.symbol_id)}</td>"
        f"<td>{escape(row.confidence)}</td><td><code>{escape(row.handle)}</code></td></tr>"
        for row in view.affected_interfaces
    )
    note = _truncation_note(view.affected_interfaces_total, len(view.affected_interfaces))
    return (
        "<h2>Affected interfaces</h2>"
        "<table><tr><th>Path</th><th>Symbol</th><th>Confidence</th>"
        "<th>Handle</th></tr>" + rows + "</table>" + note
    )


def _test_candidates_table(view: DiffView) -> str:
    if not view.test_candidates:
        return '<h2>Test candidates</h2><p class="empty">none</p>'
    rows = "".join(
        f"<tr><td>{escape(row.path)}</td><td>{escape(row.reason)}</td>"
        f"<td>{escape(row.confidence)}</td><td><code>{escape(row.handle)}</code></td></tr>"
        for row in view.test_candidates
    )
    note = _truncation_note(view.test_candidates_total, len(view.test_candidates))
    return (
        "<h2>Test candidates</h2>"
        "<table><tr><th>Path</th><th>Reason</th><th>Confidence</th>"
        "<th>Handle</th></tr>" + rows + "</table>" + note
    )


def _unsupported_files_list(view: DiffView) -> str:
    if not view.unsupported_files:
        return '<h2>Unsupported files</h2><p class="empty">none</p>'
    items = "".join(
        f"<li>{escape(row.path)} ({escape(row.reason)})</li>" for row in view.unsupported_files
    )
    note = _truncation_note(view.unsupported_files_total, len(view.unsupported_files))
    return f"<h2>Unsupported files</h2><ul>{items}</ul>{note}"


def _truncation_note(total: int, shown: int) -> str:
    if total <= shown:
        return ""
    return f'<p class="note">Showing {shown} of {total}.</p>'


def render_error_page(status: int, title: str, message: str) -> str:
    """A minimal, data-free error page for 400/403/404/405 responses."""
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n<title>{status} {escape(title)}</title>\n'
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"<main><h2>{status} {escape(title)}</h2><p>{escape(message)}</p></main>\n"
        "</body>\n</html>\n"
    )


def render_module_map_page(manifest: ManifestView, view: ModuleMapView) -> str:
    if not view.available:
        body = (
            "<h2>Module Map</h2>"
            '<p class="empty">No graph artifact provided. '
            "Pass <code>--graph</code> (an <code>archex graph export</code> output) "
            "to <code>archex explore</code> to enable this view.</p>"
        )
        return render_page("Module Map", manifest, body)

    if not view.modules:
        body = '<h2>Module Map</h2><p class="empty">The graph has no nodes.</p>'
        return render_page("Module Map", manifest, body)

    rows = "".join(
        "<tr>"
        f"<td>{escape(row.module)}</td><td>{row.node_count}</td><td>{row.file_count}</td>"
        f"<td>{row.symbol_count}</td><td>{row.interface_count}</td>"
        "</tr>"
        for row in view.modules
    )
    note = _truncation_note(view.modules_total, len(view.modules))
    body = (
        "<h2>Module Map</h2>"
        '<p class="note">Module-aggregated node counts (not a per-node graph). '
        "Look up a module's file or symbol under Target Neighborhood.</p>"
        "<table><tr><th>Module</th><th>Nodes</th><th>Files</th>"
        "<th>Symbols</th><th>Interfaces</th></tr>" + rows + "</table>" + note
    )
    return render_page("Module Map", manifest, body)


def render_receipt_page(manifest: ManifestView, view: ReceiptView) -> str:
    evidence_rows_html = "".join(
        "<tr>"
        f"<td>{escape(item.path)}</td>"
        f"<td>{item.start_line if item.start_line is not None else '-'}</td>"
        f"<td>{item.end_line if item.end_line is not None else '-'}</td>"
        f"<td>{escape(item.handle or '-')}</td>"
        f"<td>{escape(item.description)}</td>"
        "</tr>"
        for item in view.evidence_locations
    )
    evidence_note = _truncation_note(view.evidence_locations_total, len(view.evidence_locations))
    excluded_rows = (
        "".join(
            f"<li>{escape(key)}: {value}</li>"
            for key, value in sorted(view.excluded_counts.items())
        )
        or '<li class="empty">none</li>'
    )
    unknown_rows = (
        "".join(
            f"<li>{escape(key)}: {value}</li>" for key, value in sorted(view.unknown_counts.items())
        )
        or '<li class="empty">none</li>'
    )
    body = (
        "<h2>Receipt</h2>"
        "<ul>"
        f"<li>Freshness: {escape(view.freshness)}</li>"
        f"<li>Completeness: {escape(view.completeness)}</li>"
        f"<li>Confidence: {escape(view.confidence)}</li>"
        f"<li>Redaction mode: {escape(view.redaction_mode)}</li>"
        f"<li>Generated at: {escape(view.generated_at)}</li>"
        "</ul>"
        "<h2>Excluded counts</h2><ul>" + excluded_rows + "</ul>"
        "<h2>Unknown counts</h2><ul>" + unknown_rows + "</ul>"
        "<h2>Evidence locations</h2>"
        + (
            "<table><tr><th>Path</th><th>Start</th><th>End</th>"
            "<th>Handle</th><th>Description</th></tr>"
            + evidence_rows_html
            + "</table>"
            + evidence_note
            if view.evidence_locations
            else '<p class="empty">none</p>'
        )
    )
    return render_page("Receipt Inspector", manifest, body)


def render_health_page(manifest: ManifestView, view: HealthView) -> str:
    parser_rows = (
        "".join(
            f"<li>{escape(language)}: {escape(version)}</li>"
            for language, version in sorted(view.parser_versions.items())
        )
        or '<li class="empty">none recorded</li>'
    )
    body = (
        "<h2>Index Health</h2>"
        "<ul>"
        f"<li>archex version: {escape(view.archex_version)}</li>"
        f"<li>Artifact schema version: {escape(view.schema_version)}</li>"
        f"<li>Index generation: {escape(view.index_generation)}</li>"
        f"<li>Index schema version: {escape(view.index_schema_version)}</li>"
        f"<li>Chunker revision: {escape(view.chunker_revision)}</li>"
        f"<li>Retrieval profile: {escape(view.retrieval_profile or 'default')}</li>"
        f"<li>Config fingerprint: {escape(view.config_fingerprint)}</li>"
        f"<li>Working tree fingerprint: {escape(view.working_tree_fingerprint)}</li>"
        f"<li>Producer: {escape(view.producer)} {escape(view.producer_version)}</li>"
        "</ul>"
        "<h2>Parser versions</h2><ul>" + parser_rows + "</ul>"
    )
    return render_page("Index Health", manifest, body)


def render_neighborhood_page(manifest: ManifestView, view: NeighborhoodView) -> str:
    form = (
        '<form method="get" action="/view/neighborhood">'
        f'<input type="text" name="node" placeholder="file or symbol id" '
        f'value="{escape(view.query or "")}">'
        '<select name="direction">'
        + "".join(
            f'<option value="{d}"{" selected" if d == view.direction else ""}>{d}</option>'
            for d in ("both", "out", "in")
        )
        + "</select>"
        f'<input type="number" name="depth" min="1" value="{view.depth}">'
        f'<input type="number" name="limit" min="1" value="{view.limit}">'
        '<button type="submit">Find neighbors</button>'
        "</form>"
    )
    if not view.available:
        body = (
            "<h2>Target Neighborhood</h2>" + form + '<p class="empty">No graph artifact provided. '
            "Pass <code>--graph</code> to <code>archex explore</code> to enable this view.</p>"
        )
        return render_page("Target Neighborhood", manifest, body)

    if view.error is not None:
        body = "<h2>Target Neighborhood</h2>" + form + f'<p class="empty">{escape(view.error)}</p>'
        return render_page("Target Neighborhood", manifest, body)

    if view.seed is None:
        body = (
            "<h2>Target Neighborhood</h2>"
            + form
            + '<p class="empty">Enter a file path or symbol id to see its bounded '
            "neighborhood.</p>"
        )
        return render_page("Target Neighborhood", manifest, body)

    node_rows = "".join(
        f"<tr><td>{escape(node.id)}</td><td>{escape(node.type)}</td>"
        f"<td>{escape(node.label)}</td><td>{node.degree}</td></tr>"
        for node in view.nodes
    )
    edge_rows = "".join(
        f"<tr><td>{escape(edge.source_id)}</td><td>{escape(edge.type)}</td>"
        f"<td>{escape(edge.target_id)}</td><td>{escape(edge.confidence)}</td></tr>"
        for edge in view.edges
    )
    hub_rows = "".join(f"<li>{escape(hub.id)}</li>" for hub in view.hubs) or (
        '<li class="empty">none</li>'
    )
    truncation = (
        f'<p class="note">Truncated: {view.omitted_edges} edges omitted at limit {view.limit}.</p>'
        if view.truncated
        else ""
    )
    body = (
        "<h2>Target Neighborhood</h2>" + form + f"<p>Seed: <code>{escape(view.seed.id)}</code> "
        f"({escape(view.seed.type)}, degree {view.seed.degree}) &middot; "
        f"direction: {escape(view.direction)} &middot; depth: {view.depth}</p>"
        + truncation
        + "<h3>Nodes</h3>"
        "<table><tr><th>ID</th><th>Type</th><th>Label</th><th>Degree</th></tr>"
        + node_rows
        + "</table>"
        "<h3>Edges</h3>"
        "<table><tr><th>Source</th><th>Type</th><th>Target</th><th>Confidence</th></tr>"
        + edge_rows
        + "</table>"
        "<h3>Hubs skipped (high fan-out)</h3><ul>" + hub_rows + "</ul>"
    )
    return render_page("Target Neighborhood", manifest, body)
