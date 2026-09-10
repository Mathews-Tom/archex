"""Self-contained static export of the explorer's views.

Writes the same view models the loopback server renders into a directory of
sibling HTML files that open directly from a `file://` URL: no server, no
session token, no client-side script, no remote stylesheet, font, or image,
and no query-string routing. Everything a reviewer needs travels in the
directory, so the bundle can be attached to a pull request as a read-only
build artifact and opened offline.

Three properties are load-bearing:

* **Canonical semantics.** Every page is produced by the same
  `archex.explorer.viewmodel` builder and the same `archex.explorer.render`
  renderer the server uses, selected through a non-interactive `LinkScheme`.
  The export adds no second interpretation of the artifact, no parsing, and no
  graph construction.
* **No source body.** `AnalysisArtifactV1` and `ArchGraph` carry no raw source
  text, so neither does the export. It writes exactly what the server would
  show and nothing more, which keeps the redaction contract structural rather
  than dependent on this module getting an escape right.
* **Bounded payload.** A static bundle cannot paginate the way a server can,
  so the node index and the per-node neighborhood pages are capped, each cap
  is reported on the page rather than applied silently, and the per-node pages
  cover the highest-degree nodes -- the ones a reviewer asks about.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.explorer.render import (
    LinkScheme,
    render_diff_page,
    render_health_page,
    render_module_map_page,
    render_neighborhood_page,
    render_node_search_page,
    render_page,
    render_receipt_page,
)
from archex.explorer.viewmodel import (
    build_diff_view,
    build_health_view,
    build_manifest_view,
    build_module_map_view,
    build_neighborhood_view,
    build_node_index_view,
    build_receipt_view,
)
from archex.graph_query import GraphQuery

if TYPE_CHECKING:
    from pathlib import Path

    from archex.explorer.loader import ExplorerData

#: Per-node neighborhood pages written for the highest-degree nodes. A bundle
#: is opened in a browser and carried around as a CI artifact; an unbounded
#: page-per-node export of a large repository is neither reviewable nor safe
#: to attach.
MAX_EXPORT_NODE_PAGES = 200

#: Rows in the exported node index -- one document a reviewer scans with
#: find-in-page, so its cap is independent of the per-node page cap.
MAX_EXPORT_INDEX_ROWS = 1000

INDEX_FILE = "index.html"
MODULES_FILE = "modules.html"
DIFF_FILE = "diff.html"
NODES_FILE = "nodes.html"
NEIGHBORHOOD_FILE = "neighborhood.html"
RECEIPT_FILE = "receipt.html"
HEALTH_FILE = "health.html"

_UNSAFE_SLUG_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


class ExplorerExportError(ValueError):
    """Raised when the export destination cannot be used."""


@dataclass(frozen=True)
class ExportResult:
    """What was written, and what the caps left out."""

    destination: Path
    files: list[str]
    node_pages: int
    node_pages_omitted: int
    index_rows: int
    index_rows_omitted: int


def node_page_name(node_id: str) -> str:
    """A deterministic, filesystem-safe filename for NODE_ID's neighborhood.

    Node ids carry `:`, `#`, and `/`, so the readable part is sanitized for
    orientation while a digest of the full id guarantees uniqueness -- two ids
    that sanitize identically still get separate files.
    """
    readable = _UNSAFE_SLUG_CHARS.sub("-", node_id).strip("-")[:60]
    digest = hashlib.blake2b(node_id.encode("utf-8"), digest_size=6).hexdigest()
    return f"node-{readable}-{digest}.html" if readable else f"node-{digest}.html"


def export_explorer_site(
    data: ExplorerData,
    destination: Path,
    *,
    max_node_pages: int = MAX_EXPORT_NODE_PAGES,
    max_index_rows: int = MAX_EXPORT_INDEX_ROWS,
) -> ExportResult:
    """Write the explorer's views into DESTINATION as offline-openable HTML.

    DESTINATION is treated as owned by the export: any HTML left by an earlier
    export is removed first. Per-node filenames are keyed on the current
    artifact's node ids, so a previous export's pages would otherwise survive
    under different names, unlinked from the regenerated index but still
    openable -- and carrying the earlier artifact's identity and revision into
    a bundle attached for review of a different change.
    """
    if destination.exists() and not destination.is_dir():
        raise ExplorerExportError(f"export destination {destination} is not a directory")
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.html"):
            stale.unlink()
    except OSError as exc:
        raise ExplorerExportError(
            f"cannot prepare export destination {destination}: {exc}"
        ) from exc

    graph_query = GraphQuery(data.graph) if data.graph is not None else None
    index_view = build_node_index_view(data, limit=max_index_rows, graph_query=graph_query)
    page_node_ids = [match.id for match in index_view.matches[: max(max_node_pages, 0)]]
    node_pages = {node_id: node_page_name(node_id) for node_id in page_node_ids}
    total_nodes = len(index_view.matches) + index_view.omitted
    pages_omitted = max(total_nodes - len(page_node_ids), 0)

    links = LinkScheme(
        modules=MODULES_FILE,
        diff=DIFF_FILE,
        search=NODES_FILE,
        neighborhood=NEIGHBORHOOD_FILE,
        receipt=RECEIPT_FILE,
        health=HEALTH_FILE,
        interactive=False,
        node_pages=node_pages,
    )
    manifest = build_manifest_view(data)
    written: list[str] = []

    def write(name: str, html: str) -> None:
        try:
            (destination / name).write_text(html, encoding="utf-8")
        except OSError as exc:
            raise ExplorerExportError(f"cannot write {destination / name}: {exc}") from exc
        written.append(name)

    write(
        INDEX_FILE,
        render_page(
            "archex explorer (static export)",
            manifest,
            _index_body(
                links,
                node_pages=len(node_pages),
                node_pages_omitted=pages_omitted,
                index_omitted=index_view.omitted,
            ),
            links=links,
        ),
    )
    write(DIFF_FILE, render_diff_page(manifest, build_diff_view(data), links=links))
    write(MODULES_FILE, render_module_map_page(manifest, build_module_map_view(data), links=links))
    write(RECEIPT_FILE, render_receipt_page(manifest, build_receipt_view(data), links=links))
    write(HEALTH_FILE, render_health_page(manifest, build_health_view(data), links=links))
    write(NODES_FILE, render_node_search_page(manifest, index_view, links=links))
    write(
        NEIGHBORHOOD_FILE,
        render_neighborhood_page(
            manifest,
            build_neighborhood_view(data, None, graph_query=graph_query),
            links=links,
        ),
    )
    for node_id in page_node_ids:
        write(
            node_pages[node_id],
            render_neighborhood_page(
                manifest,
                build_neighborhood_view(data, node_id, graph_query=graph_query),
                links=links,
            ),
        )

    return ExportResult(
        destination=destination,
        files=written,
        node_pages=len(page_node_ids),
        node_pages_omitted=pages_omitted,
        index_rows=len(index_view.matches),
        index_rows_omitted=index_view.omitted,
    )


def _index_body(
    links: LinkScheme,
    *,
    node_pages: int,
    node_pages_omitted: int,
    index_omitted: int,
) -> str:
    entries = "".join(f'<li><a href="{path}">{label}</a></li>' for label, path in links.nav())
    pages_note = (
        f" {node_pages_omitted} further node(s) have no page here at the export's page cap; "
        "run the local server to reach them."
        if node_pages_omitted
        else ""
    )
    index_note = (
        f" {index_omitted} further node(s) are omitted from the node index at its row cap."
        if index_omitted
        else ""
    )
    return (
        "<h2>Views</h2>\n"
        f"<ul>{entries}</ul>\n"
        '<p class="note">Static export: every page here is the same view the local '
        "<code>archex explore</code> server renders, written offline with no script, no "
        "remote reference, and no session token. Node search, direction, depth, and "
        "edge-type filtering are interactive controls and need that server; this bundle "
        f"ships {node_pages} pre-rendered per-node neighborhood page(s) for the "
        f"highest-degree nodes instead.{pages_note}{index_note}</p>"
    )
