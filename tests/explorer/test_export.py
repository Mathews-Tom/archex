"""Tests for the self-contained static explorer export.

The properties that matter for a bundle attached to a pull request and opened
from a `file://` URL: it must be openable with no server, carry no script and
no remote reference, expose no session token, keep the canonical artifact
semantics the loopback server renders, leak no source body, and stay bounded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from archex.explorer.export import (
    DIFF_FILE,
    HEALTH_FILE,
    INDEX_FILE,
    MODULES_FILE,
    NEIGHBORHOOD_FILE,
    NODES_FILE,
    RECEIPT_FILE,
    ExplorerExportError,
    export_explorer_site,
    node_page_name,
)
from archex.explorer.loader import ExplorerData
from archex.explorer.viewmodel import build_neighborhood_view, build_node_index_view
from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
)
from archex.report.artifact import (
    AnalysisArtifactV1,
    DiffAnalysis,
    DiffFileChange,
    ReportSchemaVersion,
)

_REMOTE_REF = re.compile(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""")


def _artifact() -> AnalysisArtifactV1:
    return AnalysisArtifactV1(
        schema_version=ReportSchemaVersion(),
        archex_version="0.28.0",
        generated_at="2026-09-10T00:00:00Z",
        source_identity="acme/widget",
        source_root="/repo",
        source_revision="deadbeef",
        working_tree_fingerprint="fp",
        index_generation="gen1",
        index_schema_version="1",
        chunker_revision="c1",
        config_fingerprint="cfg1",
        diff=DiffAnalysis(
            base_ref="main",
            changed_files=[DiffFileChange(path="hub.py", status="M", handle="file:hub.py")],
            changed_files_total=1,
        ),
    )


def _graph(extra_nodes: int = 0) -> ArchGraph:
    nodes = [
        GraphNode(id="file:hub.py", type=GraphNodeType.FILE, label="hub.py", module="pkg"),
        GraphNode(id="file:leaf.py", type=GraphNodeType.FILE, label="leaf.py", module="pkg"),
        GraphNode(
            id="symbol:hub.py::helper#function",
            type=GraphNodeType.SYMBOL,
            label="helper",
            module="pkg",
        ),
    ]
    nodes.extend(
        GraphNode(
            id=f"file:filler_{index}.py",
            type=GraphNodeType.FILE,
            label=f"filler_{index}.py",
            module="pkg",
        )
        for index in range(extra_nodes)
    )
    return ArchGraph(
        project=GraphProject(name="widget", total_files=2 + extra_nodes),
        metadata=GraphExportMetadata(archex_version="0.28.0"),
        nodes=nodes,
        edges=[
            GraphEdge(source="file:leaf.py", target="file:hub.py", type=GraphEdgeType.IMPORTS),
            GraphEdge(
                source="file:hub.py",
                target="symbol:hub.py::helper#function",
                type=GraphEdgeType.CONTAINS,
            ),
        ],
    )


def _data(extra_nodes: int = 0) -> ExplorerData:
    return ExplorerData(artifact=_artifact(), graph=_graph(extra_nodes))


def _pages(destination: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in destination.glob("*.html")}


def test_export_writes_every_view_as_a_sibling_file(tmp_path: Path) -> None:
    result = export_explorer_site(_data(), tmp_path / "site")

    written = set(result.files)
    for name in (
        INDEX_FILE,
        DIFF_FILE,
        MODULES_FILE,
        NODES_FILE,
        NEIGHBORHOOD_FILE,
        RECEIPT_FILE,
        HEALTH_FILE,
    ):
        assert name in written
        assert (result.destination / name).is_file()


def test_exported_pages_carry_no_script_no_remote_reference_and_no_token(tmp_path: Path) -> None:
    export_explorer_site(_data(), tmp_path / "site")

    for name, html in _pages(tmp_path / "site").items():
        assert "<script" not in html, name
        assert _REMOTE_REF.search(html) is None, name
        assert "token=" not in html, name
        assert "archex_session" not in html, name


def test_exported_links_are_relative_so_the_bundle_opens_over_file_urls(tmp_path: Path) -> None:
    export_explorer_site(_data(), tmp_path / "site")
    destination = tmp_path / "site"

    for name, html in _pages(destination).items():
        for href in re.findall(r'href="([^"]+)"', html):
            assert not href.startswith("/"), f"{name} links to absolute path {href}"
            assert "?" not in href, f"{name} links to a query string {href}"
            assert (destination / href).is_file(), f"{name} links to missing {href}"


def test_exported_pages_omit_controls_no_server_can_answer(tmp_path: Path) -> None:
    export_explorer_site(_data(), tmp_path / "site")

    for name, html in _pages(tmp_path / "site").items():
        assert "<form" not in html, name
    index = (tmp_path / "site" / NODES_FILE).read_text(encoding="utf-8")
    assert "Incremental search needs the local" in index


def test_export_node_pages_cover_the_highest_degree_nodes(tmp_path: Path) -> None:
    result = export_explorer_site(_data(), tmp_path / "site")

    hub_page = tmp_path / "site" / node_page_name("file:hub.py")
    assert hub_page.is_file()
    assert result.node_pages == 3
    html = hub_page.read_text(encoding="utf-8")
    # The seed's own inbound and outbound edges are both distinguishable offline.
    assert 'class="orientation-in"' in html
    assert 'class="orientation-out"' in html


def test_export_caps_node_pages_and_reports_what_it_omitted(tmp_path: Path) -> None:
    result = export_explorer_site(_data(extra_nodes=10), tmp_path / "site", max_node_pages=2)

    assert result.node_pages == 2
    assert result.node_pages_omitted == 11
    node_page_files = list((tmp_path / "site").glob("node-*.html"))
    assert len(node_page_files) == 2
    # The highest-degree node keeps its page when the cap bites.
    assert (tmp_path / "site" / node_page_name("file:hub.py")).is_file()


def test_export_caps_index_rows_and_reports_the_remainder(tmp_path: Path) -> None:
    result = export_explorer_site(_data(extra_nodes=10), tmp_path / "site", max_index_rows=4)

    assert result.index_rows == 4
    assert result.index_rows_omitted == 9
    html = (tmp_path / "site" / NODES_FILE).read_text(encoding="utf-8")
    assert "9 further node(s) omitted" in html


def test_export_projects_the_same_canonical_artifact_values(tmp_path: Path) -> None:
    data = _data()

    export_explorer_site(data, tmp_path / "site")

    diff_html = (tmp_path / "site" / DIFF_FILE).read_text(encoding="utf-8")
    health_html = (tmp_path / "site" / HEALTH_FILE).read_text(encoding="utf-8")
    assert data.artifact.source_identity in diff_html
    assert data.artifact.source_revision in diff_html
    assert data.artifact.diff.changed_files[0].path in diff_html
    assert data.artifact.index_generation in health_html
    assert data.artifact.redaction_mode.value in diff_html


def test_export_without_a_graph_still_writes_every_page(tmp_path: Path) -> None:
    data = ExplorerData(artifact=_artifact(), graph=None)

    result = export_explorer_site(data, tmp_path / "site")

    assert result.node_pages == 0
    assert (tmp_path / "site" / NODES_FILE).is_file()
    assert "No graph artifact provided" in (tmp_path / "site" / NODES_FILE).read_text()


def test_export_rejects_a_destination_that_is_not_a_directory(tmp_path: Path) -> None:
    occupied = tmp_path / "already-a-file"
    occupied.write_text("x")

    with pytest.raises(ExplorerExportError, match="not a directory"):
        export_explorer_site(_data(), occupied)


def test_node_page_names_are_filesystem_safe_and_collision_free() -> None:
    first = node_page_name("symbol:pkg/mod.py::Cls#method")
    second = node_page_name("symbol:pkg/mod.py::Cls#other")

    for name in (first, second):
        assert re.fullmatch(r"node-[A-Za-z0-9._-]+\.html", name)
        assert "/" not in name
    assert first != second
    # Two ids that sanitize to the same readable stem still get separate files.
    assert node_page_name("a:b") != node_page_name("a/b")


def _graph_with_dangling_edge() -> ArchGraph:
    """`hub.py` imports an unindexed module, so one edge target has no node.

    Canonical `archex graph export` output contains exactly this shape whenever
    a file imports something outside the index (external package, stdlib, or a
    path the parser skipped).
    """
    return ArchGraph(
        project=GraphProject(name="widget", total_files=2),
        metadata=GraphExportMetadata(archex_version="0.28.0"),
        nodes=[
            GraphNode(id="file:hub.py", type=GraphNodeType.FILE, label="hub.py", module="pkg"),
            GraphNode(id="file:leaf.py", type=GraphNodeType.FILE, label="leaf.py", module="pkg"),
        ],
        edges=[
            GraphEdge(source="file:leaf.py", target="file:hub.py", type=GraphEdgeType.IMPORTS),
            GraphEdge(source="file:hub.py", target="file:absent.py", type=GraphEdgeType.IMPORTS),
        ],
    )


def test_node_index_degree_agrees_with_the_neighborhood_view_it_links_to() -> None:
    """A dangling edge must not make the index disagree with the node's own page."""
    data = ExplorerData(artifact=_artifact(), graph=_graph_with_dangling_edge())

    index = build_node_index_view(data)
    by_id = {row.id: row.degree for row in index.matches}

    for node_id, indexed_degree in by_id.items():
        neighborhood = build_neighborhood_view(data, node_id)
        assert neighborhood.seed is not None
        assert neighborhood.seed.degree == indexed_degree, node_id
    # The dangling edge contributes to neither, so hub.py and leaf.py both read 1.
    assert by_id == {"file:hub.py": 1, "file:leaf.py": 1}


def test_reexport_into_a_reused_directory_removes_the_previous_bundle(tmp_path: Path) -> None:
    destination = tmp_path / "site"
    export_explorer_site(_data(), destination)
    stale_page = destination / node_page_name("file:leaf.py")
    assert stale_page.is_file()

    other = ExplorerData(artifact=_artifact(), graph=_graph_with_dangling_edge())
    result = export_explorer_site(other, destination)

    # The earlier artifact's per-node pages must not survive under their own
    # filenames; they would still open from a file:// URL and show its identity.
    assert not (destination / node_page_name("symbol:hub.py::helper#function")).exists()
    on_disk = {path.name for path in destination.glob("*.html")}
    assert on_disk == set(result.files)


def test_export_reports_the_page_cap_on_the_index_page(tmp_path: Path) -> None:
    result = export_explorer_site(_data(extra_nodes=10), tmp_path / "site", max_node_pages=2)

    html = (tmp_path / "site" / INDEX_FILE).read_text(encoding="utf-8")
    assert f"{result.node_pages_omitted} further node(s) have no page here" in html
    assert "run the local server to reach them" in html


def test_export_honors_a_zero_row_cap_instead_of_forcing_one_row(tmp_path: Path) -> None:
    result = export_explorer_site(_data(), tmp_path / "site", max_index_rows=0)

    assert result.index_rows == 0
    assert result.node_pages == 0
    assert result.index_rows_omitted == 3


def test_export_reports_a_write_failure_as_an_export_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(_self: Path, _content: str, **_kwargs: object) -> int:
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "write_text", _fail)

    with pytest.raises(ExplorerExportError, match="no space left on device"):
        export_explorer_site(_data(), tmp_path / "site")
