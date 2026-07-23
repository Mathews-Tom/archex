"""Tests for server-rendered explorer HTML: escaping, offline, no client-side JS."""

from __future__ import annotations

from archex.explorer.render import (
    render_diff_page,
    render_error_page,
    render_health_page,
    render_module_map_page,
    render_neighborhood_page,
    render_page,
    render_receipt_page,
)
from archex.explorer.viewmodel import (
    DiffFileRow,
    DiffView,
    HealthView,
    ManifestView,
    ModuleMapView,
    ModuleRow,
    NeighborEdgeRow,
    NeighborhoodView,
    NeighborNodeRow,
    ReceiptView,
)


def _manifest(source_identity: str = "acme/widget") -> ManifestView:
    return ManifestView(
        source_identity=source_identity,
        source_revision="deadbeef",
        archex_version="0.22.0",
        schema_version="1.0.0",
        generated_at="2026-07-24T00:00:00Z",
        freshness="clean",
        completeness="complete",
        confidence="high",
        redaction_mode="redacted",
        has_graph=False,
        excluded_total=0,
        unknown_total=0,
        evidence_count=1,
    )


def _diff_view(path: str = "hub.py") -> DiffView:
    return DiffView(
        base_ref="main",
        base_resolved_sha="deadbeef",
        head_ref="",
        risk_level="high",
        risk_reasons=["public_interface_changed"],
        changed_files=[
            DiffFileRow(path=path, status="M", handle=f"file:{path}", old_path=None, hunks=[])
        ],
        changed_files_total=1,
        symbol_candidates=[],
        symbol_candidates_total=0,
        affected_interfaces=[],
        affected_interfaces_total=0,
        test_candidates=[],
        test_candidates_total=0,
        unsupported_files=[],
        unsupported_files_total=0,
    )


def test_render_page_is_offline_and_script_free() -> None:
    html = render_page("archex explorer", _manifest(), "<h2>Body</h2>")

    assert html.startswith("<!DOCTYPE html>")
    assert "<script" not in html.lower()
    assert "https://" not in html
    assert "http://" not in html
    assert "cdn." not in html.lower()


def test_render_page_escapes_manifest_fields() -> None:
    manifest = _manifest(source_identity="<script>alert(1)</script>")

    html = render_page("t", manifest, "")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_diff_page_escapes_untrusted_path_values() -> None:
    view = _diff_view(path="<img src=x onerror=alert(1)>.py")

    html = render_diff_page(_manifest(), view)

    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_render_diff_page_shows_truncation_note_when_bounded() -> None:
    view = _diff_view()
    html = render_diff_page(_manifest(), view)

    assert "Showing" not in html  # not truncated: 1 shown of 1 total


def test_render_error_page_never_echoes_artifact_content() -> None:
    html = render_error_page(403, "Forbidden", "A valid session token is required.")

    assert "403 Forbidden" in html
    assert "session token" in html
    assert "<script" not in html.lower()


def _receipt_view() -> ReceiptView:
    return ReceiptView(
        freshness="clean",
        completeness="complete",
        confidence="high",
        redaction_mode="redacted",
        generated_at="2026-07-24T00:00:00Z",
        evidence_locations=[],
        evidence_locations_total=0,
        excluded_counts={"unmapped": 2},
        unknown_counts={},
    )


def _health_view() -> HealthView:
    return HealthView(
        archex_version="0.22.0",
        schema_version="1.0.0",
        index_generation="gen1",
        index_schema_version="1",
        chunker_revision="c1",
        parser_versions={"python": "tree-sitter-python"},
        retrieval_profile=None,
        config_fingerprint="cfg1",
        working_tree_fingerprint="fp",
        producer="archex-cli",
        producer_version="0.22.0",
    )


def test_render_module_map_page_reports_unavailable_without_graph() -> None:
    view = ModuleMapView(available=False, modules=[], modules_total=0)

    html = render_module_map_page(_manifest(), view)

    assert "No graph artifact provided" in html
    assert "<table" not in html


def test_render_module_map_page_renders_aggregated_rows() -> None:
    view = ModuleMapView(
        available=True,
        modules=[
            ModuleRow(module="pkg", node_count=3, file_count=2, symbol_count=1, interface_count=0)
        ],
        modules_total=1,
    )

    html = render_module_map_page(_manifest(), view)

    assert "pkg" in html
    assert "<table" in html


def test_render_neighborhood_page_reports_unavailable_without_graph() -> None:
    view = NeighborhoodView(
        available=False,
        query=None,
        error="no graph artifact provided",
        seed=None,
        direction="both",
        depth=1,
        limit=25,
        nodes=[],
        edges=[],
        hubs=[],
        truncated=False,
        omitted_edges=0,
    )

    html = render_neighborhood_page(_manifest(), view)

    assert "No graph artifact provided" in html
    assert "<form" in html  # the search form still renders


def test_render_neighborhood_page_escapes_node_and_edge_ids() -> None:
    seed = NeighborNodeRow(
        id="<script>x</script>", type="file", label="a.py", path="a.py", degree=1
    )
    view = NeighborhoodView(
        available=True,
        query="a.py",
        error=None,
        seed=seed,
        direction="both",
        depth=1,
        limit=25,
        nodes=[seed],
        edges=[
            NeighborEdgeRow(
                source_id="<script>x</script>",
                target_id="file:b.py",
                type="imports",
                confidence="extracted",
            )
        ],
        hubs=[],
        truncated=False,
        omitted_edges=0,
    )

    html = render_neighborhood_page(_manifest(), view)

    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_neighborhood_page_reports_unresolved_query_error() -> None:
    view = NeighborhoodView(
        available=True,
        query="does-not-exist",
        error="no node matched 'does-not-exist'",
        seed=None,
        direction="both",
        depth=1,
        limit=25,
        nodes=[],
        edges=[],
        hubs=[],
        truncated=False,
        omitted_edges=0,
    )

    html = render_neighborhood_page(_manifest(), view)

    assert "no node matched" in html


def test_render_receipt_page_shows_excluded_counts() -> None:
    html = render_receipt_page(_manifest(), _receipt_view())

    assert "unmapped: 2" in html
    assert "Freshness: clean" in html


def test_render_health_page_shows_identity_fields() -> None:
    html = render_health_page(_manifest(), _health_view())

    assert "gen1" in html
    assert "tree-sitter-python" in html
