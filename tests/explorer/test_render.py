"""Tests for server-rendered explorer HTML: escaping, offline, no client-side JS."""

from __future__ import annotations

from archex.explorer.render import render_diff_page, render_error_page, render_page
from archex.explorer.viewmodel import DiffFileRow, DiffView, ManifestView


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
