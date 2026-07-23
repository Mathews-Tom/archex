"""Security and size tests for the static, offline HTML AnalysisArtifactV1 renderer."""

from __future__ import annotations

import subprocess
from pathlib import Path

from archex.report.artifact import build_analysis_artifact
from archex.report.render_html import MAX_HTML_ROWS, render_html


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def test_render_html_has_no_script_tags(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    assert "<script" not in html.lower()


def test_render_html_has_no_remote_references(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html.lower()
    assert "cdn" not in html.lower()


def test_render_html_changed_file_paths_are_clickable_editor_links(
    impact_diff_repo: Path,
) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    expected_href = f'href="vscode://file/{artifact.source_root}/hub.py:5"'
    assert expected_href in html
    assert '<a class="file-link"' in html


def test_render_html_symbol_candidate_lines_are_clickable(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    shared_helper = next(
        c for c in artifact.diff.symbol_candidates if c.symbol_name == "shared_helper"
    )

    html = render_html(artifact)

    expected_href = f'href="vscode://file/{artifact.source_root}/hub.py:{shared_helper.start_line}"'
    assert expected_href in html


def test_render_html_path_list_links_have_no_line_suffix(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    interface_path = artifact.diff.affected_interfaces[0].path
    expected_href = f'href="vscode://file/{artifact.source_root}/{interface_path}"'
    assert expected_href in html


def test_render_html_is_a_single_self_contained_document(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert html.count("<html") == 1
    assert html.rstrip().endswith("</html>")


def test_render_html_never_embeds_raw_source_text(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    assert "value * 3" not in html
    assert "return value" not in html


def test_render_html_escapes_path_and_symbol_text(impact_diff_repo: Path) -> None:
    weird = impact_diff_repo / 'weird "quoted"<tag>.py'
    weird.write_text("def f():\n    return 1\n")
    subprocess.run(
        ["git", "add", weird.name], cwd=impact_diff_repo, check=True, capture_output=True
    )

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    html = render_html(artifact)

    assert "<tag>" not in html
    assert "&lt;tag&gt;" in html


def test_render_html_bounds_table_rows(impact_diff_repo: Path) -> None:
    for i in range(MAX_HTML_ROWS + 20):
        (impact_diff_repo / f"extra_{i}.py").write_text(f"def f_{i}():\n    return {i}\n")
    subprocess.run(["git", "add", "."], cwd=impact_diff_repo, check=True, capture_output=True)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    html = render_html(artifact)

    assert html.count("<tr><td><code>A</code>") <= MAX_HTML_ROWS
    assert "additional entries omitted" in html or "additional entry omitted" in html


def test_render_html_stays_within_a_reasonable_size_budget(impact_diff_repo: Path) -> None:
    for i in range(MAX_HTML_ROWS + 50):
        (impact_diff_repo / f"extra_{i}.py").write_text(f"def f_{i}():\n    return {i}\n")
    subprocess.run(["git", "add", "."], cwd=impact_diff_repo, check=True, capture_output=True)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    html = render_html(artifact)

    # Even with far more changed files than the row cap, the static page
    # stays well under a size a browser opens instantly from a local file.
    assert len(html.encode("utf-8")) < 200_000


def test_render_html_reports_none_for_empty_diff(impact_diff_repo: Path) -> None:
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    html = render_html(artifact)

    assert "No changed files to diagram." in html
    assert "<em>None</em>" in html
