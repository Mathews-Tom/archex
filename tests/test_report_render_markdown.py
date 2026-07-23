"""Renderer-parity tests for the Markdown AnalysisArtifactV1 projection.

Every value asserted here must already exist on the artifact under test --
these tests catch a renderer inventing, dropping, or mislabeling data, not
new analysis.
"""

from __future__ import annotations

from pathlib import Path

from archex.report.artifact import build_analysis_artifact
from archex.report.render_markdown import render_markdown


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def test_render_markdown_preserves_provenance_and_handles(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    markdown = render_markdown(artifact)

    assert f"# Diff Review: {artifact.source_identity}" in markdown
    assert f"`{artifact.schema_version.value}`" in markdown
    assert f"`{artifact.diff.base_ref}`" in markdown
    assert f"`{artifact.diff.base_resolved_sha}`" in markdown
    assert "`file:hub.py`" in markdown
    shared_helper = next(
        c for c in artifact.diff.symbol_candidates if c.symbol_name == "shared_helper"
    )
    assert f"`{shared_helper.handle}`" in markdown
    assert f"`{artifact.diff.risk_level.value}`" in markdown


def test_render_markdown_never_embeds_raw_source_text(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    markdown = render_markdown(artifact)

    # The edited line's literal source text must never appear -- only
    # path/line/symbol identity, per the artifact's structural redaction.
    assert "value * 3" not in markdown
    assert "return value" not in markdown


def test_render_markdown_reports_none_for_empty_sections(impact_diff_repo: Path) -> None:
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    markdown = render_markdown(artifact)

    assert "_none_" in markdown or "| - | _none_ | - | - |" in markdown
    assert "## Affected Public Interfaces" in markdown
    assert "- None" in markdown
