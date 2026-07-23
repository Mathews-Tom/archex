"""Renderer-parity tests for the Markdown + Mermaid AnalysisArtifactV1 projection.

Every value asserted here must already exist on the artifact under test --
these tests catch a renderer inventing, dropping, or mislabeling data, not
new analysis.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from archex.report.artifact import build_analysis_artifact
from archex.report.render_markdown import (
    MAX_MERMAID_FILE_NODES,
    render_markdown,
    render_mermaid,
)


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


def test_render_mermaid_none_for_empty_diff(impact_diff_repo: Path) -> None:
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert render_mermaid(artifact) is None
    assert "_No changed files to diagram._" in render_markdown(artifact)


def test_render_mermaid_colors_nodes_by_risk_tier(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    diagram = render_mermaid(artifact)

    assert diagram is not None
    assert diagram.startswith("flowchart TD")
    assert '["hub.py"]:::high' in diagram
    assert '["shared_helper"]:::high' in diagram
    assert "-->" in diagram


def test_render_mermaid_escapes_special_characters_in_labels(impact_diff_repo: Path) -> None:
    weird = impact_diff_repo / 'weird "quoted".py'
    weird.write_text("def f():\n    return 1\n")
    subprocess.run(
        ["git", "add", weird.name], cwd=impact_diff_repo, check=True, capture_output=True
    )

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    diagram = render_mermaid(artifact)

    assert diagram is not None
    assert '\\"' not in diagram
    assert "&quot;" in diagram


def test_render_mermaid_bounds_file_nodes(impact_diff_repo: Path) -> None:
    for i in range(MAX_MERMAID_FILE_NODES + 5):
        (impact_diff_repo / f"extra_{i}.py").write_text(f"def f_{i}():\n    return {i}\n")
    subprocess.run(["git", "add", "."], cwd=impact_diff_repo, check=True, capture_output=True)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    diagram = render_mermaid(artifact)

    assert diagram is not None
    file_node_count = diagram.count('["extra_')
    assert file_node_count <= MAX_MERMAID_FILE_NODES
    assert "more changed file" in diagram
