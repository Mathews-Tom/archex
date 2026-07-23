"""Tests for the bounded, read-only diff-review delta summary."""

from __future__ import annotations

import subprocess
from pathlib import Path

from archex.report.artifact import build_analysis_artifact
from archex.report.delta import MAX_DELTA_RISK_CANDIDATES, ReportDelta, build_report_delta


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def test_build_report_delta_summarizes_artifact_fields(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    delta = build_report_delta(artifact)

    assert delta.schema_version == artifact.schema_version.value
    assert delta.source_revision == artifact.source_revision
    assert delta.base_ref == artifact.diff.base_ref
    assert delta.base_resolved_sha == artifact.diff.base_resolved_sha
    assert delta.freshness == artifact.freshness.value
    assert delta.risk_level == artifact.diff.risk_level.value
    assert delta.changed_files_total == artifact.diff.changed_files_total
    assert delta.symbol_candidates_total == artifact.diff.symbol_candidates_total
    assert delta.high_risk_symbol_count >= 1


def test_build_report_delta_caps_top_risk_candidates(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    delta = build_report_delta(artifact)

    assert len(delta.top_risk_candidates) <= MAX_DELTA_RISK_CANDIDATES
    assert all(candidate.startswith("high:") for candidate in delta.top_risk_candidates)


def test_build_report_delta_is_deterministic_for_the_same_commit(impact_diff_repo: Path) -> None:
    """Two builds against the same committed state must agree, other than the timestamp.

    This is the property the CI example depends on: a pinned commit yields
    a reproducible delta, not one that drifts run to run.
    """
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    first = build_report_delta(artifact).model_dump(exclude={"generated_at"})
    second = build_report_delta(artifact).model_dump(exclude={"generated_at"})

    assert first == second


def test_report_delta_json_round_trips(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    delta = build_report_delta(artifact)

    restored = ReportDelta.model_validate_json(delta.to_json())

    assert restored == delta


def test_report_delta_markdown_never_embeds_raw_source_text(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    delta = build_report_delta(artifact)

    markdown = delta.to_markdown()

    assert "value * 3" not in markdown
    assert f"`{artifact.diff.risk_level.value}`" in markdown


def test_report_delta_stays_small(impact_diff_repo: Path) -> None:
    for i in range(200):
        (impact_diff_repo / f"extra_{i}.py").write_text(f"def f_{i}():\n    return {i}\n")
    subprocess.run(["git", "add", "."], cwd=impact_diff_repo, check=True, capture_output=True)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    delta = build_report_delta(artifact)

    # Bounded independent of how large the underlying diff is -- suitable
    # for a CI job summary or PR-comment-sized artifact.
    assert len(delta.to_json().encode("utf-8")) < 5_000
