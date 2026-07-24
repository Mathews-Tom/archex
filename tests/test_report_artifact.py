"""Provenance and schema-compatibility tests for AnalysisArtifactV1.

Covers `build_analysis_artifact`'s projection of the existing impact/graph/
index surfaces into one canonical artifact, and `load_analysis_artifact`'s
schema-version compatibility gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from archex.impact import ImpactRiskLevel, SymbolRiskLevel
from archex.models import ContextCompletenessStatus, ContextFreshness
from archex.report.artifact import (
    MAX_CHANGED_FILES,
    AnalysisArtifactV1,
    AnalysisConfidence,
    RedactionMode,
    ReportArtifactError,
    ReportSchemaVersion,
    assert_supported_report_schema_version,
    build_analysis_artifact,
    load_analysis_artifact,
)


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def _git_add(repo: Path, path: str) -> None:
    subprocess.run(["git", "add", path], cwd=repo, check=True, capture_output=True)


def test_build_analysis_artifact_reports_full_provenance(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.schema_version.value == "1.0.0"
    assert artifact.archex_version
    assert artifact.producer == "archex-cli"
    assert artifact.producer_version == artifact.archex_version
    assert artifact.source_identity == str(impact_diff_repo)
    assert artifact.source_revision
    assert artifact.working_tree_fingerprint
    assert artifact.index_generation
    assert artifact.index_schema_version
    assert artifact.chunker_revision
    assert artifact.retrieval_profile is None
    assert artifact.config_fingerprint
    assert artifact.freshness == ContextFreshness.CLEAN
    assert artifact.redaction_mode == RedactionMode.REDACTED
    assert artifact.parser_versions.get("python")
    assert artifact.source_root == str(impact_diff_repo)


def test_build_analysis_artifact_reports_diff_symbol_risk(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.base_ref == "HEAD"
    assert artifact.diff.base_resolved_sha
    assert artifact.diff.head_ref == artifact.source_revision
    assert artifact.diff.changed_files_total == 1
    change = artifact.diff.changed_files[0]
    assert change.path == "hub.py"
    assert change.handle == "file:hub.py"
    assert change.hunks

    assert artifact.diff.symbol_candidates_total >= 1
    shared_helper = next(
        c for c in artifact.diff.symbol_candidates if c.symbol_name == "shared_helper"
    )
    assert shared_helper.handle.startswith("symbol:") or shared_helper.handle.startswith("chunk:")
    assert shared_helper.risk_level == SymbolRiskLevel.HIGH.value
    assert shared_helper.confidence == AnalysisConfidence.HIGH
    assert shared_helper.evidence
    assert artifact.diff.risk_level == ImpactRiskLevel.HIGH
    assert "high_fan_in" in artifact.diff.risk_reasons


def test_interface_candidates_carry_resolvable_symbol_handles(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.affected_interfaces_total >= 1
    interface = next(
        c for c in artifact.diff.affected_interfaces if c.symbol_id.startswith("hub.py::")
    )
    assert interface.path == "hub.py"
    assert interface.handle == f"symbol:{interface.symbol_id}"
    assert "::" not in interface.path


def test_fully_unmapped_diff_is_low_confidence(impact_diff_repo: Path) -> None:
    new_file = impact_diff_repo / "notes.bin"
    new_file.write_bytes(b"\x00\x01binary-blob")
    _git_add(impact_diff_repo, "notes.bin")

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.unsupported_files_total == 1
    assert artifact.diff.unsupported_files[0].path == "notes.bin"
    assert artifact.excluded_counts["unmapped_changed_files"] == 1
    assert artifact.completeness == ContextCompletenessStatus.INCOMPLETE
    assert artifact.confidence == AnalysisConfidence.LOW


def test_partially_unmapped_diff_is_medium_confidence(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    (impact_diff_repo / "notes.bin").write_bytes(b"\x00\x01binary-blob")
    _git_add(impact_diff_repo, "notes.bin")

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.changed_files_total == 2
    assert artifact.diff.unsupported_files_total == 1
    assert artifact.confidence == AnalysisConfidence.MEDIUM


def test_import_only_change_is_unknown_not_silently_clean(impact_diff_repo: Path) -> None:
    hub = impact_diff_repo / "hub.py"
    hub.write_text(
        hub.read_text().replace(
            "from __future__ import annotations",
            "from __future__ import annotations\nimport os  # unused, module-level only",
        )
    )

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.unknown_counts["changed_files_without_symbol_coverage"] == 1
    assert not any(c.file_path == "hub.py" for c in artifact.diff.symbol_candidates)


def test_empty_diff_is_high_confidence_complete(impact_diff_repo: Path) -> None:
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.changed_files_total == 0
    assert artifact.confidence == AnalysisConfidence.HIGH
    assert artifact.completeness == ContextCompletenessStatus.COMPLETE
    assert artifact.diff.risk_level == ImpactRiskLevel.LOW


def test_json_round_trip_preserves_semantics(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    restored = AnalysisArtifactV1.model_validate_json(artifact.to_json())

    assert restored == artifact


def test_load_analysis_artifact_round_trips_through_disk(
    impact_diff_repo: Path, tmp_path: Path
) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    output = tmp_path / "artifact.json"
    output.write_text(artifact.to_json())

    loaded = load_analysis_artifact(output)

    assert loaded == artifact


def test_load_analysis_artifact_rejects_malformed_json(tmp_path: Path) -> None:
    output = tmp_path / "broken.json"
    output.write_text("{not valid json")

    with pytest.raises(ReportArtifactError):
        load_analysis_artifact(output)


def test_assert_supported_report_schema_version_rejects_future_major() -> None:
    with pytest.raises(ReportArtifactError):
        assert_supported_report_schema_version(ReportSchemaVersion(value="2.0.0"))


def test_bounded_lists_carry_true_totals(
    impact_diff_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("archex.report.artifact.MAX_CHANGED_FILES", 0)
    _edit_hub(impact_diff_repo)

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.changed_files == []
    assert artifact.diff.changed_files_total == 1
    assert MAX_CHANGED_FILES == 200  # module default unaffected outside this test


def test_invalid_base_ref_raises_report_artifact_error(impact_diff_repo: Path) -> None:
    with pytest.raises(ReportArtifactError):
        build_analysis_artifact(impact_diff_repo, base_ref="not-a-real-ref")


def test_deleted_file_is_reported_without_symbol_resolution(impact_diff_repo: Path) -> None:
    (impact_diff_repo / "leaf.py").unlink()

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    deleted = next(c for c in artifact.diff.changed_files if c.path == "leaf.py")
    assert deleted.status == "D"
    assert not any(c.file_path == "leaf.py" for c in artifact.diff.symbol_candidates)


def test_source_revision_matches_git_head(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=impact_diff_repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.source_revision == head
    assert artifact.diff.base_resolved_sha == head


def test_symbol_candidate_carries_runtime_evidence_when_available(
    impact_diff_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M7: a symbol candidate reports runtime-profile provenance when the
    runtime_profile provider is enabled and its evidence covers the
    symbol's file and qualified name.
    """
    import json

    _edit_hub(impact_diff_repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=impact_diff_repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    # First build discovers shared_helper's exact qualified_name without guessing.
    baseline = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    shared_helper = next(
        c for c in baseline.diff.symbol_candidates if c.symbol_name == "shared_helper"
    )
    assert shared_helper.runtime_sample_count is None

    evidence_dir = impact_diff_repo / ".archex" / "runtime-evidence" / "profile"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "manifest.json").write_text(json.dumps({"revision": head, "tool": "cProfile"}))
    label = shared_helper.qualified_name or shared_helper.symbol_name
    frame = f"{shared_helper.file_path}:{label}"
    (evidence_dir / "profile.folded").write_text(f"{frame} 7\n")

    from archex.config import load_index_config as original_load_index_config
    from archex.models import IndexConfig, RepoSource

    def _patched_load_index_config(source: RepoSource) -> IndexConfig:
        config = original_load_index_config(source)
        return config.model_copy(update={"runtime_evidence_providers": ["runtime_profile"]})

    monkeypatch.setattr("archex.report.artifact.load_index_config", _patched_load_index_config)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    enriched = next(c for c in artifact.diff.symbol_candidates if c.symbol_name == "shared_helper")
    assert enriched.runtime_sample_count == 7
    assert enriched.runtime_revision == head
    assert enriched.runtime_stale is False


def test_coverage_for_path_reports_match_and_staleness() -> None:
    from archex.integrations.runtime.models import CoverageFileEvidence
    from archex.report.artifact import _coverage_for_path  # pyright: ignore[reportPrivateUsage]

    records = [CoverageFileEvidence(file_path="a.py", line_rate=0.5, revision="rev-a")]

    rate, revision, stale = _coverage_for_path("a.py", records, "rev-a")
    assert (rate, revision, stale) == (0.5, "rev-a", False)

    rate, revision, stale = _coverage_for_path("a.py", records, "rev-b")
    assert (rate, revision, stale) == (0.5, "rev-a", True)

    rate, revision, stale = _coverage_for_path("missing.py", records, "rev-a")
    assert (rate, revision, stale) == (None, None, False)


def test_runtime_sample_count_for_symbol_matches_file_and_qualified_name() -> None:
    from archex.integrations.runtime.models import RuntimeProfileEvidence, RuntimeStackSample
    from archex.report.artifact import (
        _runtime_sample_count_for_symbol,  # pyright: ignore[reportPrivateUsage]
    )

    records = [
        RuntimeProfileEvidence(
            samples=[
                RuntimeStackSample(frames=("a.py:outer", "a.py:inner"), sample_count=3),
                RuntimeStackSample(frames=("a.py:inner",), sample_count=2),
                RuntimeStackSample(frames=("b.py:other",), sample_count=9),
            ],
            total_samples=14,
            revision="rev-a",
        )
    ]

    count, revision, stale = _runtime_sample_count_for_symbol("a.py", "inner", records, "rev-a")
    assert (count, revision, stale) == (5, "rev-a", False)

    count, revision, stale = _runtime_sample_count_for_symbol("a.py", "inner", records, "rev-b")
    assert (count, revision, stale) == (5, "rev-a", True)

    count, revision, stale = _runtime_sample_count_for_symbol("c.py", None, records, "rev-a")
    assert (count, revision, stale) == (None, None, False)


def test_history_disabled_by_default_with_sparse_real_repo(
    impact_diff_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M8: a real (but commit-sparse) repo enables the git_log provider and
    successfully collects real evidence (AVAILABLE receipt), but the
    eligibility policy disables exposure because density is far below
    threshold -- proving sparse-history results carry no hidden change.
    """
    _edit_hub(impact_diff_repo)

    from archex.config import load_index_config as original_load_index_config
    from archex.models import IndexConfig, RepoSource

    def _patched_load_index_config(source: RepoSource) -> IndexConfig:
        config = original_load_index_config(source)
        return config.model_copy(update={"history_evidence_providers": ["git_log"]})

    monkeypatch.setattr("archex.report.artifact.load_index_config", _patched_load_index_config)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")

    assert artifact.diff.history_eligibility is not None
    assert artifact.diff.history_eligibility.enabled is False
    for change in artifact.diff.changed_files:
        assert change.history_change_count is None
        assert change.history_linked_references == []


def test_history_eligibility_absent_when_channel_unconfigured(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    assert artifact.diff.history_eligibility is None


def test_history_summary_for_path_counts_matching_cards_and_references() -> None:
    from archex.integrations.history.models import ChangeCard, LinkedReference
    from archex.report.artifact import (
        _history_summary_for_path,  # pyright: ignore[reportPrivateUsage]
    )

    cards = [
        ChangeCard(
            commit_sha="c1",
            commit_subject="s",
            committed_at="t",
            changed_files=["a.py", "b.py"],
            linked_references=[LinkedReference(raw_text="#1", identifier="1")],
            revision="rev",
        ),
        ChangeCard(
            commit_sha="c2",
            commit_subject="s",
            committed_at="t",
            changed_files=["a.py"],
            linked_references=[LinkedReference(raw_text="#2", identifier="2")],
            revision="rev",
        ),
        ChangeCard(
            commit_sha="c3",
            commit_subject="s",
            committed_at="t",
            changed_files=["b.py"],
            revision="rev",
        ),
    ]

    count, references = _history_summary_for_path("a.py", cards)
    assert count == 2
    assert references == ["1", "2"]

    count, references = _history_summary_for_path("missing.py", cards)
    assert count == 0
    assert references == []
