"""Tests for the M9 dimensioned StatusCard model and builder.

Covers `build_status_card`'s projection of M9's documentation-evidence
channel plus locally verifiable release/CI evidence into one canonical,
never-scored status card.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from archex.report.status_card import (
    StatusCard,
    StatusDimension,
    StatusDimensionEvidence,
    StatusDimensionState,
    build_status_card,
)

if TYPE_CHECKING:
    from archex.models import IndexConfig, RepoSource


def _enable_documentation_providers(monkeypatch: pytest.MonkeyPatch, providers: list[str]) -> None:
    from archex.config import load_index_config as original_load_index_config

    def _patched(source: RepoSource) -> IndexConfig:
        config = original_load_index_config(source)
        return config.model_copy(update={"documentation_evidence_providers": providers})

    monkeypatch.setattr("archex.report.status_card.load_index_config", _patched)


class TestStatusDimensionEvidence:
    def test_rejects_empty_description(self) -> None:
        with pytest.raises(ValueError, match="description"):
            StatusDimensionEvidence(description=" ", location="README.md")

    def test_rejects_empty_location(self) -> None:
        with pytest.raises(ValueError, match="location"):
            StatusDimensionEvidence(description="a link", location=" ")


class TestStatusDimension:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            StatusDimension(
                name=" ", state=StatusDimensionState.UNKNOWN, detail="not configured", provider="x"
            )

    def test_rejects_empty_detail(self) -> None:
        with pytest.raises(ValueError, match="detail"):
            StatusDimension(name="X", state=StatusDimensionState.UNKNOWN, detail=" ", provider="x")

    def test_evidenced_state_requires_evidence(self) -> None:
        with pytest.raises(ValueError, match="evidence"):
            StatusDimension(
                name="X",
                state=StatusDimensionState.EVIDENCED,
                detail="found something",
                provider="x",
            )

    def test_unknown_state_allows_no_evidence(self) -> None:
        dimension = StatusDimension(
            name="X", state=StatusDimensionState.UNKNOWN, detail="not configured", provider="x"
        )
        assert dimension.evidence == []


class TestStatusCardHasNoCompositeField:
    """Structural guardrail: no score/grade/health field exists on the model.

    The absence must be a type-design fact, not a runtime check, so no
    call site can construct or emit a composite rating even by accident.
    """

    _BANNED_SUBSTRINGS = ("score", "grade", "health", "rating", "rank")

    def test_status_card_fields_have_no_banned_names(self) -> None:
        for field_name in StatusCard.model_fields:
            lowered = field_name.lower()
            assert not any(banned in lowered for banned in self._BANNED_SUBSTRINGS), field_name

    def test_status_dimension_fields_have_no_banned_names(self) -> None:
        for field_name in StatusDimension.model_fields:
            lowered = field_name.lower()
            assert not any(banned in lowered for banned in self._BANNED_SUBSTRINGS), field_name


class TestBuildStatusCard:
    def test_dimensions_are_unknown_without_configured_providers(
        self, python_simple_repo: Path
    ) -> None:
        card = build_status_card(python_simple_repo)

        by_provider = {dimension.provider: dimension for dimension in card.dimensions}
        assert by_provider["doc_link"].state == StatusDimensionState.UNKNOWN
        assert by_provider["adr"].state == StatusDimensionState.UNKNOWN
        assert by_provider["ownership"].state == StatusDimensionState.UNKNOWN

    def test_doc_link_dimension_evidenced_when_configured_and_present(
        self, python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (python_simple_repo / "README.md").write_text("See [main](main.py).\n")
        _enable_documentation_providers(monkeypatch, ["doc_link"])

        card = build_status_card(python_simple_repo)

        by_provider = {dimension.provider: dimension for dimension in card.dimensions}
        doc_link = by_provider["doc_link"]
        assert doc_link.state == StatusDimensionState.EVIDENCED
        assert any(item.location == "main.py" for item in doc_link.evidence)

    def test_release_dimension_reads_changelog_and_ci_workflow(
        self, python_simple_repo: Path
    ) -> None:
        (python_simple_repo / "CHANGELOG.md").write_text(
            "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- staged work\n\n"
            "## [1.2.3] - 2026-01-01\n\n### Added\n\n- first release\n"
        )
        workflow_dir = python_simple_repo / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        (workflow_dir / "report-diff.yml").write_text("name: Report diff\n")

        card = build_status_card(python_simple_repo)

        by_provider = {dimension.provider: dimension for dimension in card.dimensions}
        release = by_provider["release"]
        assert release.state == StatusDimensionState.EVIDENCED
        assert "1.2.3" in release.detail
        assert "Unreleased" not in release.detail

    def test_release_dimension_unknown_without_any_local_evidence(
        self, python_simple_repo: Path
    ) -> None:
        card = build_status_card(python_simple_repo)

        by_provider = {dimension.provider: dimension for dimension in card.dimensions}
        assert by_provider["release"].state == StatusDimensionState.UNKNOWN

    def test_to_json_round_trips(self, python_simple_repo: Path) -> None:
        card = build_status_card(python_simple_repo)
        restored = StatusCard.model_validate_json(card.to_json())
        assert restored.source_identity == card.source_identity
        assert len(restored.dimensions) == len(card.dimensions)

    def test_deterministic_evidence_bound(
        self, python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        readme = python_simple_repo / "README.md"
        lines = "\n".join(f"See [x{i}](main.py) too." for i in range(20))
        readme.write_text(lines + "\n")
        _enable_documentation_providers(monkeypatch, ["doc_link"])

        card = build_status_card(python_simple_repo)
        by_provider = {dimension.provider: dimension for dimension in card.dimensions}
        assert len(by_provider["doc_link"].evidence) <= 10
