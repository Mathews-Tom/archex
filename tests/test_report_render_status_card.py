"""Tests for the StatusCard markdown renderer (M9)."""

from __future__ import annotations

import re

from archex.report.render_status_card import render_status_card_markdown
from archex.report.status_card import (
    StatusCard,
    StatusDimension,
    StatusDimensionEvidence,
    StatusDimensionState,
)

_BANNED_PATTERNS = (
    re.compile(r"grade\s*[:=]", re.IGNORECASE),
    re.compile(r"\boverall\s*(score|grade|rating|health)\b", re.IGNORECASE),
    re.compile(r"\bhealth\s*score\b", re.IGNORECASE),
    re.compile(r"^\s*[A-F][+-]?\s*$", re.MULTILINE),
)


def _sample_card() -> StatusCard:
    return StatusCard(
        source_identity="example/repo",
        revision="0123456789ab",
        generated_at="123.0",
        dimensions=[
            StatusDimension(
                name="Documentation linkage",
                state=StatusDimensionState.EVIDENCED,
                detail="3 documentation link(s) reference 2 distinct source path(s)",
                provider="doc_link",
                evidence=[
                    StatusDimensionEvidence(description="linked from README.md", location="a.py")
                ],
            ),
            StatusDimension(
                name="ADR provenance",
                state=StatusDimensionState.UNKNOWN,
                detail="no ADR directory found",
                provider="adr",
            ),
        ],
    )


class TestRenderStatusCardMarkdown:
    def test_includes_source_identity_and_revision(self) -> None:
        markdown = render_status_card_markdown(_sample_card())
        assert "example/repo" in markdown
        assert "0123456789ab" in markdown

    def test_renders_each_dimension_by_name(self) -> None:
        markdown = render_status_card_markdown(_sample_card())
        assert "### Documentation linkage" in markdown
        assert "### ADR provenance" in markdown

    def test_evidenced_dimension_lists_its_evidence(self) -> None:
        markdown = render_status_card_markdown(_sample_card())
        assert "**Evidenced**" in markdown
        assert "`a.py`" in markdown
        assert "linked from README.md" in markdown

    def test_unknown_dimension_shows_reason_without_evidence_list(self) -> None:
        markdown = render_status_card_markdown(_sample_card())
        adr_section = markdown.split("### ADR provenance", 1)[1]
        assert "**Unknown**" in adr_section
        assert "no ADR directory found" in adr_section
        assert "Evidence:" not in adr_section

    def test_never_renders_a_composite_grade_or_overall_score(self) -> None:
        markdown = render_status_card_markdown(_sample_card())
        for pattern in _BANNED_PATTERNS:
            assert not pattern.search(markdown), pattern.pattern

    def test_states_no_composite_score_disclaimer(self) -> None:
        markdown = render_status_card_markdown(_sample_card())
        assert "no composite score or letter grade" in markdown.lower()
