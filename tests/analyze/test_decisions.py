"""Tests for architecture decision inference."""

from __future__ import annotations

from archex.analyze.decisions import infer_decisions
from archex.models import (
    ArchDecision,
    DetectedPattern,
    PatternCategory,
    PatternEvidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pattern(
    name: str = "middleware_chain",
    display_name: str = "Middleware Chain",
    confidence: float = 0.8,
    description: str = "Chain of responsibility pattern for request processing",
    category: PatternCategory = PatternCategory.BEHAVIORAL,
    num_evidence: int = 1,
) -> DetectedPattern:
    evidence = [
        PatternEvidence(
            file_path=f"src/middleware_{i}.py",
            start_line=1 + i * 10,
            end_line=10 + i * 10,
            symbol=f"Middleware{i}",
            explanation=f"Base middleware class #{i}",
        )
        for i in range(num_evidence)
    ]
    return DetectedPattern(
        name=name,
        display_name=display_name,
        confidence=confidence,
        evidence=evidence,
        description=description,
        category=category,
    )


# ---------------------------------------------------------------------------
# Structural decision generation
# ---------------------------------------------------------------------------


def test_infer_decisions_from_patterns() -> None:
    pattern = _make_pattern()
    decisions = infer_decisions([pattern])

    assert len(decisions) == 1
    d = decisions[0]
    assert isinstance(d, ArchDecision)
    assert "Middleware Chain" in d.decision
    assert d.source == "structural"
    assert len(d.alternatives) > 0
    assert len(d.implications) > 0


def test_decisions_include_evidence_from_patterns() -> None:
    pattern = _make_pattern(num_evidence=2)
    decisions = infer_decisions([pattern])

    assert len(decisions) == 1
    evidence = decisions[0].evidence
    assert len(evidence) == 2
    assert any("src/middleware_0.py" in e for e in evidence)
    assert any("src/middleware_1.py" in e for e in evidence)


def test_infer_decisions_empty_patterns_returns_empty() -> None:
    decisions = infer_decisions([])
    assert decisions == []


def test_infer_decisions_returns_structural_source() -> None:
    patterns = [_make_pattern("factory", "Factory", 0.9)]
    decisions = infer_decisions(patterns)

    assert len(decisions) == 1
    assert decisions[0].source == "structural"


def test_confidence_filtering_below_threshold() -> None:
    low = _make_pattern(confidence=0.49)
    high = _make_pattern(name="factory", display_name="Factory", confidence=0.5)
    decisions = infer_decisions([low, high])

    assert len(decisions) == 1
    assert "Factory" in decisions[0].decision


def test_confidence_exactly_at_threshold_is_included() -> None:
    pattern = _make_pattern(confidence=0.5)
    decisions = infer_decisions([pattern])
    assert len(decisions) == 1


def test_confidence_zero_is_excluded() -> None:
    pattern = _make_pattern(confidence=0.0)
    decisions = infer_decisions([pattern])
    assert decisions == []


def test_multiple_patterns_generate_multiple_decisions() -> None:
    patterns = [
        _make_pattern("factory", "Factory", 0.9),
        _make_pattern("singleton", "Singleton", 0.7),
        _make_pattern("observer", "Observer", 0.6),
    ]
    decisions = infer_decisions(patterns)
    assert len(decisions) == 3
