"""Tests for analyze/patterns.py: architectural pattern detection."""

from __future__ import annotations

from pathlib import Path

from archex.analyze.patterns import detect_patterns
from archex.index.graph import DependencyGraph
from archex.models import ParsedFile
from archex.parse.adapters.python import PythonAdapter
from archex.parse.engine import TreeSitterEngine

PATTERNS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "python_patterns"
SIMPLE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "python_simple"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_file(filename: str, fixture_dir: Path) -> ParsedFile:
    engine = TreeSitterEngine()
    adapter = PythonAdapter()
    fpath = fixture_dir / filename
    tree = engine.parse_file(str(fpath), "python")
    source = fpath.read_bytes()
    symbols = adapter.extract_symbols(tree, source, filename)
    return ParsedFile(path=filename, language="python", symbols=symbols)


def _graph_for(parsed_files: list[ParsedFile]) -> DependencyGraph:
    graph = DependencyGraph()
    for pf in parsed_files:
        graph.add_file_node(pf.path)
    return graph


def _pattern_names(parsed_files: list[ParsedFile]) -> set[str]:
    graph = _graph_for(parsed_files)
    patterns = detect_patterns(parsed_files, graph, [])
    return {p.name for p in patterns}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_middleware_chain_detected() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph, [])
    names = {p.name for p in patterns}
    assert "middleware_chain" in names


def test_middleware_chain_high_confidence() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph, [])
    pattern = next(p for p in patterns if p.name == "middleware_chain")
    assert pattern.confidence >= 0.6


def test_middleware_chain_evidence_has_correct_file() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph, [])
    pattern = next(p for p in patterns if p.name == "middleware_chain")
    for ev in pattern.evidence:
        assert ev.file_path == "middleware.py"


def test_middleware_chain_evidence_has_valid_lines() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph, [])
    pattern = next(p for p in patterns if p.name == "middleware_chain")
    for ev in pattern.evidence:
        assert ev.start_line >= 1
        assert ev.end_line >= ev.start_line


def test_plugin_system_detected() -> None:
    pf = _parse_file("plugins.py", PATTERNS_FIXTURE)
    assert "plugin_system" in _pattern_names([pf])


def test_event_bus_detected() -> None:
    pf = _parse_file("events.py", PATTERNS_FIXTURE)
    assert "event_bus" in _pattern_names([pf])


def test_repository_detected() -> None:
    pf = _parse_file("repository.py", PATTERNS_FIXTURE)
    assert "repository" in _pattern_names([pf])


def test_strategy_detected() -> None:
    pf = _parse_file("strategies.py", PATTERNS_FIXTURE)
    assert "strategy" in _pattern_names([pf])


def test_strategy_evidence_symbols() -> None:
    pf = _parse_file("strategies.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph, [])
    pattern = next(p for p in patterns if p.name == "strategy")
    symbols_found = {ev.symbol for ev in pattern.evidence}
    # SortStrategy is the protocol; BubbleSort/QuickSort are concretes
    assert "SortStrategy" in symbols_found
    assert "BubbleSort" in symbols_found or "QuickSort" in symbols_found


def test_no_false_positives_on_utils() -> None:
    pf = _parse_file("utils.py", SIMPLE_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph, [])
    assert patterns == [], f"Expected no patterns on utils.py, got: {[p.name for p in patterns]}"


def test_confidence_in_range() -> None:
    pfs = [
        _parse_file(name, PATTERNS_FIXTURE)
        for name in ["middleware.py", "plugins.py", "events.py", "repository.py", "strategies.py"]
    ]
    graph = _graph_for(pfs)
    patterns = detect_patterns(pfs, graph, [])
    for p in patterns:
        assert 0.0 <= p.confidence <= 1.0, f"Pattern '{p.name}' confidence out of range"


def test_evidence_symbols_non_empty() -> None:
    pfs = [
        _parse_file(name, PATTERNS_FIXTURE)
        for name in ["middleware.py", "plugins.py", "events.py", "repository.py", "strategies.py"]
    ]
    graph = _graph_for(pfs)
    patterns = detect_patterns(pfs, graph, [])
    for p in patterns:
        for ev in p.evidence:
            assert ev.symbol, f"Pattern '{p.name}' has evidence with empty symbol"
