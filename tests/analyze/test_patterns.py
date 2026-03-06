"""Tests for analyze/patterns.py: architectural pattern detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from archex.analyze.patterns import (
    PatternRegistry,
    _confidence,  # pyright: ignore[reportPrivateUsage]
    _method_names,  # pyright: ignore[reportPrivateUsage]
    detect_patterns,
)
from archex.index.graph import DependencyGraph
from archex.models import ParsedFile, Symbol, SymbolKind, Visibility
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
    patterns = detect_patterns(parsed_files, graph)
    return {p.name for p in patterns}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_middleware_chain_detected() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    names = {p.name for p in patterns}
    assert "middleware_chain" in names


def test_middleware_chain_high_confidence() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    pattern = next(p for p in patterns if p.name == "middleware_chain")
    assert pattern.confidence >= 0.6


def test_middleware_chain_evidence_has_correct_file() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    pattern = next(p for p in patterns if p.name == "middleware_chain")
    for ev in pattern.evidence:
        assert ev.file_path == "middleware.py"


def test_middleware_chain_evidence_has_valid_lines() -> None:
    pf = _parse_file("middleware.py", PATTERNS_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
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
    patterns = detect_patterns([pf], graph)
    pattern = next(p for p in patterns if p.name == "strategy")
    symbols_found = {ev.symbol for ev in pattern.evidence}
    # SortStrategy is the protocol; BubbleSort/QuickSort are concretes
    assert "SortStrategy" in symbols_found
    assert "BubbleSort" in symbols_found or "QuickSort" in symbols_found


def test_no_false_positives_on_utils() -> None:
    pf = _parse_file("utils.py", SIMPLE_FIXTURE)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    assert patterns == [], f"Expected no patterns on utils.py, got: {[p.name for p in patterns]}"


def test_confidence_in_range() -> None:
    pfs = [
        _parse_file(name, PATTERNS_FIXTURE)
        for name in ["middleware.py", "plugins.py", "events.py", "repository.py", "strategies.py"]
    ]
    graph = _graph_for(pfs)
    patterns = detect_patterns(pfs, graph)
    for p in patterns:
        assert 0.0 <= p.confidence <= 1.0, f"Pattern '{p.name}' confidence out of range"


def test_evidence_symbols_non_empty() -> None:
    pfs = [
        _parse_file(name, PATTERNS_FIXTURE)
        for name in ["middleware.py", "plugins.py", "events.py", "repository.py", "strategies.py"]
    ]
    graph = _graph_for(pfs)
    patterns = detect_patterns(pfs, graph)
    for p in patterns:
        for ev in p.evidence:
            assert ev.symbol, f"Pattern '{p.name}' has evidence with empty symbol"


# ---------------------------------------------------------------------------
# PatternRegistry tests
# ---------------------------------------------------------------------------


def test_pattern_registry_add() -> None:
    registry = PatternRegistry()

    def my_detector(parsed_files: list[ParsedFile], graph: DependencyGraph) -> None:
        return None

    assert registry.detectors == []
    registry.add(my_detector)  # pyright: ignore[reportArgumentType]
    assert registry.detectors == [my_detector]


def test_pattern_registry_load_entry_points_success() -> None:
    registry = PatternRegistry()

    def fake_detector(parsed_files: list[ParsedFile], graph: DependencyGraph) -> None:
        return None

    ep = MagicMock()
    ep.name = "fake_detector"
    ep.load.return_value = fake_detector

    with patch("archex.analyze.patterns.importlib.metadata.entry_points", return_value=[ep]):
        registry.load_entry_points()

    assert fake_detector in registry.detectors


def test_pattern_registry_load_entry_points_failure() -> None:
    registry = PatternRegistry()

    ep = MagicMock()
    ep.name = "broken_detector"
    ep.load.side_effect = ImportError("missing dep")

    with patch("archex.analyze.patterns.importlib.metadata.entry_points", return_value=[ep]):
        registry.load_entry_points()

    assert registry.detectors == []


# ---------------------------------------------------------------------------
# _confidence edge case
# ---------------------------------------------------------------------------


def test_confidence_zero_evidence() -> None:
    assert _confidence(0) == 0.0  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Event bus partial detection
# ---------------------------------------------------------------------------


def _make_symbol(
    name: str,
    kind: SymbolKind,
    file_path: str,
    start_line: int,
    end_line: int,
    parent: str | None = None,
    visibility: Visibility = Visibility.PUBLIC,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=f"{file_path}:{name}",
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        visibility=visibility,
        parent=parent,
    )


def test_event_bus_partial_only_subscribe() -> None:
    file_path = "partial_events.py"
    symbols = [
        _make_symbol("Subscriber", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("subscribe", SymbolKind.METHOD, file_path, 2, 4, parent="Subscriber"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    event_bus = next((p for p in patterns if p.name == "event_bus"), None)
    assert event_bus is not None
    symbols_found = {ev.symbol for ev in event_bus.evidence}
    assert "Subscriber" in symbols_found


def test_event_bus_partial_only_emit() -> None:
    file_path = "partial_emitter.py"
    symbols = [
        _make_symbol("Emitter", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("emit", SymbolKind.METHOD, file_path, 2, 4, parent="Emitter"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    event_bus = next((p for p in patterns if p.name == "event_bus"), None)
    assert event_bus is not None
    symbols_found = {ev.symbol for ev in event_bus.evidence}
    assert "Emitter" in symbols_found


# ---------------------------------------------------------------------------
# Strategy fallback: protocol found, concretes via method matching
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _method_names case normalization
# ---------------------------------------------------------------------------


def test_method_names_returns_lowercase() -> None:
    file_path = "test.py"
    symbols = [
        _make_symbol("MyClass", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("SetNext", SymbolKind.METHOD, file_path, 2, 3, parent="MyClass"),
        _make_symbol("Process", SymbolKind.METHOD, file_path, 4, 5, parent="MyClass"),
        _make_symbol("handleRequest", SymbolKind.METHOD, file_path, 6, 7, parent="MyClass"),
    ]
    names = _method_names("MyClass", symbols)  # pyright: ignore[reportPrivateUsage]
    assert names == {"setnext", "process", "handlerequest"}


# ---------------------------------------------------------------------------
# Cross-language detection (case-insensitive matching)
# ---------------------------------------------------------------------------


def test_middleware_detects_pascalcase_methods() -> None:
    """Go-style PascalCase methods like SetNext/Handle should trigger middleware detection."""
    file_path = "handler.go"
    symbols = [
        _make_symbol("BaseHandler", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("SetNext", SymbolKind.METHOD, file_path, 2, 4, parent="BaseHandler"),
        _make_symbol("Handle", SymbolKind.METHOD, file_path, 5, 8, parent="BaseHandler"),
        _make_symbol("AuthHandler", SymbolKind.CLASS, file_path, 12, 20),
        _make_symbol("Handle", SymbolKind.METHOD, file_path, 13, 18, parent="AuthHandler"),
    ]
    pf = ParsedFile(path=file_path, language="go", symbols=symbols)
    names = _pattern_names([pf])
    assert "middleware_chain" in names


def test_repository_detects_camelcase_methods() -> None:
    """Java-style camelCase CRUD methods should trigger repository detection."""
    file_path = "UserRepository.java"
    symbols = [
        _make_symbol("UserRepository", SymbolKind.CLASS, file_path, 1, 20),
        _make_symbol("findById", SymbolKind.METHOD, file_path, 2, 4, parent="UserRepository"),
        _make_symbol("findAll", SymbolKind.METHOD, file_path, 5, 7, parent="UserRepository"),
        _make_symbol("save", SymbolKind.METHOD, file_path, 8, 10, parent="UserRepository"),
        _make_symbol("delete", SymbolKind.METHOD, file_path, 11, 13, parent="UserRepository"),
    ]
    pf = ParsedFile(path=file_path, language="java", symbols=symbols)
    names = _pattern_names([pf])
    assert "repository" in names


# ---------------------------------------------------------------------------
# False-positive rejection
# ---------------------------------------------------------------------------


def test_repository_rejects_utility_with_get_all() -> None:
    """A class with only get+all but no repo-indicating name should NOT trigger repository."""
    file_path = "utils.py"
    symbols = [
        _make_symbol("ConfigManager", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("get", SymbolKind.METHOD, file_path, 2, 4, parent="ConfigManager"),
        _make_symbol("all", SymbolKind.METHOD, file_path, 5, 7, parent="ConfigManager"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    names = _pattern_names([pf])
    assert "repository" not in names


def test_repository_accepts_named_repo_with_two_methods() -> None:
    """A class named *Repository with 2 CRUD methods SHOULD trigger."""
    file_path = "repo.py"
    symbols = [
        _make_symbol("UserRepository", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("get", SymbolKind.METHOD, file_path, 2, 4, parent="UserRepository"),
        _make_symbol("save", SymbolKind.METHOD, file_path, 5, 7, parent="UserRepository"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    names = _pattern_names([pf])
    assert "repository" in names


def test_event_bus_rejects_on_only_generic_class() -> None:
    """A class with only 'on' method and generic name should NOT trigger event bus."""
    file_path = "widget.py"
    symbols = [
        _make_symbol("ButtonWidget", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("on", SymbolKind.METHOD, file_path, 2, 4, parent="ButtonWidget"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    names = _pattern_names([pf])
    assert "event_bus" not in names


def test_event_bus_accepts_on_with_event_class_name() -> None:
    """A class named *EventBus with only 'on' SHOULD still trigger."""
    file_path = "events.py"
    symbols = [
        _make_symbol("CustomEventBus", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("on", SymbolKind.METHOD, file_path, 2, 4, parent="CustomEventBus"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    names = _pattern_names([pf])
    assert "event_bus" in names


def test_middleware_rejects_process_only_generic_class() -> None:
    """A class with only 'process' and no chain-indicating name should NOT trigger middleware."""
    file_path = "processor.py"
    symbols = [
        _make_symbol("DataProcessor", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("process", SymbolKind.METHOD, file_path, 2, 4, parent="DataProcessor"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    names = _pattern_names([pf])
    assert "middleware_chain" not in names


def test_middleware_accepts_process_with_handler_name() -> None:
    """A class named *Handler with 'handle' SHOULD trigger middleware (chain participant)."""
    file_path = "handlers.py"
    symbols = [
        _make_symbol("RequestHandler", SymbolKind.CLASS, file_path, 1, 10),
        _make_symbol("handle", SymbolKind.METHOD, file_path, 2, 4, parent="RequestHandler"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    names = _pattern_names([pf])
    assert "middleware_chain" in names


# ---------------------------------------------------------------------------
# Strategy fallback: protocol found, concretes via method matching
# ---------------------------------------------------------------------------


def test_strategy_protocol_fallback_finds_concretes() -> None:
    # Setup: two protocol-named classes share a PUBLIC method ("execute"), making
    # shared_methods non-empty so the early-exit `continue` is not hit.
    # SortAlgorithm also has an INTERNAL "sort" method; because _public_methods_of
    # filters by PUBLIC visibility, the internal method is excluded from
    # method_to_classes, so "sort" does not appear in shared_methods.
    # Worker has only a PUBLIC "sort" method — it is absent from shared_methods,
    # so in_shared is False and it falls through all main-loop branches, leaving
    # concrete_candidates empty.  The fallback (lines 407-424) builds proto_methods
    # from raw pf.symbols with no visibility filter, picks up the internal "sort",
    # and matches it against Worker's PUBLIC "sort" via _method_names, adding Worker
    # to concrete_candidates.
    file_path = "fallback_strategy.py"
    symbols = [
        # Protocol 1: PUBLIC "execute" + INTERNAL "sort" → protocol_candidates
        _make_symbol("SortAlgorithm", SymbolKind.CLASS, file_path, 1, 6),
        _make_symbol("execute", SymbolKind.METHOD, file_path, 2, 3, parent="SortAlgorithm"),
        _make_symbol(
            "sort",
            SymbolKind.METHOD,
            file_path,
            4,
            5,
            parent="SortAlgorithm",
            visibility=Visibility.INTERNAL,
        ),
        # Protocol 2: PUBLIC "execute" → protocol_candidates; also makes
        # shared_methods["execute"] = [SortAlgorithm, SortInterface] (non-empty)
        _make_symbol("SortInterface", SymbolKind.CLASS, file_path, 8, 11),
        _make_symbol("execute", SymbolKind.METHOD, file_path, 9, 10, parent="SortInterface"),
        # Concrete: neutral name, PUBLIC "sort" only.
        # "sort" appears in method_to_classes only for Worker (SortAlgorithm's sort is
        # INTERNAL → excluded), so it is not in shared_methods → in_shared=False →
        # Worker falls through all main-loop branches → found by fallback instead.
        _make_symbol("Worker", SymbolKind.CLASS, file_path, 13, 17),
        _make_symbol("sort", SymbolKind.METHOD, file_path, 14, 16, parent="Worker"),
    ]
    pf = ParsedFile(path=file_path, language="python", symbols=symbols)
    graph = _graph_for([pf])
    patterns = detect_patterns([pf], graph)
    strategy = next((p for p in patterns if p.name == "strategy"), None)
    assert strategy is not None
    symbols_found = {ev.symbol for ev in strategy.evidence}
    assert "SortAlgorithm" in symbols_found
    assert "Worker" in symbols_found
