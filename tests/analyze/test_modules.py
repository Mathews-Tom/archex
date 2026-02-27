"""Tests for analyze/modules.py: module boundary detection via Louvain clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.analyze.modules import detect_modules
from archex.index.graph import DependencyGraph
from archex.models import ParsedFile
from archex.parse import (
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.parse.adapters.python import PythonAdapter

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph_and_parsed(repo_path: Path) -> tuple[DependencyGraph, list[ParsedFile]]:
    """Full pipeline: discover + parse + build dependency graph for a local repo."""
    from archex.acquire import discover_files

    files = discover_files(repo_path, languages=["python"])
    engine = TreeSitterEngine()
    adapters = {"python": PythonAdapter()}

    parsed_files = extract_symbols(files, engine, adapters)
    import_map = parse_imports(files, engine, adapters)
    file_map = build_file_map(files)
    file_languages = {f.path: f.language for f in files}
    resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

    graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)
    return graph, parsed_files


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_modules_returns_at_least_one(python_simple_repo: Path) -> None:
    graph, parsed_files = _build_graph_and_parsed(python_simple_repo)
    modules = detect_modules(graph, parsed_files)
    assert len(modules) >= 1


def test_detect_modules_files_are_subset_of_parsed(python_simple_repo: Path) -> None:
    graph, parsed_files = _build_graph_and_parsed(python_simple_repo)
    all_paths = {pf.path for pf in parsed_files}
    modules = detect_modules(graph, parsed_files)
    for module in modules:
        for f in module.files:
            assert f in all_paths, f"Module file '{f}' not in parsed files"


def test_detect_modules_cohesion_in_range(python_simple_repo: Path) -> None:
    graph, parsed_files = _build_graph_and_parsed(python_simple_repo)
    modules = detect_modules(graph, parsed_files)
    for module in modules:
        assert 0.0 <= module.cohesion_score <= 1.0, (
            f"Module '{module.name}' has out-of-range cohesion: {module.cohesion_score}"
        )


def test_detect_modules_file_count_matches(python_simple_repo: Path) -> None:
    graph, parsed_files = _build_graph_and_parsed(python_simple_repo)
    modules = detect_modules(graph, parsed_files)
    for module in modules:
        assert module.file_count == len(module.files)


def test_detect_modules_covers_all_files(python_simple_repo: Path) -> None:
    graph, parsed_files = _build_graph_and_parsed(python_simple_repo)
    modules = detect_modules(graph, parsed_files)
    covered = {f for module in modules for f in module.files}
    all_paths = {pf.path for pf in parsed_files}
    assert covered == all_paths, "Not all parsed files are covered by modules"


def test_detect_modules_name_inferred(python_simple_repo: Path) -> None:
    graph, parsed_files = _build_graph_and_parsed(python_simple_repo)
    modules = detect_modules(graph, parsed_files)
    for module in modules:
        assert module.name, "Module name should not be empty"
        assert isinstance(module.name, str)


def test_detect_modules_single_file() -> None:
    """Single-file repo should yield exactly 1 module."""
    pf = ParsedFile(path="module/solo.py", language="python", lines=10)
    graph = DependencyGraph()
    graph.add_file_node("module/solo.py")

    modules = detect_modules(graph, [pf])
    assert len(modules) == 1
    assert modules[0].file_count == 1
    assert modules[0].files == ["module/solo.py"]


def test_detect_modules_empty_input() -> None:
    graph = DependencyGraph()
    modules = detect_modules(graph, [])
    assert modules == []
