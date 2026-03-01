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


# ---------------------------------------------------------------------------
# _infer_module_name unit tests
# ---------------------------------------------------------------------------


def test_infer_module_name_empty_files() -> None:
    from archex.analyze.modules import _infer_module_name  # pyright: ignore[reportPrivateUsage]

    assert _infer_module_name([]) == "unknown"


def test_infer_module_name_single_file_with_parent() -> None:
    from archex.analyze.modules import _infer_module_name  # pyright: ignore[reportPrivateUsage]

    assert _infer_module_name(["src/utils/helpers.py"]) == "utils"


def test_infer_module_name_single_file_no_parent() -> None:
    from archex.analyze.modules import _infer_module_name  # pyright: ignore[reportPrivateUsage]

    assert _infer_module_name(["helpers.py"]) == "helpers"


def test_infer_module_name_common_prefix() -> None:
    from archex.analyze.modules import _infer_module_name  # pyright: ignore[reportPrivateUsage]

    assert _infer_module_name(["src/api/routes.py", "src/api/handlers.py"]) == "api"


# ---------------------------------------------------------------------------
# detect_modules edge-case tests
# ---------------------------------------------------------------------------


def test_detect_modules_louvain_fallback() -> None:
    """When louvain_communities raises, all files collapse into one module."""
    from unittest.mock import patch

    pf_a = ParsedFile(path="pkg/a.py", language="python", lines=5)
    pf_b = ParsedFile(path="pkg/b.py", language="python", lines=8)
    graph = DependencyGraph()
    graph.add_file_node("pkg/a.py")
    graph.add_file_node("pkg/b.py")
    graph.add_file_edge("pkg/a.py", "pkg/b.py")

    with patch("archex.analyze.modules.louvain_communities", side_effect=RuntimeError("fail")):
        modules = detect_modules(graph, [pf_a, pf_b])

    assert len(modules) == 1
    assert set(modules[0].files) == {"pkg/a.py", "pkg/b.py"}


def test_detect_modules_single_node_graph() -> None:
    """A graph with exactly one node must return exactly one module."""
    pf = ParsedFile(path="solo/main.py", language="python", lines=20)
    graph = DependencyGraph()
    graph.add_file_node("solo/main.py")

    modules = detect_modules(graph, [pf])

    assert len(modules) == 1
    assert modules[0].file_count == 1
    assert modules[0].files == ["solo/main.py"]
    assert modules[0].line_count == 20


def test_build_module_file_not_in_lookup() -> None:
    """_build_module_from_community must not crash when a file is absent from file_lookup."""
    import networkx as nx  # type: ignore[import-untyped]

    from archex.analyze.modules import (
        _build_module_from_community,  # pyright: ignore[reportPrivateUsage]
    )

    g: nx.Graph[str] = nx.Graph()
    g.add_node("ghost/missing.py")  # type: ignore[no-untyped-call]

    community: set[str] = {"ghost/missing.py"}
    file_lookup: dict[str, ParsedFile] = {}  # intentionally empty

    module = _build_module_from_community(community, g, file_lookup)

    assert module.line_count == 0
    assert module.files == ["ghost/missing.py"]


# ---------------------------------------------------------------------------
# _build_module_from_community: import resolution coverage (lines 102-110)
# ---------------------------------------------------------------------------


def test_build_module_import_resolution_all_branches() -> None:
    """Cover all three import-resolution branches in _build_module_from_community:
    - resolved=None + module name → external dep (lines 103-106)
    - resolved in community → internal dep (lines 107-108)
    - resolved not in community → external dep (lines 109-110)
    """
    import networkx as nx  # type: ignore[import-untyped]

    from archex.analyze.modules import (
        _build_module_from_community,  # pyright: ignore[reportPrivateUsage]
    )
    from archex.models import ImportStatement

    # "pkg/a.py" imports:
    #   1. requests.api → resolved=None, module="requests.api" → external "requests"
    #   2. pkg/b.py     → resolved="pkg/b.py" (in community) → internal dep
    #   3. other/c.py   → resolved="other/c.py" (not in community) → external dep
    pf_a = ParsedFile(
        path="pkg/a.py",
        language="python",
        lines=10,
        imports=[
            ImportStatement(
                module="requests.api",
                symbols=["get"],
                file_path="pkg/a.py",
                line=1,
                resolved_path=None,
            ),
            ImportStatement(
                module="pkg.b",
                symbols=["B"],
                file_path="pkg/a.py",
                line=2,
                resolved_path="pkg/b.py",
            ),
            ImportStatement(
                module="other.c",
                symbols=["C"],
                file_path="pkg/a.py",
                line=3,
                resolved_path="other/c.py",
            ),
        ],
    )
    pf_b = ParsedFile(path="pkg/b.py", language="python", lines=5)

    community: set[str] = {"pkg/a.py", "pkg/b.py"}
    file_lookup: dict[str, ParsedFile] = {"pkg/a.py": pf_a, "pkg/b.py": pf_b}

    g: nx.Graph[str] = nx.Graph()
    g.add_node("pkg/a.py")  # type: ignore[no-untyped-call]
    g.add_node("pkg/b.py")  # type: ignore[no-untyped-call]
    g.add_edge("pkg/a.py", "pkg/b.py")  # type: ignore[no-untyped-call]

    module = _build_module_from_community(community, g, file_lookup)

    assert "pkg/b.py" in module.internal_deps
    assert "requests" in module.external_deps
    assert "other/c.py" in module.external_deps
    assert "pkg/b.py" not in module.external_deps


def test_build_module_unresolved_import_empty_module_name() -> None:
    """resolved=None with empty module string must not add to external_deps."""
    import networkx as nx  # type: ignore[import-untyped]

    from archex.analyze.modules import (
        _build_module_from_community,  # pyright: ignore[reportPrivateUsage]
    )
    from archex.models import ImportStatement

    pf = ParsedFile(
        path="pkg/a.py",
        language="python",
        lines=3,
        imports=[
            ImportStatement(
                module="", symbols=["x"], file_path="pkg/a.py", line=1, resolved_path=None
            ),
        ],
    )
    community: set[str] = {"pkg/a.py"}
    file_lookup: dict[str, ParsedFile] = {"pkg/a.py": pf}

    g: nx.Graph[str] = nx.Graph()
    g.add_node("pkg/a.py")  # type: ignore[no-untyped-call]

    module = _build_module_from_community(community, g, file_lookup)

    assert module.external_deps == []
    assert module.internal_deps == []


# ---------------------------------------------------------------------------
# detect_modules: zero-nodes nx graph (line 143-144)
# ---------------------------------------------------------------------------


def test_detect_modules_zero_nx_nodes() -> None:
    """_build_nx_graph returns 0 nodes when parsed_files is non-empty but the mock
    replaces it with an empty graph, hitting the g.number_of_nodes() == 0 guard."""
    from unittest.mock import patch

    import networkx as nx  # type: ignore[import-untyped]

    pf = ParsedFile(path="orphan.py", language="python", lines=5)
    graph = DependencyGraph()

    empty_g: nx.Graph[str] = nx.Graph()

    with patch("archex.analyze.modules._build_nx_graph", return_value=empty_g):
        modules = detect_modules(graph, [pf])

    assert modules == []
