from __future__ import annotations

from typing import TYPE_CHECKING

from archex.index.graph import DependencyGraph
from archex.models import ImportStatement, ParsedFile, Symbol, SymbolKind, Visibility

if TYPE_CHECKING:
    from pathlib import Path


def _make_parsed_files() -> list[ParsedFile]:
    return [
        ParsedFile(
            path="main.py",
            language="python",
            symbols=[
                Symbol(
                    name="main",
                    qualified_name="main.main",
                    kind=SymbolKind.FUNCTION,
                    file_path="main.py",
                    start_line=1,
                    end_line=5,
                    visibility=Visibility.PUBLIC,
                )
            ],
            lines=10,
        ),
        ParsedFile(
            path="models.py",
            language="python",
            symbols=[
                Symbol(
                    name="User",
                    qualified_name="models.User",
                    kind=SymbolKind.CLASS,
                    file_path="models.py",
                    start_line=1,
                    end_line=20,
                    visibility=Visibility.PUBLIC,
                )
            ],
            lines=25,
        ),
        ParsedFile(
            path="utils.py",
            language="python",
            symbols=[],
            lines=5,
        ),
    ]


def _make_import_map() -> dict[str, list[ImportStatement]]:
    return {
        "main.py": [
            ImportStatement(
                module="models",
                file_path="main.py",
                line=1,
                resolved_path="models.py",
            ),
        ],
        "models.py": [],
        "utils.py": [],
    }


def test_node_counts() -> None:
    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    assert graph.file_count == 3
    assert graph.symbol_count == 2


def test_edge_count() -> None:
    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    assert graph.file_edge_count == 1


def test_file_edges_returns_edge_objects() -> None:
    from archex.models import Edge, EdgeKind

    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    edges = graph.file_edges()
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge, Edge)
    assert edge.source == "main.py"
    assert edge.target == "models.py"
    assert edge.kind == EdgeKind.IMPORTS


def test_neighborhood_bfs() -> None:
    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    neighbors = graph.neighborhood("main.py", hops=1)
    assert "models.py" in neighbors
    assert "main.py" not in neighbors


def test_neighborhood_missing_node() -> None:
    graph = DependencyGraph()
    assert graph.neighborhood("nonexistent.py") == set()


def test_structural_centrality() -> None:
    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    centrality = graph.structural_centrality()
    assert isinstance(centrality, dict)
    assert len(centrality) == 3
    for val in centrality.values():
        assert isinstance(val, float)


def test_sqlite_round_trip(tmp_path: Path) -> None:
    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    db_path = tmp_path / "graph.db"
    graph.to_sqlite(db_path)

    restored = DependencyGraph.from_sqlite(db_path)
    assert restored.file_count == graph.file_count
    assert restored.file_edge_count == graph.file_edge_count


# ---------------------------------------------------------------------------
# DependencyGraph.update_files
# ---------------------------------------------------------------------------


class TestUpdateFiles:
    def test_removes_node_and_edges(self) -> None:

        parsed = [
            ParsedFile(path="a.py", language="python"),
            ParsedFile(path="b.py", language="python"),
        ]
        import_map = {
            "a.py": [
                ImportStatement(
                    module="b",
                    symbols=[],
                    file_path="a.py",
                    line=1,
                    resolved_path="b.py",
                )
            ]
        }
        graph = DependencyGraph.from_parsed_files(parsed, import_map)
        assert graph.file_count == 2
        assert graph.file_edge_count == 1

        graph.update_files({"a.py"}, [])
        assert graph.file_count == 1
        assert graph.file_edge_count == 0

    def test_adds_new_edges(self) -> None:
        from archex.models import Edge, EdgeKind

        parsed = [
            ParsedFile(path="a.py", language="python"),
            ParsedFile(path="b.py", language="python"),
        ]
        graph = DependencyGraph.from_parsed_files(parsed, {})
        assert graph.file_edge_count == 0

        new_edges = [Edge(source="a.py", target="b.py", kind=EdgeKind.IMPORTS)]
        graph.update_files(set(), new_edges)
        assert graph.file_edge_count == 1

    def test_invalidates_centrality(self) -> None:

        parsed = [
            ParsedFile(path="a.py", language="python"),
            ParsedFile(path="b.py", language="python"),
        ]
        graph = DependencyGraph.from_parsed_files(parsed, {})
        _ = graph.structural_centrality()  # populate cache
        assert graph._centrality_cache is not None  # pyright: ignore[reportPrivateUsage]

        graph.update_files({"a.py"}, [])
        assert graph._centrality_cache is None  # pyright: ignore[reportPrivateUsage]

    def test_empty_inputs(self) -> None:
        parsed = [ParsedFile(path="a.py", language="python")]
        graph = DependencyGraph.from_parsed_files(parsed, {})
        graph.update_files(set(), [])
        assert graph.file_count == 1

    def test_removes_only_specified_node(self) -> None:
        parsed = [
            ParsedFile(path="a.py", language="python"),
            ParsedFile(path="b.py", language="python"),
            ParsedFile(path="c.py", language="python"),
        ]
        graph = DependencyGraph.from_parsed_files(parsed, {})
        graph.update_files({"b.py"}, [])
        assert graph.file_count == 2
        assert "a.py" in graph._file_graph.nodes()  # pyright: ignore[reportPrivateUsage]
        assert "c.py" in graph._file_graph.nodes()  # pyright: ignore[reportPrivateUsage]
        assert "b.py" not in graph._file_graph.nodes()  # pyright: ignore[reportPrivateUsage]

    def test_remove_nonexistent_path_is_safe(self) -> None:
        parsed = [ParsedFile(path="a.py", language="python")]
        graph = DependencyGraph.from_parsed_files(parsed, {})
        # Should not raise even if path doesn't exist in graph
        graph.update_files({"nonexistent.py"}, [])
        assert graph.file_count == 1

    def test_replace_modified_file_edges(self) -> None:
        from archex.models import Edge, EdgeKind

        parsed = [
            ParsedFile(path="a.py", language="python"),
            ParsedFile(path="b.py", language="python"),
            ParsedFile(path="c.py", language="python"),
        ]
        import_map = {
            "a.py": [
                ImportStatement(
                    module="b",
                    symbols=[],
                    file_path="a.py",
                    line=1,
                    resolved_path="b.py",
                )
            ]
        }
        graph = DependencyGraph.from_parsed_files(parsed, import_map)
        assert graph.file_edge_count == 1

        # Remove old a.py edges, add new a.py -> c.py edge
        new_edges = [Edge(source="a.py", target="c.py", kind=EdgeKind.IMPORTS)]
        graph.update_files({"a.py"}, new_edges)

        edges = graph.file_edges()
        assert len(edges) == 1
        assert edges[0].source == "a.py"
        assert edges[0].target == "c.py"
