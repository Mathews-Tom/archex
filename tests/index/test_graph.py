from __future__ import annotations

from typing import TYPE_CHECKING

from archex import languages
from archex.index.graph import DependencyGraph
from archex.languages import LanguageSupport, get_language_tier
from archex.models import (
    Edge,
    EdgeConfidence,
    EdgeKind,
    ImportStatement,
    LanguageTier,
    ParsedFile,
    Symbol,
    SymbolKind,
    Visibility,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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


def test_resolved_import_edges_are_extracted_with_evidence() -> None:
    graph = DependencyGraph.from_parsed_files(_make_parsed_files(), _make_import_map())

    [edge] = graph.file_edges()

    assert edge.confidence == EdgeConfidence.EXTRACTED
    assert edge.confidence_score == 1.0
    assert edge.evidence == ["resolved import 'models' at main.py:1"]


def test_structured_tier_parsed_file_emits_import_edge_without_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = LanguageSupport(
        language_id="structured_stub",
        display_name="Structured Stub",
        extensions=(".structstub",),
        tier=LanguageTier.STRUCTURED,
        pack_name="structured_stub",
        chunk_node_types=frozenset({"section"}),
    )
    monkeypatch.setitem(languages.LANGUAGE_SUPPORT, "structured_stub", stub)
    assert get_language_tier("structured_stub") == LanguageTier.STRUCTURED

    parsed_files = [
        ParsedFile(path="pkg/main.structstub", language="structured_stub", symbols=[], lines=6),
        ParsedFile(path="pkg/shared.structstub", language="structured_stub", symbols=[], lines=3),
    ]
    import_map = {
        "pkg/main.structstub": [
            ImportStatement(
                module="./shared.structstub",
                file_path="pkg/main.structstub",
                line=1,
                is_relative=True,
                resolved_path="pkg/shared.structstub",
            ),
        ],
        "pkg/shared.structstub": [],
    }

    graph = DependencyGraph.from_parsed_files(parsed_files, import_map)

    assert graph.symbol_count == 0
    [edge] = graph.file_edges()
    assert edge.source == "pkg/main.structstub"
    assert edge.target == "pkg/shared.structstub"
    assert edge.kind == EdgeKind.IMPORTS
    assert edge.confidence == EdgeConfidence.EXTRACTED


def test_cross_language_html_to_js_and_css_edges_resolve_to_correct_targets(
    tmp_path: Path,
) -> None:
    """An HTML file's `script src`/`link href` references become IMPORTS
    edges to the exact JS/CSS files they name, even when the JS and CSS
    sibling both share a directory and basename -- a collision that a
    naive extension-stripped module-key lookup would resolve to only one
    of the two targets, silently dropping (or misdirecting) the other."""
    from archex.models import Config
    from archex.parse.adapters import default_adapter_registry
    from archex.pipeline.service import parse_repository

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(
        '<html><head><link rel="stylesheet" href="assets/app.css">'
        '<script src="assets/app.js"></script></head><body></body></html>'
    )
    (tmp_path / "assets" / "app.js").write_text("console.log('app');\n")
    (tmp_path / "assets" / "app.css").write_text("body { color: black; }\n")

    config = Config(languages=["html", "javascript", "css"], parallel=False)
    adapters = default_adapter_registry.build_all()
    artifacts = parse_repository(tmp_path, config, adapters)

    graph = DependencyGraph.from_parsed_files(artifacts.parsed_files, artifacts.resolved_imports)

    assert graph.imports_of("index.html") == {"assets/app.js", "assets/app.css"}


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


def test_structural_centrality_cache_result_stays_stable() -> None:
    graph = DependencyGraph.from_edges(
        [
            Edge(source="a.py", target="b.py", kind=EdgeKind.IMPORTS),
            Edge(source="c.py", target="b.py", kind=EdgeKind.IMPORTS),
        ]
    )

    first = graph.structural_centrality()
    assert graph._centrality_cache is first  # pyright: ignore[reportPrivateUsage]

    second = graph.structural_centrality()

    assert second is first
    assert second == first


def test_sqlite_round_trip(tmp_path: Path) -> None:
    parsed = _make_parsed_files()
    import_map = _make_import_map()
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    db_path = tmp_path / "graph.db"
    graph.to_sqlite(db_path)

    restored = DependencyGraph.from_sqlite(db_path)
    assert restored.file_count == graph.file_count
    assert restored.file_edge_count == graph.file_edge_count


def test_sqlite_round_trip_preserves_edge_confidence(tmp_path: Path) -> None:
    edge = Edge(
        source="a.py",
        target="b.py",
        kind=EdgeKind.IMPORTS,
        confidence=EdgeConfidence.INFERRED,
        confidence_score=0.4,
        evidence=["external analyzer matched symbol"],
    )
    graph = DependencyGraph.from_edges([edge])

    db_path = tmp_path / "graph-confidence.db"
    graph.to_sqlite(db_path)
    restored = DependencyGraph.from_sqlite(db_path)

    [restored_edge] = restored.file_edges()
    assert restored_edge.confidence == EdgeConfidence.INFERRED
    assert restored_edge.confidence_score == 0.4
    assert restored_edge.evidence == ["external analyzer matched symbol"]


def test_old_sqlite_graph_migrates_edge_confidence(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "old-graph.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE files (path TEXT PRIMARY KEY);
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT, location TEXT);
        INSERT INTO files (path) VALUES ('a.py'), ('b.py');
        INSERT INTO edges (source, target, kind, location)
        VALUES ('a.py', 'b.py', 'imports', 'a.py:1');
    """)
    conn.commit()
    conn.close()

    graph = DependencyGraph.from_sqlite(db_path)
    [edge] = graph.file_edges()

    assert edge.confidence == EdgeConfidence.EXTRACTED
    assert edge.confidence_score == 1.0
    assert edge.evidence == []


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

    def test_ambiguous_edges_do_not_drive_traversal(self) -> None:
        graph = DependencyGraph.from_edges(
            [
                Edge(
                    source="a.py",
                    target="b.py",
                    kind=EdgeKind.IMPORTS,
                    confidence=EdgeConfidence.AMBIGUOUS,
                    confidence_score=0.2,
                    evidence=["multiple possible import targets"],
                )
            ]
        )

        assert graph.imports_of("a.py") == set()
        assert graph.imported_by("b.py") == set()
        assert graph.neighborhood("a.py") == set()

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


# ---------------------------------------------------------------------------
# Co-directory edges
# ---------------------------------------------------------------------------


class TestCoDirectoryEdges:
    def test_adds_edges_for_same_directory(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("pkg/a.go")
        graph.add_file_node("pkg/b.go")
        graph.add_file_node("pkg/c.go")

        added = graph.add_co_directory_edges()
        # 3 files → 3 pairs × 2 directions = 6 edges
        assert added == 6
        assert graph.file_edge_count == 6
        assert "pkg/b.go" in graph.imports_of("pkg/a.go")
        assert "pkg/a.go" in graph.imports_of("pkg/b.go")

    def test_no_edges_across_directories(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("pkg1/a.go")
        graph.add_file_node("pkg2/b.go")

        added = graph.add_co_directory_edges()
        assert added == 0
        assert graph.file_edge_count == 0

    def test_skips_existing_edges(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("pkg/a.go")
        graph.add_file_node("pkg/b.go")
        graph.add_file_edge("pkg/a.go", "pkg/b.go", kind="imports")

        added = graph.add_co_directory_edges()
        # a→b already exists, so only b→a added
        assert added == 1
        assert graph.file_edge_count == 2

    def test_single_file_directory_no_edges(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("pkg/alone.go")

        added = graph.add_co_directory_edges()
        assert added == 0

    def test_root_directory_files(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("main.go")
        graph.add_file_node("utils.go")

        added = graph.add_co_directory_edges()
        assert added == 2
        assert "utils.go" in graph.imports_of("main.go")

    def test_invalidates_centrality_cache(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("a.go")
        graph.add_file_node("b.go")
        _ = graph.structural_centrality()
        assert graph._centrality_cache is not None  # pyright: ignore[reportPrivateUsage]

        graph.add_co_directory_edges()
        assert graph._centrality_cache is None  # pyright: ignore[reportPrivateUsage]

    def test_edge_kind_is_co_directory(self) -> None:
        from archex.models import EdgeKind

        graph = DependencyGraph()
        graph.add_file_node("pkg/a.go")
        graph.add_file_node("pkg/b.go")
        graph.add_co_directory_edges()

        edges = graph.file_edges()
        assert all(e.kind == EdgeKind.CO_DIRECTORY for e in edges)

    def test_co_directory_edges_are_heuristic_with_evidence(self) -> None:
        graph = DependencyGraph()
        graph.add_file_node("pkg/a.go")
        graph.add_file_node("pkg/b.go")

        graph.add_co_directory_edges()
        edges = graph.file_edges()

        assert {edge.confidence for edge in edges} == {EdgeConfidence.HEURISTIC}
        assert {edge.confidence_score for edge in edges} == {0.6}
        assert all(edge.evidence for edge in edges)

    def test_bounds_edges_on_large_flat_directory(self) -> None:
        from archex.index.graph import (
            _CO_DIRECTORY_DENSE_THRESHOLD,  # pyright: ignore[reportPrivateUsage]
            _CO_DIRECTORY_WINDOW_SIZE,  # pyright: ignore[reportPrivateUsage]
        )

        graph = DependencyGraph()
        file_count = 500
        for i in range(file_count):
            graph.add_file_node(f"flat/f{i:03d}.py")

        added = graph.add_co_directory_edges()

        full_pairwise = file_count * (file_count - 1)  # both directions, unbounded
        assert file_count > _CO_DIRECTORY_DENSE_THRESHOLD
        # Windowed bound: each file links to at most _CO_DIRECTORY_WINDOW_SIZE
        # forward neighbors, both directions.
        max_windowed = file_count * _CO_DIRECTORY_WINDOW_SIZE * 2
        assert added > 0
        assert added <= max_windowed
        assert added < full_pairwise // 10
