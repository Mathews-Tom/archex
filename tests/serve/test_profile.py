from __future__ import annotations

from archex.analyze.interfaces import extract_interfaces
from archex.index.graph import DependencyGraph
from archex.models import (
    ArchProfile,
    ImportStatement,
    ParsedFile,
    RepoMetadata,
    Symbol,
    SymbolKind,
    Visibility,
)
from archex.serve.profile import build_profile


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
                    signature="def main() -> None",
                ),
                Symbol(
                    name="_helper",
                    qualified_name="main._helper",
                    kind=SymbolKind.FUNCTION,
                    file_path="main.py",
                    start_line=7,
                    end_line=10,
                    visibility=Visibility.PRIVATE,
                ),
            ],
            lines=15,
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
                    signature="class User",
                )
            ],
            lines=25,
        ),
    ]


def _make_repo_metadata() -> RepoMetadata:
    return RepoMetadata(
        local_path="/tmp/test_repo",
        languages={"python": 2},
        total_files=2,
        total_lines=40,
    )


def test_build_profile_returns_arch_profile() -> None:
    parsed = _make_parsed_files()
    metadata = _make_repo_metadata()
    import_map: dict[str, list[ImportStatement]] = {"main.py": [], "models.py": []}
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    profile = build_profile(metadata, parsed, graph)
    assert isinstance(profile, ArchProfile)


def test_stats_computed_correctly() -> None:
    parsed = _make_parsed_files()
    metadata = _make_repo_metadata()
    import_map: dict[str, list[ImportStatement]] = {"main.py": [], "models.py": []}
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    profile = build_profile(metadata, parsed, graph)

    assert profile.stats.total_files == 2
    assert profile.stats.total_lines == 40
    assert profile.stats.symbol_count == 3
    assert "python" in profile.stats.languages
    assert profile.stats.languages["python"].files == 2


def test_interfaces_extracted_for_public_symbols() -> None:
    parsed = _make_parsed_files()
    metadata = _make_repo_metadata()
    import_map: dict[str, list[ImportStatement]] = {"main.py": [], "models.py": []}
    graph = DependencyGraph.from_parsed_files(parsed, import_map)
    interfaces = extract_interfaces(parsed, graph)

    profile = build_profile(metadata, parsed, graph, interfaces=interfaces)

    # Only public function/class symbols should appear
    interface_names = [iface.symbol.name for iface in profile.interface_surface]
    assert "main" in interface_names
    assert "User" in interface_names
    assert "_helper" not in interface_names


def test_dependency_graph_summary() -> None:
    parsed = _make_parsed_files()
    metadata = _make_repo_metadata()
    import_map: dict[str, list[ImportStatement]] = {
        "main.py": [
            ImportStatement(
                module="models",
                file_path="main.py",
                line=1,
                resolved_path="models.py",
            )
        ],
        "models.py": [],
    }
    graph = DependencyGraph.from_parsed_files(parsed, import_map)

    profile = build_profile(metadata, parsed, graph)

    assert profile.dependency_graph.file_count == 2
    assert profile.dependency_graph.edges == 1
