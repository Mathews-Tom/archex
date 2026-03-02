from __future__ import annotations

from archex.models import (
    ArchDecision,
    ArchProfile,
    CodeChunk,
    ContextBundle,
    DetectedPattern,
    EdgeKind,
    Interface,
    Module,
    PatternCategory,
    PatternEvidence,
    RankedChunk,
    RepoMetadata,
    Symbol,
    SymbolKind,
    SymbolRef,
    Visibility,
)


def test_symbol_kind_members() -> None:
    assert SymbolKind.FUNCTION == "function"
    assert SymbolKind.CLASS == "class"
    assert SymbolKind.METHOD == "method"
    assert SymbolKind.TYPE == "type"
    assert SymbolKind.VARIABLE == "variable"


def test_visibility_members() -> None:
    assert Visibility.PUBLIC == "public"
    assert Visibility.INTERNAL == "internal"
    assert Visibility.PRIVATE == "private"


def test_edge_kind_members() -> None:
    assert EdgeKind.IMPORTS == "imports"
    assert EdgeKind.CALLS == "calls"
    assert EdgeKind.INHERITS == "inherits"
    assert EdgeKind.IMPLEMENTS == "implements"


def test_pattern_category_members() -> None:
    assert PatternCategory.STRUCTURAL == "structural"
    assert PatternCategory.BEHAVIORAL == "behavioral"
    assert PatternCategory.CREATIONAL == "creational"


def test_repo_metadata_instantiation() -> None:
    meta = RepoMetadata(
        url="https://github.com/test/repo",
        total_files=10,
        total_lines=500,
    )
    assert meta.url == "https://github.com/test/repo"
    assert meta.total_files == 10
    assert meta.total_lines == 500


def test_symbol_instantiation() -> None:
    symbol = Symbol(
        name="my_function",
        qualified_name="module.my_function",
        kind=SymbolKind.FUNCTION,
        file_path="src/module.py",
        start_line=10,
        end_line=15,
        visibility=Visibility.PUBLIC,
    )
    assert symbol.name == "my_function"
    assert symbol.kind == SymbolKind.FUNCTION
    assert symbol.visibility == Visibility.PUBLIC


def test_code_chunk_instantiation() -> None:
    chunk = CodeChunk(
        id="chunk-1",
        content="def foo(): pass",
        file_path="src/module.py",
        start_line=1,
        end_line=1,
        language="python",
    )
    assert chunk.content == "def foo(): pass"
    assert chunk.file_path == "src/module.py"
    assert chunk.start_line == 1


def test_arch_profile_minimal() -> None:
    profile = ArchProfile(repo=RepoMetadata())
    assert profile.module_map == []
    assert profile.pattern_catalog == []
    assert profile.stats.total_files == 0


# ---------------------------------------------------------------------------
# ArchProfile.to_markdown() edge-case coverage
# ---------------------------------------------------------------------------


def test_arch_profile_to_dict() -> None:
    """to_dict() returns a plain dict with expected top-level keys."""
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"))
    result = profile.to_dict()
    assert isinstance(result, dict)
    assert "repo" in result
    assert "module_map" in result
    assert "pattern_catalog" in result


def test_arch_profile_to_json() -> None:
    """to_json() returns a valid JSON string."""
    import json

    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"))
    result = profile.to_json()
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert "repo" in parsed


def test_to_markdown_commit_hash_rendered() -> None:
    """Commit hash line appears when repo.commit_hash is set."""
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo", commit_hash="abc123"))
    md = profile.to_markdown()
    assert "abc123" in md
    assert "**Commit:**" in md


def test_to_markdown_patterns_with_evidence() -> None:
    """Pattern rows include evidence count from pat.evidence list."""
    evidence = PatternEvidence(
        file_path="src/factory.py",
        start_line=10,
        end_line=20,
        symbol="create_widget",
        explanation="Returns new instances",
    )
    pattern = DetectedPattern(
        name="factory_method",
        display_name="Factory Method",
        confidence=0.85,
        evidence=[evidence, evidence],
        description="Creates objects via factory",
        category=PatternCategory.CREATIONAL,
    )
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"), pattern_catalog=[pattern])
    md = profile.to_markdown()
    assert "Factory Method" in md
    assert "2 items" in md
    assert "85%" in md


def test_to_markdown_interface_surface() -> None:
    """Interface surface entries appear with signature and file path."""
    sym_ref = SymbolRef(
        name="MyService",
        qualified_name="mymod.MyService",
        file_path="src/service.py",
        kind=SymbolKind.CLASS,
    )
    iface = Interface(symbol=sym_ref, signature="def process(self, data: bytes) -> None")
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"), interface_surface=[iface])
    md = profile.to_markdown()
    assert "## Interface Surface" in md
    assert "def process(self, data: bytes) -> None" in md
    assert "src/service.py" in md


def test_to_markdown_decision_log_with_alternatives() -> None:
    """Decision log entries render alternatives when present."""
    decision = ArchDecision(
        decision="Use SQLite for persistence",
        alternatives=["PostgreSQL", "DynamoDB"],
        source="structural",
    )
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"), decision_log=[decision])
    md = profile.to_markdown()
    assert "## Architecture Decisions" in md
    assert "Use SQLite for persistence" in md
    assert "PostgreSQL" in md
    assert "DynamoDB" in md


def test_to_markdown_decision_log_no_alternatives() -> None:
    """Decision log entries without alternatives render without the alternatives line."""
    decision = ArchDecision(
        decision="Monorepo layout",
        alternatives=[],
        source="llm_inferred",
    )
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"), decision_log=[decision])
    md = profile.to_markdown()
    assert "Monorepo layout" in md
    assert "Alternatives:" not in md


def test_to_markdown_modules_with_exports() -> None:
    """Module section includes export names when exports are present."""
    sym_ref = SymbolRef(
        name="Widget",
        qualified_name="widgets.Widget",
        file_path="src/widgets.py",
        kind=SymbolKind.CLASS,
    )
    mod = Module(
        name="widgets",
        root_path="src/widgets",
        exports=[sym_ref],
        file_count=3,
        line_count=120,
        cohesion_score=0.75,
    )
    profile = ArchProfile(repo=RepoMetadata(local_path="/repo"), module_map=[mod])
    md = profile.to_markdown()
    assert "## Modules" in md
    assert "### widgets" in md
    assert "`Widget`" in md


# ---------------------------------------------------------------------------
# ContextBundle.to_prompt() format branches
# ---------------------------------------------------------------------------


def _make_bundle(query: str = "test query") -> ContextBundle:
    chunk = CodeChunk(
        id="f.py:fn:1",
        content="def fn(): pass",
        file_path="f.py",
        start_line=1,
        end_line=1,
        language="python",
        symbol_name="fn",
    )
    ranked = RankedChunk(chunk=chunk, final_score=0.9)
    return ContextBundle(query=query, chunks=[ranked], token_count=10, token_budget=500)


def test_context_bundle_to_prompt_xml() -> None:
    """to_prompt('xml') produces XML-structured output."""
    bundle = _make_bundle("explain caching")
    result = bundle.to_prompt(format="xml")
    assert "<" in result  # XML tags present
    assert "explain caching" in result


def test_context_bundle_to_prompt_markdown() -> None:
    """to_prompt('markdown') produces Markdown-structured output."""
    bundle = _make_bundle("how does auth work")
    result = bundle.to_prompt(format="markdown")
    assert "# Context:" in result
    assert "how does auth work" in result
    assert "def fn(): pass" in result


def test_context_bundle_to_prompt_json() -> None:
    """to_prompt('json') produces JSON-serialisable string."""
    import json

    bundle = _make_bundle("list modules")
    result = bundle.to_prompt(format="json")
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


def test_context_bundle_to_prompt_unknown_format_raises() -> None:
    """to_prompt raises ValueError for unrecognised format."""
    import pytest as _pytest

    bundle = _make_bundle()
    with _pytest.raises(ValueError, match="Unknown format"):
        bundle.to_prompt(format="yaml")


# ---------------------------------------------------------------------------
# Precision Symbol Tool models
# ---------------------------------------------------------------------------


def test_file_tree_entry_instantiation() -> None:
    from archex.models import FileTreeEntry

    entry = FileTreeEntry(path="src/main.py", language="python", lines=50, symbol_count=3)
    assert entry.path == "src/main.py"
    assert entry.is_directory is False
    assert entry.children == []


def test_file_tree_entry_with_children() -> None:
    from archex.models import FileTreeEntry

    child = FileTreeEntry(path="src/utils.py", language="python", lines=30, symbol_count=2)
    parent = FileTreeEntry(path="src", is_directory=True, children=[child])
    assert parent.is_directory is True
    assert len(parent.children) == 1
    assert parent.children[0].path == "src/utils.py"


def test_file_tree_instantiation() -> None:
    from archex.models import FileTree, FileTreeEntry

    entry = FileTreeEntry(path="main.py", language="python", lines=100, symbol_count=5)
    tree = FileTree(root="/repo", entries=[entry], total_files=1, languages={"python": 1})
    assert tree.root == "/repo"
    assert tree.total_files == 1
    assert tree.languages["python"] == 1


def test_symbol_outline_instantiation() -> None:
    from archex.models import SymbolKind, SymbolOutline, Visibility

    outline = SymbolOutline(
        symbol_id="src/main.py::MyClass#class",
        name="MyClass",
        kind=SymbolKind.CLASS,
        file_path="src/main.py",
        start_line=10,
        end_line=50,
        signature="class MyClass",
        visibility=Visibility.PUBLIC,
    )
    assert outline.symbol_id == "src/main.py::MyClass#class"
    assert outline.children == []


def test_symbol_outline_with_children() -> None:
    from archex.models import SymbolKind, SymbolOutline

    child = SymbolOutline(
        symbol_id="f.py::Cls.method#method",
        name="method",
        kind=SymbolKind.METHOD,
        file_path="f.py",
        start_line=5,
        end_line=10,
    )
    parent = SymbolOutline(
        symbol_id="f.py::Cls#class",
        name="Cls",
        kind=SymbolKind.CLASS,
        file_path="f.py",
        start_line=1,
        end_line=10,
        children=[child],
    )
    assert len(parent.children) == 1


def test_file_outline_instantiation() -> None:
    from archex.models import FileOutline, SymbolKind, SymbolOutline

    sym = SymbolOutline(
        symbol_id="f.py::foo#function",
        name="foo",
        kind=SymbolKind.FUNCTION,
        file_path="f.py",
        start_line=1,
        end_line=5,
    )
    outline = FileOutline(
        file_path="f.py", language="python", lines=100, symbols=[sym], token_count_raw=500
    )
    assert outline.file_path == "f.py"
    assert outline.token_count_raw == 500
    assert len(outline.symbols) == 1


def test_symbol_match_instantiation() -> None:
    from archex.models import SymbolKind, SymbolMatch

    match = SymbolMatch(
        symbol_id="f.py::foo#function",
        name="foo",
        kind=SymbolKind.FUNCTION,
        file_path="f.py",
        start_line=1,
    )
    assert match.relevance_score == 0.0
    assert match.visibility == "public"


def test_symbol_source_instantiation() -> None:
    from archex.models import SymbolKind, SymbolSource

    source = SymbolSource(
        symbol_id="f.py::foo#function",
        name="foo",
        kind=SymbolKind.FUNCTION,
        file_path="f.py",
        start_line=1,
        end_line=5,
        source="def foo(): pass",
        token_count=10,
    )
    assert source.source == "def foo(): pass"
    assert source.imports_context == ""


def test_symbol_source_round_trip() -> None:
    import json

    from archex.models import SymbolKind, SymbolSource

    source = SymbolSource(
        symbol_id="f.py::foo#function",
        name="foo",
        kind=SymbolKind.FUNCTION,
        file_path="f.py",
        start_line=1,
        end_line=5,
        source="def foo(): pass",
    )
    data = json.loads(source.model_dump_json())
    restored = SymbolSource(**data)
    assert restored.symbol_id == source.symbol_id
    assert restored.source == source.source
