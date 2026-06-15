from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.java import JavaAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "java_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> JavaAdapter:
    return JavaAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "java")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: JavaAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: JavaAdapter) -> None:
    assert adapter.language_id == "java"


def test_file_extensions(adapter: JavaAdapter) -> None:
    assert adapter.file_extensions == [".java"]


def test_tree_sitter_name(adapter: JavaAdapter) -> None:
    assert adapter.tree_sitter_name == "java"


# ---------------------------------------------------------------------------
# extract_symbols: classes
# ---------------------------------------------------------------------------


def test_extract_class(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "Main.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.java")
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    assert any(s.name == "Main" for s in classes)


def test_class_visibility(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "Main.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.java")
    main_cls = next(s for s in symbols if s.name == "Main")
    assert main_cls.visibility == Visibility.PUBLIC


def test_inner_class(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    address = next(s for s in symbols if s.name == "Address")
    assert address.kind == SymbolKind.CLASS
    assert address.qualified_name == "User.Address"
    assert address.parent == "User"


def test_inner_class_members(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    get_street = next(s for s in symbols if s.name == "getStreet")
    assert get_street.qualified_name == "User.Address.getStreet"
    assert get_street.parent == "User.Address"


# ---------------------------------------------------------------------------
# extract_symbols: interfaces
# ---------------------------------------------------------------------------


def test_extract_interface(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.java")
    iface = next(s for s in symbols if s.name == "UserService")
    assert iface.kind == SymbolKind.INTERFACE
    assert iface.visibility == Visibility.PUBLIC


def test_interface_methods_public_by_default(
    engine: TreeSitterEngine, adapter: JavaAdapter
) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.java")
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD]
    assert len(methods) == 3
    for m in methods:
        assert m.visibility == Visibility.PUBLIC
        assert m.parent == "UserService"


# ---------------------------------------------------------------------------
# extract_symbols: enums
# ---------------------------------------------------------------------------


def test_extract_enum(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "enums" / "Status.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Status.java")
    status = next(s for s in symbols if s.name == "Status")
    assert status.kind == SymbolKind.ENUM


def test_enum_constants(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "enums" / "Status.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Status.java")
    consts = [s for s in symbols if s.kind == SymbolKind.CONSTANT]
    names = {s.name for s in consts}
    assert names == {"ACTIVE", "INACTIVE", "PENDING"}
    for c in consts:
        assert c.parent == "Status"
        assert c.qualified_name.startswith("Status.")


def test_enum_methods(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "enums" / "Status.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Status.java")
    get_label = next(s for s in symbols if s.name == "getLabel")
    assert get_label.kind == SymbolKind.METHOD
    assert get_label.parent == "Status"


# ---------------------------------------------------------------------------
# extract_symbols: methods
# ---------------------------------------------------------------------------


def test_method_visibility(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "User"}
    assert methods["getName"].visibility == Visibility.PUBLIC
    assert methods["getEmail"].visibility == Visibility.INTERNAL  # package-private
    assert methods["validate"].visibility == Visibility.PRIVATE


def test_method_signature(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    set_name = next(s for s in symbols if s.name == "setName")
    assert set_name.signature is not None
    assert "setName" in set_name.signature
    assert "String name" in set_name.signature


def test_constructor(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    ctor = next(
        s
        for s in symbols
        if s.kind == SymbolKind.METHOD and s.name == "User" and s.parent == "User"
    )
    assert ctor.qualified_name == "User.User"
    assert ctor.visibility == Visibility.PUBLIC
    assert ctor.signature is not None
    assert "String name" in ctor.signature


def test_static_method(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "StringUtils.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "StringUtils.java")
    capitalize = next(s for s in symbols if s.name == "capitalize")
    assert capitalize.kind == SymbolKind.METHOD
    assert capitalize.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# extract_symbols: fields and constants
# ---------------------------------------------------------------------------


def test_field_visibility(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    fields = {s.name: s for s in symbols if s.kind == SymbolKind.VARIABLE and s.parent == "User"}
    assert fields["name"].visibility == Visibility.PRIVATE
    assert fields["age"].visibility == Visibility.INTERNAL  # protected → INTERNAL


def test_static_final_is_constant(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.java")
    max_name = next(s for s in symbols if s.name == "MAX_NAME_LENGTH")
    assert max_name.kind == SymbolKind.CONSTANT
    assert max_name.visibility == Visibility.PUBLIC


def test_string_constant(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "StringUtils.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "StringUtils.java")
    empty = next(s for s in symbols if s.name == "EMPTY")
    assert empty.kind == SymbolKind.CONSTANT
    assert empty.visibility == Visibility.PUBLIC


def test_private_constant(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "StringUtils.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "StringUtils.java")
    pad = next(s for s in symbols if s.name == "DEFAULT_PAD")
    assert pad.kind == SymbolKind.CONSTANT
    assert pad.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_standard_imports(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "Main.java").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Main.java")
    modules = {i.module for i in imports}
    assert "com.example.models.User" in modules
    assert "com.example.services.UserService" in modules
    assert "java.util.List" in modules


def test_parse_static_import(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "StringUtils.java").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "StringUtils.java")
    static_imp = next(i for i in imports if "max" in i.symbols)
    assert static_imp.module == "java.lang.Math"
    assert static_imp.symbols == ["max"]


def test_parse_wildcard_import(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = b"package test;\nimport java.util.*;\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "test.java")
    assert len(imports) == 1
    assert imports[0].module == "java.util.*"


def test_imports_not_relative(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "Main.java").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Main.java")
    for imp in imports:
        assert imp.is_relative is False


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_internal_import(adapter: JavaAdapter) -> None:
    file_map = {
        "models/User.java": "/repo/models/User.java",
    }
    imp = ImportStatement(
        module="com.example.models.User",
        file_path="Main.java",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/models/User.java"


def test_resolve_external_returns_none(adapter: JavaAdapter) -> None:
    file_map = {"Main.java": "/repo/Main.java"}
    imp = ImportStatement(
        module="java.util.List",
        file_path="Main.java",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


def test_resolve_wildcard_returns_none(adapter: JavaAdapter) -> None:
    file_map = {"models/User.java": "/repo/models/User.java"}
    imp = ImportStatement(
        module="com.example.models.*",
        file_path="Main.java",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_public_symbol(adapter: JavaAdapter) -> None:
    s = Symbol(
        name="Main",
        qualified_name="Main",
        kind=SymbolKind.CLASS,
        file_path="Main.java",
        start_line=1,
        end_line=10,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_private_symbol(adapter: JavaAdapter) -> None:
    s = Symbol(
        name="validate",
        qualified_name="User.validate",
        kind=SymbolKind.METHOD,
        file_path="User.java",
        start_line=1,
        end_line=5,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_main_entry_point(adapter: JavaAdapter) -> None:
    files = [
        DiscoveredFile(
            path="Main.java",
            absolute_path=str(FIXTURES_DIR / "Main.java"),
            language="java",
        ),
        DiscoveredFile(
            path="models/User.java",
            absolute_path=str(FIXTURES_DIR / "models" / "User.java"),
            language="java",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "Main.java" in entry_points
    assert "models/User.java" not in entry_points


def test_no_entry_point_for_utility(adapter: JavaAdapter) -> None:
    files = [
        DiscoveredFile(
            path="utils/StringUtils.java",
            absolute_path=str(FIXTURES_DIR / "utils" / "StringUtils.java"),
            language="java",
        ),
    ]
    assert adapter.detect_entry_points(files) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Empty.java")
    imports = adapter.parse_imports(tree, source, "Empty.java")
    assert symbols == []
    assert imports == []


def test_package_private_method(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "StringUtils.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "StringUtils.java")
    pad_size = next(s for s in symbols if s.name == "padSize")
    assert pad_size.visibility == Visibility.INTERNAL


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.java"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_members_have_parent(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.java"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            if s.kind in (SymbolKind.METHOD, SymbolKind.VARIABLE, SymbolKind.CONSTANT):
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"


# ---------------------------------------------------------------------------
# extract_symbols: records
# ---------------------------------------------------------------------------


def test_extract_record_as_class(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "Record.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/Record.java")
    point = next(s for s in symbols if s.name == "Point")
    assert point.kind == SymbolKind.CLASS
    assert point.qualified_name == "Point"
    assert point.visibility == Visibility.PUBLIC


def test_record_method_member(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "Record.java").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/Record.java")
    distance = next(s for s in symbols if s.name == "distance")
    assert distance.kind == SymbolKind.METHOD
    assert distance.qualified_name == "Point.distance"
    assert distance.parent == "Point"


def test_record_nested_in_class(engine: TreeSitterEngine, adapter: JavaAdapter) -> None:
    source = b"""public class Outer {
    public record Inner(String value) {
        public String upper() { return value.toUpperCase(); }
    }
}"""
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Outer.java")
    inner = next(s for s in symbols if s.name == "Inner")
    assert inner.kind == SymbolKind.CLASS
    assert inner.qualified_name == "Outer.Inner"
    assert inner.parent == "Outer"
    upper = next(s for s in symbols if s.name == "upper")
    assert upper.parent == "Outer.Inner"


# ---------------------------------------------------------------------------
# Round-trip: parse → chunk → store → retrieve
# ---------------------------------------------------------------------------


def test_round_trip_parse_chunk_store_retrieve(
    engine: TreeSitterEngine,
    adapter: JavaAdapter,
    tmp_path: Path,
) -> None:
    from archex.index.store import IndexStore
    from archex.models import IndexConfig, ParsedFile
    from archex.pipeline.chunker import ASTChunker

    source = (FIXTURES_DIR / "models" / "User.java").read_bytes()
    tree = engine.parse_bytes(source, adapter.language_id)
    symbols = adapter.extract_symbols(tree, source, "models/User.java")
    assert len(symbols) > 0

    parsed = ParsedFile(path="models/User.java", language=adapter.language_id, symbols=symbols)
    chunker = ASTChunker(config=IndexConfig())
    chunks = chunker.chunk_files([parsed], {"models/User.java": source})
    assert len(chunks) > 0

    store = IndexStore(tmp_path / "test.db")
    try:
        store.insert_chunks(chunks)
        retrieved = store.get_chunks_for_file("models/User.java")
        assert len(retrieved) == len(chunks)
        for chunk in retrieved:
            assert chunk.content
            assert chunk.language == adapter.language_id
    finally:
        store.close()
