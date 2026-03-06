from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tree_sitter_swift  # noqa: F401  # pyright: ignore[reportUnusedImport,reportMissingImports]
except ImportError:
    try:
        from tree_sitter_language_pack import get_language as _get_language

        _get_language("swift")
    except (ImportError, Exception):
        pytest.skip("tree-sitter-swift not available", allow_module_level=True)

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.swift import SwiftAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "swift_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> SwiftAdapter:
    return SwiftAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "swift")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: SwiftAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: SwiftAdapter) -> None:
    assert adapter.language_id == "swift"


def test_file_extensions(adapter: SwiftAdapter) -> None:
    assert adapter.file_extensions == [".swift"]


def test_tree_sitter_name(adapter: SwiftAdapter) -> None:
    assert adapter.tree_sitter_name == "swift"


# ---------------------------------------------------------------------------
# extract_symbols: struct
# ---------------------------------------------------------------------------


def test_extract_struct_as_type(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    user = next(s for s in symbols if s.name == "User")
    assert user.kind == SymbolKind.TYPE
    assert user.visibility == Visibility.PUBLIC


def test_main_struct_as_class(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "main.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "main.swift")
    app_entry = next(s for s in symbols if s.name == "AppEntry")
    assert app_entry.kind == SymbolKind.CLASS


def test_struct_members(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    member_names = {s.name for s in symbols if s.parent == "User"}
    assert "name" in member_names
    assert "age" in member_names
    assert "displayName" in member_names
    assert "validate" in member_names


def test_struct_property_visibility(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    members = {s.name: s for s in symbols if s.parent == "User"}
    assert members["name"].visibility == Visibility.PUBLIC
    assert members["_id"].visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: class
# ---------------------------------------------------------------------------


def test_extract_class(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    svc = next(s for s in symbols if s.name == "UserService")
    assert svc.kind == SymbolKind.CLASS
    assert svc.visibility == Visibility.PUBLIC


def test_class_open_visibility(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"open class Foo {}"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Foo.swift")
    foo = next(s for s in symbols if s.name == "Foo")
    assert foo.visibility == Visibility.PUBLIC


def test_class_internal_default(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"class Bar {}"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Bar.swift")
    bar = next(s for s in symbols if s.name == "Bar")
    assert bar.visibility == Visibility.INTERNAL


def test_class_methods(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    methods = {
        s.name: s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "UserService"
    }
    assert "findById" in methods
    assert "findAll" in methods
    assert "save" in methods
    assert "addUser" in methods


def test_method_visibility(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    methods = {
        s.name: s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "UserService"
    }
    assert methods["findById"].visibility == Visibility.PUBLIC
    assert methods["internalCleanup"].visibility == Visibility.PRIVATE


def test_fileprivate_visibility(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"class X { fileprivate func hidden() {} }"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "X.swift")
    hidden = next(s for s in symbols if s.name == "hidden")
    assert hidden.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: protocol
# ---------------------------------------------------------------------------


def test_extract_protocol(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    proto = next(s for s in symbols if s.name == "UserRepository")
    assert proto.kind == SymbolKind.INTERFACE
    assert proto.visibility == Visibility.PUBLIC


def test_protocol_methods_public(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    proto_methods = [s for s in symbols if s.parent == "UserRepository"]
    assert len(proto_methods) == 3
    for m in proto_methods:
        assert m.visibility == Visibility.PUBLIC


def test_protocol_property(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    proto_id = next(s for s in symbols if s.name == "id" and s.parent == "Identifiable")
    assert proto_id.kind == SymbolKind.VARIABLE
    assert proto_id.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# extract_symbols: enum
# ---------------------------------------------------------------------------


def test_extract_enum(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    enum = next(s for s in symbols if s.name == "UserStatus")
    assert enum.kind == SymbolKind.ENUM
    assert enum.visibility == Visibility.PUBLIC


def test_enum_cases(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    cases = [s for s in symbols if s.kind == SymbolKind.CONSTANT and s.parent == "UserStatus"]
    names = {s.name for s in cases}
    assert names == {"active", "inactive", "suspended"}
    for c in cases:
        assert c.qualified_name == f"UserStatus.{c.name}"


def test_enum_method(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    label = next(s for s in symbols if s.name == "label" and s.parent == "UserStatus")
    assert label.kind == SymbolKind.METHOD


# ---------------------------------------------------------------------------
# extract_symbols: extension
# ---------------------------------------------------------------------------


def test_extension_methods_get_parent(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    user_count = next(s for s in symbols if s.name == "userCount")
    assert user_count.parent == "UserService"
    assert user_count.qualified_name == "UserService.userCount"


def test_extension_no_own_symbol(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    # Extension itself must not emit a separate symbol
    source = b"extension String { func x() {} }"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Ext.swift")
    # Only "x" should appear, not "String" as a duplicate type symbol
    names = [s.name for s in symbols]
    assert "String" not in names
    assert "x" in names


def test_extension_protocol_conformance_methods(
    engine: TreeSitterEngine, adapter: SwiftAdapter
) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    # User.id and User.describe come from extension User: Identifiable
    user_id = next(s for s in symbols if s.name == "id" and s.parent == "User")
    assert user_id.qualified_name == "User.id"


# ---------------------------------------------------------------------------
# extract_symbols: typealias
# ---------------------------------------------------------------------------


def test_extract_typealias(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "User.swift")
    uid = next(s for s in symbols if s.name == "UserID")
    assert uid.kind == SymbolKind.TYPE
    assert uid.visibility == Visibility.PUBLIC


def test_extract_function_typealias(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Utils" / "Extensions.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Extensions.swift")
    completion = next(s for s in symbols if s.name == "Completion")
    assert completion.kind == SymbolKind.TYPE


# ---------------------------------------------------------------------------
# extract_symbols: subscript
# ---------------------------------------------------------------------------


def test_extract_subscript(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Views" / "ContentView.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "ContentView.swift")
    sub = next(s for s in symbols if s.name == "subscript")
    assert sub.kind == SymbolKind.METHOD
    assert sub.parent == "UserListView"
    assert sub.qualified_name == "UserListView.subscript"


# ---------------------------------------------------------------------------
# extract_symbols: top-level function
# ---------------------------------------------------------------------------


def test_top_level_function(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "main.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "main.swift")
    bootstrap = next(s for s in symbols if s.name == "bootstrap")
    assert bootstrap.kind == SymbolKind.FUNCTION
    assert bootstrap.parent is None
    assert bootstrap.qualified_name == "bootstrap"


# ---------------------------------------------------------------------------
# extract_symbols: method signatures
# ---------------------------------------------------------------------------


def test_method_has_signature(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.swift").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "UserService.swift")
    find_by_id = next(s for s in symbols if s.name == "findById" and s.parent == "UserService")
    assert find_by_id.signature is not None
    assert "findById" in find_by_id.signature


def test_function_signature_contains_params(
    engine: TreeSitterEngine, adapter: SwiftAdapter
) -> None:
    source = b"class Foo { func greet(name: String, age: Int) {} }"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Foo.swift")
    greet = next(s for s in symbols if s.name == "greet")
    assert greet.signature is not None
    assert "name" in greet.signature


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_simple_imports(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "main.swift").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "main.swift")
    modules = {i.module for i in imports}
    assert "Foundation" in modules
    assert "SwiftUI" in modules


def test_parse_multiple_imports(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "Views" / "ContentView.swift").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "ContentView.swift")
    assert len(imports) == 2
    modules = {i.module for i in imports}
    assert "SwiftUI" in modules
    assert "Foundation" in modules


def test_imports_not_relative(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = (FIXTURES_DIR / "main.swift").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "main.swift")
    for imp in imports:
        assert imp.is_relative is False


def test_imports_have_correct_line(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"import Foundation\nimport UIKit\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "test.swift")
    assert len(imports) == 2
    foundation = next(i for i in imports if i.module == "Foundation")
    uikit = next(i for i in imports if i.module == "UIKit")
    assert foundation.line == 1
    assert uikit.line == 2


def test_testable_import(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"@testable import MyModule\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "test.swift")
    assert len(imports) == 1
    assert imports[0].module == "MyModule"


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_internal_module(adapter: SwiftAdapter) -> None:
    file_map = {
        "Services/UserService.swift": "/repo/Services/UserService.swift",
        "Models/User.swift": "/repo/Models/User.swift",
    }
    imp = ImportStatement(
        module="Services",
        file_path="main.swift",
        line=1,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/Services/UserService.swift"


def test_resolve_external_framework_returns_none(adapter: SwiftAdapter) -> None:
    file_map = {"main.swift": "/repo/main.swift"}
    for framework in ("Foundation", "UIKit", "SwiftUI", "Combine", "XCTest"):
        imp = ImportStatement(
            module=framework,
            file_path="main.swift",
            line=1,
            is_relative=False,
        )
        assert adapter.resolve_import(imp, file_map) is None


def test_resolve_unknown_module_returns_none(adapter: SwiftAdapter) -> None:
    file_map = {"Models/User.swift": "/repo/Models/User.swift"}
    imp = ImportStatement(
        module="NonExistentModule",
        file_path="main.swift",
        line=1,
        is_relative=False,
    )
    assert adapter.resolve_import(imp, file_map) is None


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_public_symbol(adapter: SwiftAdapter) -> None:
    s = Symbol(
        name="UserService",
        qualified_name="UserService",
        kind=SymbolKind.CLASS,
        file_path="UserService.swift",
        start_line=1,
        end_line=30,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_private_symbol(adapter: SwiftAdapter) -> None:
    s = Symbol(
        name="validate",
        qualified_name="User.validate",
        kind=SymbolKind.METHOD,
        file_path="User.swift",
        start_line=20,
        end_line=22,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE


def test_classify_internal_symbol(adapter: SwiftAdapter) -> None:
    s = Symbol(
        name="helper",
        qualified_name="helper",
        kind=SymbolKind.FUNCTION,
        file_path="main.swift",
        start_line=5,
        end_line=7,
        visibility=Visibility.INTERNAL,
    )
    assert adapter.classify_visibility(s) == Visibility.INTERNAL


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_main_attribute_entry_point(adapter: SwiftAdapter) -> None:
    files = [
        DiscoveredFile(
            path="main.swift",
            absolute_path=str(FIXTURES_DIR / "main.swift"),
            language="swift",
        ),
        DiscoveredFile(
            path="Models/User.swift",
            absolute_path=str(FIXTURES_DIR / "Models" / "User.swift"),
            language="swift",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "main.swift" in entry_points
    assert "Models/User.swift" not in entry_points


def test_no_entry_point_for_plain_file(adapter: SwiftAdapter) -> None:
    files = [
        DiscoveredFile(
            path="Utils/Extensions.swift",
            absolute_path=str(FIXTURES_DIR / "Utils" / "Extensions.swift"),
            language="swift",
        ),
    ]
    assert adapter.detect_entry_points(files) == []


def test_detect_xctest_entry_point(adapter: SwiftAdapter, tmp_path: Path) -> None:
    test_file = tmp_path / "MyTests.swift"
    test_file.write_text(
        "import XCTest\nclass MyTests: XCTestCase {\n    func testSomething() {}\n}\n"
    )
    files = [
        DiscoveredFile(
            path="MyTests.swift",
            absolute_path=str(test_file),
            language="swift",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "MyTests.swift" in entry_points


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Empty.swift")
    imports = adapter.parse_imports(tree, source, "Empty.swift")
    assert symbols == []
    assert imports == []


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.swift"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_members_have_parent(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.swift"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            if s.kind in (SymbolKind.METHOD, SymbolKind.VARIABLE, SymbolKind.CONSTANT):
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"


def test_actor_extracted_as_class(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"actor DataActor { var count: Int = 0\nfunc update() {} }"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "DataActor.swift")
    actor = next(s for s in symbols if s.name == "DataActor")
    assert actor.kind == SymbolKind.CLASS


def test_internal_visibility_default(engine: TreeSitterEngine, adapter: SwiftAdapter) -> None:
    source = b"struct Foo { var x: Int = 0\nfunc bar() {} }"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Foo.swift")
    foo = next(s for s in symbols if s.name == "Foo")
    assert foo.visibility == Visibility.INTERNAL


# ---------------------------------------------------------------------------
# Round-trip: parse → chunk → store → retrieve
# ---------------------------------------------------------------------------


def test_round_trip_parse_chunk_store_retrieve(
    engine: TreeSitterEngine,
    adapter: SwiftAdapter,
    tmp_path: Path,
) -> None:
    from archex.index.chunker import ASTChunker
    from archex.index.store import IndexStore
    from archex.models import IndexConfig, ParsedFile

    source = (FIXTURES_DIR / "Models" / "User.swift").read_bytes()
    tree = engine.parse_bytes(source, adapter.language_id)
    symbols = adapter.extract_symbols(tree, source, "Models/User.swift")
    assert len(symbols) > 0

    parsed = ParsedFile(path="Models/User.swift", language=adapter.language_id, symbols=symbols)
    chunker = ASTChunker(config=IndexConfig())
    chunks = chunker.chunk_files([parsed], {"Models/User.swift": source})
    assert len(chunks) > 0

    store = IndexStore(tmp_path / "test.db")
    try:
        store.insert_chunks(chunks)
        retrieved = store.get_chunks_for_file("Models/User.swift")
        assert len(retrieved) == len(chunks)
        for chunk in retrieved:
            assert chunk.content
            assert chunk.language == adapter.language_id
    finally:
        store.close()
