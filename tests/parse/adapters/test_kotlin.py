from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.kotlin import KotlinAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "kotlin_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> KotlinAdapter:
    return KotlinAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "kotlin")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: KotlinAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: KotlinAdapter) -> None:
    assert adapter.language_id == "kotlin"


def test_file_extensions(adapter: KotlinAdapter) -> None:
    assert adapter.file_extensions == [".kt", ".kts"]


def test_tree_sitter_name(adapter: KotlinAdapter) -> None:
    assert adapter.tree_sitter_name == "kotlin"


# ---------------------------------------------------------------------------
# extract_symbols: top-level functions
# ---------------------------------------------------------------------------


def test_extract_top_level_function(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.kt")
    funcs = [s for s in symbols if s.kind == SymbolKind.FUNCTION]
    names = {s.name for s in funcs}
    assert "main" in names
    assert "greet" in names


def test_top_level_function_visibility_default_public(
    engine: TreeSitterEngine, adapter: KotlinAdapter
) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.kt")
    greet = next(s for s in symbols if s.name == "greet")
    assert greet.visibility == Visibility.PUBLIC


def test_private_top_level_function(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.kt")
    helper = next(s for s in symbols if s.name == "internalHelper")
    assert helper.visibility == Visibility.PRIVATE


def test_top_level_function_has_no_parent(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.kt")
    greet = next(s for s in symbols if s.name == "greet")
    assert greet.parent is None


def test_top_level_function_signature(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Main.kt")
    greet = next(s for s in symbols if s.name == "greet")
    assert greet.signature is not None
    assert "greet" in greet.signature
    assert "name" in greet.signature


# ---------------------------------------------------------------------------
# extract_symbols: data class
# ---------------------------------------------------------------------------


def test_extract_data_class(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    assert any(s.name == "User" for s in classes)


def test_data_class_visibility(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    user = next(s for s in symbols if s.name == "User")
    assert user.visibility == Visibility.PUBLIC


def test_data_class_methods(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "User"}
    assert "displayName" in methods
    assert "validate" in methods


def test_method_visibility(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "User"}
    assert methods["displayName"].visibility == Visibility.PUBLIC
    assert methods["validate"].visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: sealed class
# ---------------------------------------------------------------------------


def test_extract_sealed_class(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    assert any(s.name == "UserResult" for s in classes)


def test_sealed_class_nested_data_classes(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    nested = {s.name: s for s in symbols if s.parent == "UserResult"}
    assert "Success" in nested
    assert "Error" in nested
    assert nested["Success"].qualified_name == "UserResult.Success"


# ---------------------------------------------------------------------------
# extract_symbols: companion object
# ---------------------------------------------------------------------------


def test_companion_object_extracted(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    companion = next(s for s in symbols if s.name == "Companion")
    assert companion.kind == SymbolKind.CLASS
    assert companion.qualified_name == "User.Companion"
    assert companion.parent == "User"


def test_companion_object_property(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    max_len = next(s for s in symbols if s.name == "MAX_NAME_LENGTH")
    assert max_len.kind == SymbolKind.VARIABLE
    assert max_len.parent == "User.Companion"


def test_companion_object_method(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    from_map = next(s for s in symbols if s.name == "fromMap")
    assert from_map.kind == SymbolKind.METHOD
    assert from_map.parent == "User.Companion"
    assert from_map.qualified_name == "User.Companion.fromMap"


# ---------------------------------------------------------------------------
# extract_symbols: interface
# ---------------------------------------------------------------------------


def test_extract_interface(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    iface = next(s for s in symbols if s.name == "UserService")
    assert iface.kind == SymbolKind.INTERFACE
    assert iface.visibility == Visibility.PUBLIC


def test_interface_methods(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "UserService"]
    names = {s.name for s in methods}
    # findById, findAll, save are interface members; exists is an extension function
    assert {"findById", "findAll", "save", "exists"} == names


def test_interface_methods_default_public(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "UserService"]
    for m in methods:
        assert m.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# extract_symbols: class implementing interface
# ---------------------------------------------------------------------------


def test_extract_impl_class(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    impl = next(s for s in symbols if s.name == "UserServiceImpl")
    assert impl.kind == SymbolKind.CLASS


def test_impl_class_methods(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    methods = {
        s.name: s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "UserServiceImpl"
    }
    assert "findById" in methods
    assert "findAll" in methods
    assert "save" in methods
    assert "count" in methods


def test_internal_method_visibility(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    count = next(s for s in symbols if s.name == "count" and s.parent == "UserServiceImpl")
    assert count.visibility == Visibility.INTERNAL


def test_private_property(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    users_prop = next(s for s in symbols if s.name == "users" and s.parent == "UserServiceImpl")
    assert users_prop.kind == SymbolKind.VARIABLE
    assert users_prop.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: extension functions
# ---------------------------------------------------------------------------


def test_extension_function_extracted(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.kt")
    ext = next(s for s in symbols if s.name == "exists")
    assert ext.kind == SymbolKind.METHOD
    assert ext.parent == "UserService"
    assert ext.qualified_name == "UserService.exists"


def test_extension_functions_in_utils(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "Extensions.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "utils/Extensions.kt")
    ext_methods = [s for s in symbols if s.kind == SymbolKind.METHOD]
    names = {s.name for s in ext_methods}
    assert "toTitleCase" in names
    assert "isEmail" in names
    assert "clamp" in names


def test_extension_function_parent_is_receiver(
    engine: TreeSitterEngine, adapter: KotlinAdapter
) -> None:
    source = (FIXTURES_DIR / "utils" / "Extensions.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "utils/Extensions.kt")
    to_title = next(s for s in symbols if s.name == "toTitleCase")
    assert to_title.parent == "String"
    assert to_title.qualified_name == "String.toTitleCase"


# ---------------------------------------------------------------------------
# extract_symbols: typealias
# ---------------------------------------------------------------------------


def test_extract_typealias(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "utils" / "Extensions.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "utils/Extensions.kt")
    type_syms = [s for s in symbols if s.kind == SymbolKind.TYPE]
    names = {s.name for s in type_syms}
    assert "StringMap" in names
    assert "UserList" in names


# ---------------------------------------------------------------------------
# extract_symbols: object declaration
# ---------------------------------------------------------------------------


def test_extract_object_declaration(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "config" / "AppConfig.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "config/AppConfig.kt")
    obj = next(s for s in symbols if s.name == "AppConfig")
    assert obj.kind == SymbolKind.CLASS
    assert obj.visibility == Visibility.PUBLIC


def test_object_properties(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "config" / "AppConfig.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "config/AppConfig.kt")
    props = {
        s.name: s for s in symbols if s.kind == SymbolKind.VARIABLE and s.parent == "AppConfig"
    }
    assert "appName" in props
    assert "version" in props
    assert "secret" in props
    assert props["secret"].visibility == Visibility.PRIVATE


def test_object_method(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "config" / "AppConfig.kt").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "config/AppConfig.kt")
    build_url = next(s for s in symbols if s.name == "buildUrl")
    assert build_url.kind == SymbolKind.METHOD
    assert build_url.parent == "AppConfig"


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_standard_imports(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Main.kt")
    modules = {i.module for i in imports}
    assert "com.example.models.User" in modules
    assert "com.example.services.UserService" in modules
    assert "kotlin.collections.List" in modules


def test_parse_wildcard_import(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = b"package test\nimport com.example.*\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "test.kt")
    assert len(imports) == 1
    assert imports[0].module == "com.example.*"


def test_imports_not_relative(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = (FIXTURES_DIR / "Main.kt").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Main.kt")
    for imp in imports:
        assert imp.is_relative is False


def test_parse_import_with_alias(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = b"package test\nimport com.example.Foo as Bar\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "test.kt")
    assert len(imports) == 1
    # Module should be the full path (not the alias)
    assert imports[0].module == "com.example.Foo"


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_internal_import(adapter: KotlinAdapter) -> None:
    file_map = {
        "models/User.kt": "/repo/models/User.kt",
    }
    imp = ImportStatement(
        module="com.example.models.User",
        file_path="Main.kt",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/models/User.kt"


def test_resolve_external_returns_none(adapter: KotlinAdapter) -> None:
    file_map = {"Main.kt": "/repo/Main.kt"}
    imp = ImportStatement(
        module="kotlin.collections.List",
        file_path="Main.kt",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


def test_resolve_wildcard_returns_none(adapter: KotlinAdapter) -> None:
    file_map = {"models/User.kt": "/repo/models/User.kt"}
    imp = ImportStatement(
        module="com.example.models.*",
        file_path="Main.kt",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


def test_resolve_java_file_from_kotlin(adapter: KotlinAdapter) -> None:
    file_map = {
        "models/User.java": "/repo/models/User.java",
    }
    imp = ImportStatement(
        module="com.example.models.User",
        file_path="Main.kt",
        line=3,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/models/User.java"


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_public_symbol(adapter: KotlinAdapter) -> None:
    s = Symbol(
        name="AppConfig",
        qualified_name="AppConfig",
        kind=SymbolKind.CLASS,
        file_path="config/AppConfig.kt",
        start_line=1,
        end_line=10,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_private_symbol(adapter: KotlinAdapter) -> None:
    s = Symbol(
        name="secret",
        qualified_name="AppConfig.secret",
        kind=SymbolKind.VARIABLE,
        file_path="config/AppConfig.kt",
        start_line=5,
        end_line=5,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_main_entry_point(adapter: KotlinAdapter) -> None:
    files = [
        DiscoveredFile(
            path="Main.kt",
            absolute_path=str(FIXTURES_DIR / "Main.kt"),
            language="kotlin",
        ),
        DiscoveredFile(
            path="models/User.kt",
            absolute_path=str(FIXTURES_DIR / "models" / "User.kt"),
            language="kotlin",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "Main.kt" in entry_points
    assert "models/User.kt" not in entry_points


def test_no_entry_point_for_config(adapter: KotlinAdapter) -> None:
    files = [
        DiscoveredFile(
            path="config/AppConfig.kt",
            absolute_path=str(FIXTURES_DIR / "config" / "AppConfig.kt"),
            language="kotlin",
        ),
    ]
    assert adapter.detect_entry_points(files) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Empty.kt")
    imports = adapter.parse_imports(tree, source, "Empty.kt")
    assert symbols == []
    assert imports == []


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.kt"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_members_have_parent(engine: TreeSitterEngine, adapter: KotlinAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.kt"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            if s.kind == SymbolKind.METHOD:
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"
            if s.kind == SymbolKind.VARIABLE:
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"


# ---------------------------------------------------------------------------
# Round-trip: parse → chunk → store → retrieve
# ---------------------------------------------------------------------------


def test_round_trip_parse_chunk_store_retrieve(
    engine: TreeSitterEngine,
    adapter: KotlinAdapter,
    tmp_path: Path,
) -> None:
    from archex.index.store import IndexStore
    from archex.models import IndexConfig, ParsedFile
    from archex.pipeline.chunker import ASTChunker

    source = (FIXTURES_DIR / "models" / "User.kt").read_bytes()
    tree = engine.parse_bytes(source, adapter.language_id)
    symbols = adapter.extract_symbols(tree, source, "models/User.kt")
    assert len(symbols) > 0

    parsed = ParsedFile(path="models/User.kt", language=adapter.language_id, symbols=symbols)
    chunker = ASTChunker(config=IndexConfig())
    chunks = chunker.chunk_files([parsed], {"models/User.kt": source})
    assert len(chunks) > 0

    store = IndexStore(tmp_path / "test.db")
    try:
        store.insert_chunks(chunks)
        retrieved = store.get_chunks_for_file("models/User.kt")
        assert len(retrieved) == len(chunks)
        for chunk in retrieved:
            assert chunk.content
            assert chunk.language == adapter.language_id
    finally:
        store.close()
