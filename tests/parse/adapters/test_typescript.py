from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.typescript import TypeScriptAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "typescript_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> TypeScriptAdapter:
    return TypeScriptAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "typescript")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: TypeScriptAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: TypeScriptAdapter) -> None:
    assert adapter.language_id == "typescript"


def test_file_extensions(adapter: TypeScriptAdapter) -> None:
    exts = adapter.file_extensions
    assert ".ts" in exts
    assert ".tsx" in exts
    assert ".js" in exts
    assert ".jsx" in exts


def test_tree_sitter_name(adapter: TypeScriptAdapter) -> None:
    assert adapter.tree_sitter_name == "typescript"


# ---------------------------------------------------------------------------
# extract_symbols: types.ts — enum, interface, type alias
# ---------------------------------------------------------------------------


def test_types_ts_enum(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "types.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/types.ts")
    enums = [s for s in symbols if s.kind == SymbolKind.ENUM]
    assert len(enums) == 1
    assert enums[0].name == "Role"
    assert enums[0].visibility == Visibility.PUBLIC


def test_types_ts_interface(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "types.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/types.ts")
    interfaces = [s for s in symbols if s.kind == SymbolKind.INTERFACE]
    assert len(interfaces) == 1
    assert interfaces[0].name == "User"
    assert interfaces[0].visibility == Visibility.PUBLIC


def test_types_ts_type_alias(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "types.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/types.ts")
    type_aliases = [s for s in symbols if s.kind == SymbolKind.TYPE]
    assert len(type_aliases) == 1
    assert type_aliases[0].name == "AuthResult"
    assert type_aliases[0].visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# extract_symbols: utils.ts — exported functions, constant
# ---------------------------------------------------------------------------


def test_utils_ts_functions(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/utils.ts")
    funcs = {s.name for s in symbols if s.kind == SymbolKind.FUNCTION}
    assert "hashPassword" in funcs
    assert "validateEmail" in funcs
    for s in symbols:
        if s.name in ("hashPassword", "validateEmail"):
            assert s.visibility == Visibility.PUBLIC


def test_utils_ts_unexported_constant(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/utils.ts")
    consts = [s for s in symbols if s.kind in (SymbolKind.CONSTANT, SymbolKind.VARIABLE)]
    names = {s.name for s in consts}
    assert "EMAIL_RE" in names
    email_re = next(s for s in consts if s.name == "EMAIL_RE")
    assert email_re.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: auth.ts — functions
# ---------------------------------------------------------------------------


def test_auth_ts_functions(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "handlers" / "auth.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/handlers/auth.ts")
    funcs = {s.name for s in symbols if s.kind == SymbolKind.FUNCTION}
    assert "handleLogin" in funcs
    assert "handleLogout" in funcs
    for s in symbols:
        if s.name in ("handleLogin", "handleLogout"):
            assert s.visibility == Visibility.PUBLIC


def test_auth_ts_unexported_constant(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "handlers" / "auth.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/handlers/auth.ts")
    non_exported = [s for s in symbols if s.visibility == Visibility.PRIVATE]
    names = {s.name for s in non_exported}
    assert "sessions" in names


# ---------------------------------------------------------------------------
# parse_imports: utils.ts — ES module import from external package
# ---------------------------------------------------------------------------


def test_utils_ts_imports(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.ts").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/utils.ts")
    modules = [i.module for i in imports]
    assert "crypto" in modules
    crypto_imp = next(i for i in imports if i.module == "crypto")
    assert "createHash" in crypto_imp.symbols
    assert crypto_imp.is_relative is False


# ---------------------------------------------------------------------------
# parse_imports: auth.ts — type-only imports and named imports
# ---------------------------------------------------------------------------


def test_auth_ts_imports_type_only(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "handlers" / "auth.ts").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/handlers/auth.ts")
    # Should have at least 2 imports: types.js and utils.js
    assert len(imports) >= 2
    modules = [i.module for i in imports]
    assert any("types" in m for m in modules)
    assert any("utils" in m for m in modules)

    types_imp = next(i for i in imports if "types" in i.module)
    assert types_imp.is_relative is True
    # type-only imports may have AuthResult and/or User
    assert len(types_imp.symbols) >= 1


def test_auth_ts_named_imports(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "handlers" / "auth.ts").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/handlers/auth.ts")
    utils_imp = next(i for i in imports if "utils" in i.module)
    assert "hashPassword" in utils_imp.symbols
    assert "validateEmail" in utils_imp.symbols
    assert utils_imp.is_relative is True


# ---------------------------------------------------------------------------
# parse_imports: index.ts — re-export statements
# ---------------------------------------------------------------------------


def test_index_ts_reexports(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "index.ts").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/index.ts")
    assert len(imports) >= 3
    modules = [i.module for i in imports]
    assert any("types" in m for m in modules)
    assert any("utils" in m for m in modules)
    assert any("auth" in m for m in modules)
    for imp in imports:
        assert imp.is_relative is True


# ---------------------------------------------------------------------------
# parse_imports: inline source tests
# ---------------------------------------------------------------------------


def test_default_import(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = b"import React from 'react';\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.tsx")
    assert len(imports) == 1
    assert imports[0].module == "react"
    assert "React" in imports[0].symbols


def test_namespace_import(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = b"import * as fs from 'fs';\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.ts")
    assert len(imports) == 1
    assert imports[0].module == "fs"
    assert imports[0].alias is not None or "*" in imports[0].symbols


def test_side_effect_import(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = b"import './polyfill';\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.ts")
    assert len(imports) == 1
    assert imports[0].module == "./polyfill"
    assert imports[0].is_relative is True


def test_commonjs_require(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = b"const path = require('path');\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.js")
    assert len(imports) == 1
    assert imports[0].module == "path"
    assert imports[0].is_relative is False


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_relative_ts_file(adapter: TypeScriptAdapter) -> None:
    file_map = {
        "src/types.ts": "/repo/src/types.ts",
        "src/utils.ts": "/repo/src/utils.ts",
    }
    imp = ImportStatement(
        module="../types.js",
        symbols=["User"],
        file_path="src/handlers/auth.ts",
        line=1,
        is_relative=True,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/src/types.ts"


def test_resolve_relative_without_extension(adapter: TypeScriptAdapter) -> None:
    file_map = {"src/utils.ts": "/repo/src/utils.ts"}
    imp = ImportStatement(
        module="./utils",
        symbols=["hashPassword"],
        file_path="src/index.ts",
        line=1,
        is_relative=True,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/src/utils.ts"


def test_resolve_external_returns_none(adapter: TypeScriptAdapter) -> None:
    file_map = {"src/utils.ts": "/repo/src/utils.ts"}
    imp = ImportStatement(
        module="react",
        symbols=["useState"],
        file_path="src/app.tsx",
        line=1,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


def test_resolve_index_ts(adapter: TypeScriptAdapter) -> None:
    file_map = {"src/handlers/index.ts": "/repo/src/handlers/index.ts"}
    imp = ImportStatement(
        module="./handlers",
        symbols=[],
        file_path="src/index.ts",
        line=1,
        is_relative=True,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/src/handlers/index.ts"


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_visibility_public(adapter: TypeScriptAdapter) -> None:
    s = Symbol(
        name="MyClass",
        qualified_name="MyClass",
        kind=SymbolKind.CLASS,
        file_path="x.ts",
        start_line=1,
        end_line=10,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_visibility_private(adapter: TypeScriptAdapter) -> None:
    s = Symbol(
        name="internalHelper",
        qualified_name="internalHelper",
        kind=SymbolKind.FUNCTION,
        file_path="x.ts",
        start_line=1,
        end_line=5,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE


def test_exported_symbol_is_public(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = b"export function greet(): void {}\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "x.ts")
    assert len(symbols) == 1
    assert symbols[0].visibility == Visibility.PUBLIC


def test_unexported_symbol_is_private(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = b"function helper(): void {}\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "x.ts")
    assert len(symbols) == 1
    assert symbols[0].visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_index_ts(adapter: TypeScriptAdapter, tmp_path: Path) -> None:
    index_file = tmp_path / "index.ts"
    index_file.write_text("export const x = 1;\n")
    files = [
        DiscoveredFile(
            path="src/index.ts",
            absolute_path=str(index_file),
            language="typescript",
        )
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "src/index.ts" in entry_points


def test_detect_main_ts(adapter: TypeScriptAdapter, tmp_path: Path) -> None:
    main_file = tmp_path / "main.ts"
    main_file.write_text("console.log('hello');\n")
    files = [
        DiscoveredFile(
            path="src/main.ts",
            absolute_path=str(main_file),
            language="typescript",
        )
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "src/main.ts" in entry_points


def test_detect_export_default(adapter: TypeScriptAdapter, tmp_path: Path) -> None:
    src_file = tmp_path / "app.ts"
    src_file.write_text("export default function App() {}\n")
    files = [
        DiscoveredFile(
            path="src/app.ts",
            absolute_path=str(src_file),
            language="typescript",
        )
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "src/app.ts" in entry_points


def test_non_entry_point_excluded(adapter: TypeScriptAdapter, tmp_path: Path) -> None:
    src_file = tmp_path / "utils.ts"
    src_file.write_text("export function helper() {}\n")
    files = [
        DiscoveredFile(
            path="src/utils.ts",
            absolute_path=str(src_file),
            language="typescript",
        )
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "src/utils.ts" not in entry_points


# ---------------------------------------------------------------------------
# Integration: typescript_simple fixture — index.ts is an entry point
# ---------------------------------------------------------------------------


def test_typescript_simple_entry_points(adapter: TypeScriptAdapter) -> None:
    fixture_dir = FIXTURES_DIR
    files = [
        DiscoveredFile(
            path="src/types.ts",
            absolute_path=str(fixture_dir / "src" / "types.ts"),
            language="typescript",
        ),
        DiscoveredFile(
            path="src/utils.ts",
            absolute_path=str(fixture_dir / "src" / "utils.ts"),
            language="typescript",
        ),
        DiscoveredFile(
            path="src/handlers/auth.ts",
            absolute_path=str(fixture_dir / "src" / "handlers" / "auth.ts"),
            language="typescript",
        ),
        DiscoveredFile(
            path="src/index.ts",
            absolute_path=str(fixture_dir / "src" / "index.ts"),
            language="typescript",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "src/index.ts" in entry_points


def test_all_types_ts_symbols_found(engine: TreeSitterEngine, adapter: TypeScriptAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "types.ts").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/types.ts")
    names = {s.name for s in symbols}
    assert "Role" in names
    assert "User" in names
    assert "AuthResult" in names
    assert len(symbols) == 3
