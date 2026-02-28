from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.rust import RustAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "rust_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> RustAdapter:
    return RustAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "rust")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: RustAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: RustAdapter) -> None:
    assert adapter.language_id == "rust"


def test_file_extensions(adapter: RustAdapter) -> None:
    assert adapter.file_extensions == [".rs"]


def test_tree_sitter_name(adapter: RustAdapter) -> None:
    assert adapter.tree_sitter_name == "rust"


# ---------------------------------------------------------------------------
# extract_symbols: main.rs — functions
# ---------------------------------------------------------------------------


def test_main_rs_functions(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "main.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/main.rs")
    funcs = [s for s in symbols if s.kind == SymbolKind.FUNCTION]
    names = [f.name for f in funcs]
    assert "main" in names
    assert "setup_logging" in names


def test_main_rs_function_visibility(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "main.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/main.rs")
    main_fn = next(s for s in symbols if s.name == "main")
    assert main_fn.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: lib.rs — traits
# ---------------------------------------------------------------------------


def test_lib_rs_traits(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "lib.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/lib.rs")
    traits = [s for s in symbols if s.kind == SymbolKind.INTERFACE]
    names = [t.name for t in traits]
    assert "Processor" in names
    assert "Serializable" in names


def test_lib_rs_trait_methods(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "lib.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/lib.rs")
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "Processor"]
    names = [m.name for m in methods]
    assert "process" in names
    assert "validate" in names


def test_lib_rs_pub_function(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "lib.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/lib.rs")
    init_fn = next(s for s in symbols if s.name == "initialize")
    assert init_fn.kind == SymbolKind.FUNCTION
    assert init_fn.visibility == Visibility.PUBLIC


def test_lib_rs_private_function(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "lib.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/lib.rs")
    helper_fn = next(s for s in symbols if s.name == "internal_helper")
    assert helper_fn.visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: models.rs — structs, enums, impl blocks
# ---------------------------------------------------------------------------


def test_models_rs_structs(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    structs = [s for s in symbols if s.kind == SymbolKind.TYPE]
    names = [s.name for s in structs]
    assert "User" in names
    assert "Config" in names
    assert "Pagination" in names


def test_models_rs_pub_struct(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    user = next(s for s in symbols if s.name == "User" and s.kind == SymbolKind.TYPE)
    assert user.visibility == Visibility.PUBLIC


def test_models_rs_pub_crate_struct(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    config = next(s for s in symbols if s.name == "Config" and s.kind == SymbolKind.TYPE)
    assert config.visibility == Visibility.INTERNAL


def test_models_rs_enum(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    enums = [s for s in symbols if s.kind == SymbolKind.ENUM]
    assert len(enums) == 1
    assert enums[0].name == "Role"
    assert enums[0].visibility == Visibility.PUBLIC


def test_models_rs_impl_methods(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    user_methods = [s for s in symbols if s.kind == SymbolKind.METHOD and s.parent == "User"]
    names = [m.name for m in user_methods]
    assert "new" in names
    assert "display_name" in names
    assert "validate_email" in names


def test_models_rs_impl_method_parent(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    new_method = next(s for s in symbols if s.name == "new" and s.parent == "User")
    assert new_method.qualified_name == "User.new"
    assert new_method.visibility == Visibility.PUBLIC


def test_models_rs_private_method(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    validate = next(s for s in symbols if s.name == "validate_email")
    assert validate.visibility == Visibility.PRIVATE
    assert validate.parent == "User"


def test_models_rs_trait_impl_methods(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/models.rs")
    # impl fmt::Display for User — type node is "User"
    fmt_method = next((s for s in symbols if s.name == "fmt" and s.kind == SymbolKind.METHOD), None)
    assert fmt_method is not None
    assert fmt_method.parent == "User"


# ---------------------------------------------------------------------------
# extract_symbols: utils.rs — const, static, type alias, macro
# ---------------------------------------------------------------------------


def test_utils_rs_const(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/utils.rs")
    consts = [s for s in symbols if s.kind == SymbolKind.CONSTANT]
    names = [c.name for c in consts]
    assert "MAX_RETRIES" in names
    assert "INSTANCE_COUNT" in names


def test_utils_rs_type_alias(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/utils.rs")
    types = [s for s in symbols if s.kind == SymbolKind.TYPE]
    assert any(t.name == "Result" for t in types)


def test_utils_rs_macro(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/utils.rs")
    macros = [s for s in symbols if s.name == "log_debug"]
    assert len(macros) == 1
    assert macros[0].kind == SymbolKind.FUNCTION


def test_utils_rs_function_signature(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.rs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "src/utils.rs")
    fmt_fn = next(s for s in symbols if s.name == "format_name")
    assert fmt_fn.signature is not None
    assert "fn format_name" in fmt_fn.signature


# ---------------------------------------------------------------------------
# parse_imports: main.rs
# ---------------------------------------------------------------------------


def test_main_rs_imports(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "main.rs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/main.rs")
    modules = [i.module for i in imports]
    assert "std::collections" in modules


def test_main_rs_crate_imports(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "main.rs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/main.rs")
    crate_imports = [i for i in imports if i.is_relative]
    assert len(crate_imports) >= 2
    modules = [i.module for i in crate_imports]
    assert "crate::models" in modules
    assert "crate::utils" in modules


def test_main_rs_import_symbols(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "main.rs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/main.rs")
    models_imp = next(i for i in imports if i.module == "crate::models")
    assert "User" in models_imp.symbols
    assert "Role" in models_imp.symbols


# ---------------------------------------------------------------------------
# parse_imports: utils.rs — grouped use
# ---------------------------------------------------------------------------


def test_utils_rs_grouped_imports(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "utils.rs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/utils.rs")
    std_imports = [i for i in imports if i.module == "std"]
    assert len(std_imports) >= 1
    all_symbols: list[str] = []
    for imp in std_imports:
        all_symbols.extend(imp.symbols)
    assert "io" in all_symbols
    assert "fs" in all_symbols


# ---------------------------------------------------------------------------
# parse_imports: models.rs — super:: path
# ---------------------------------------------------------------------------


def test_models_rs_super_import(engine: TreeSitterEngine, adapter: RustAdapter) -> None:
    source = (FIXTURES_DIR / "src" / "models.rs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "src/models.rs")
    super_imports = [i for i in imports if i.module.startswith("super")]
    assert len(super_imports) >= 1
    assert super_imports[0].is_relative


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_crate_import(adapter: RustAdapter) -> None:
    imp = ImportStatement(
        module="crate::models",
        symbols=["User"],
        file_path="src/main.rs",
        line=2,
        is_relative=True,
    )
    file_map = {"models": "src/models.rs"}
    result = adapter.resolve_import(imp, file_map)
    assert result == "src/models.rs"


def test_resolve_external_import(adapter: RustAdapter) -> None:
    imp = ImportStatement(
        module="std::collections",
        symbols=["HashMap"],
        file_path="src/main.rs",
        line=1,
        is_relative=False,
    )
    result = adapter.resolve_import(imp, {})
    assert result is None


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_entry_points(adapter: RustAdapter) -> None:
    files = [
        DiscoveredFile(
            path="src/main.rs",
            absolute_path=str(FIXTURES_DIR / "src" / "main.rs"),
            language="rust",
            size_bytes=100,
        ),
        DiscoveredFile(
            path="src/lib.rs",
            absolute_path=str(FIXTURES_DIR / "src" / "lib.rs"),
            language="rust",
            size_bytes=100,
        ),
        DiscoveredFile(
            path="src/models.rs",
            absolute_path=str(FIXTURES_DIR / "src" / "models.rs"),
            language="rust",
            size_bytes=100,
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "src/main.rs" in entry_points
    assert "src/lib.rs" in entry_points
    assert "src/models.rs" not in entry_points


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_visibility_public(adapter: RustAdapter) -> None:
    sym = Symbol(
        name="User",
        qualified_name="User",
        kind=SymbolKind.TYPE,
        file_path="src/models.rs",
        start_line=1,
        end_line=5,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(sym) == Visibility.PUBLIC


def test_classify_visibility_private(adapter: RustAdapter) -> None:
    sym = Symbol(
        name="internal_helper",
        qualified_name="internal_helper",
        kind=SymbolKind.FUNCTION,
        file_path="src/lib.rs",
        start_line=1,
        end_line=3,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(sym) == Visibility.PRIVATE


def test_classify_visibility_internal(adapter: RustAdapter) -> None:
    sym = Symbol(
        name="Config",
        qualified_name="Config",
        kind=SymbolKind.TYPE,
        file_path="src/models.rs",
        start_line=1,
        end_line=5,
        visibility=Visibility.INTERNAL,
    )
    assert adapter.classify_visibility(sym) == Visibility.INTERNAL
