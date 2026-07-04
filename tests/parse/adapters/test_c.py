from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.c import CAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "c_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> CAdapter:
    return CAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "c")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: CAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: CAdapter) -> None:
    assert adapter.language_id == "c"


def test_file_extensions(adapter: CAdapter) -> None:
    assert adapter.file_extensions == [".c", ".h"]


def test_tree_sitter_name(adapter: CAdapter) -> None:
    assert adapter.tree_sitter_name == "c"


# ---------------------------------------------------------------------------
# extract_symbols: functions (definitions)
# ---------------------------------------------------------------------------


def test_extract_function_definition(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.c").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.c")
    funcs = {s.name: s for s in symbols if s.kind == SymbolKind.FUNCTION}
    assert "point_make" in funcs
    assert "point_distance_squared" in funcs


def test_static_function_is_private(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.c").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.c")
    square = next(s for s in symbols if s.name == "square")
    assert square.visibility == Visibility.PRIVATE
    assert square.kind == SymbolKind.FUNCTION


def test_non_static_function_is_public(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.c").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.c")
    make = next(s for s in symbols if s.name == "point_make")
    assert make.visibility == Visibility.PUBLIC


def test_pointer_returning_function(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "list.c").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "list.c")
    push = next(s for s in symbols if s.name == "list_push")
    assert push.kind == SymbolKind.FUNCTION
    assert "struct ListNode *list_push" in (push.signature or "")


def test_function_signature_excludes_body(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.c").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.c")
    make = next(s for s in symbols if s.name == "point_make")
    assert make.signature == "Point point_make(int x, int y)"


def test_function_qualified_name_is_flat(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.c").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.c")
    make = next(s for s in symbols if s.name == "point_make")
    assert make.qualified_name == "point_make"


# ---------------------------------------------------------------------------
# extract_symbols: functions (prototypes -- headers are mostly bodyless)
# ---------------------------------------------------------------------------


def test_extract_function_prototypes(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.h")
    funcs = {s.name: s for s in symbols if s.kind == SymbolKind.FUNCTION}
    assert "point_make" in funcs
    assert "point_distance_squared" in funcs
    # Prototypes have no body: start_line == end_line.
    assert funcs["point_make"].start_line == funcs["point_make"].end_line


def test_prototype_signature_excludes_semicolon(
    engine: TreeSitterEngine, adapter: CAdapter
) -> None:
    source = (FIXTURES_DIR / "point.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.h")
    make = next(s for s in symbols if s.name == "point_make")
    assert make.signature == "Point point_make(int x, int y)"
    assert ";" not in (make.signature or "")


def test_function_pointer_variable_is_not_a_function(
    engine: TreeSitterEngine, adapter: CAdapter
) -> None:
    source = b"int (*fp)(int, int);\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "fp.c")
    assert symbols == []


# ---------------------------------------------------------------------------
# extract_symbols: structs
# ---------------------------------------------------------------------------


def test_extract_named_typedef_struct(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.h")
    point = next(s for s in symbols if s.name == "Point")
    assert point.kind == SymbolKind.TYPE
    assert point.visibility == Visibility.PUBLIC


def test_extract_anonymous_typedef_struct(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.h")
    names = {s.name for s in symbols if s.kind == SymbolKind.TYPE}
    assert "Size" in names


def test_extract_bare_struct(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "list.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "list.h")
    node = next(s for s in symbols if s.name == "ListNode")
    assert node.kind == SymbolKind.TYPE


def test_self_referential_field_does_not_duplicate_struct(
    engine: TreeSitterEngine, adapter: CAdapter
) -> None:
    """struct ListNode { ...; struct ListNode *next; }; must report ListNode
    exactly once -- the bodyless `struct ListNode *next` field reference
    must not be mistaken for a second (invalid, bodyless) definition."""
    source = (FIXTURES_DIR / "list.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "list.h")
    names = [s.name for s in symbols if s.kind == SymbolKind.TYPE]
    assert names.count("ListNode") == 1


def test_forward_declaration_is_not_a_symbol(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = b"struct Fwd;\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "fwd.h")
    assert symbols == []


def test_kr_combined_struct_and_variable(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = b"struct Vec {\n    int x;\n    int y;\n} origin;\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "vec.c")
    vec = next(s for s in symbols if s.name == "Vec")
    assert vec.kind == SymbolKind.TYPE


def test_typedef_alias_preferred_over_tag_name(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = b"typedef struct Rect {\n    int w;\n} Rect;\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "rect.h")
    assert len(symbols) == 1
    assert symbols[0].name == "Rect"


# ---------------------------------------------------------------------------
# extract_symbols: preprocessor conditionals and extern "C" blocks
# ---------------------------------------------------------------------------


def test_extracts_through_extern_c_and_ifdef(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "platform.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "platform.h")
    names = [s.name for s in symbols]
    assert names.count("platform_sleep_ms") == 2
    assert "platform_name" in names


def test_ifdef_branch_boundaries_are_independent(
    engine: TreeSitterEngine, adapter: CAdapter
) -> None:
    """The two #ifdef/#else platform_sleep_ms prototypes must keep distinct,
    correct line ranges despite the extern "C" wrapper's contained parser
    diagnostic on this fixture (see GRAMMAR_EVALUATION.md)."""
    source = (FIXTURES_DIR / "platform.h").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "platform.h")
    sleep_variants = [s for s in symbols if s.name == "platform_sleep_ms"]
    assert len(sleep_variants) == 2
    lines = {s.start_line for s in sleep_variants}
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_quoted_include(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "point.c").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "point.c")
    assert len(imports) == 1
    assert imports[0].module == "point.h"
    assert imports[0].is_relative is True


def test_parse_angle_bracket_include(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "list.c").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "list.c")
    modules = {imp.module: imp for imp in imports}
    assert "stdlib.h" in modules
    assert modules["stdlib.h"].is_relative is False


def test_parse_includes_through_extern_c(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (
        b'#ifdef __cplusplus\nextern "C" {\n#endif\n'
        b'#include "inner.h"\n#ifdef __cplusplus\n}\n#endif\n'
    )
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "wrapped.h")
    assert [imp.module for imp in imports] == ["inner.h"]


def test_multiple_includes_in_order(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = (FIXTURES_DIR / "main.c").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "main.c")
    modules = [imp.module for imp in imports]
    assert modules == ["list.h", "platform.h", "point.h", "stdio.h"]


# ---------------------------------------------------------------------------
# Inline source: edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "empty.c") == []
    assert adapter.parse_imports(tree, source, "empty.c") == []


def test_include_only_file(engine: TreeSitterEngine, adapter: CAdapter) -> None:
    source = b'#include <stdio.h>\n#include "local.h"\n'
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "includes.h") == []
    imports = adapter.parse_imports(tree, source, "includes.h")
    assert len(imports) == 2


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_quoted_include_same_directory(adapter: CAdapter) -> None:
    file_map = {"list.h": "list.h", "point.h": "point.h"}
    imp = ImportStatement(module="point.h", file_path="list.h", line=4, is_relative=True)
    assert adapter.resolve_import(imp, file_map) == "point.h"


def test_resolve_quoted_include_subdirectory(adapter: CAdapter) -> None:
    file_map = {"src/main.c": "src/main.c", "include/foo.h": "include/foo.h"}
    imp = ImportStatement(
        module="../include/foo.h", file_path="src/main.c", line=1, is_relative=True
    )
    assert adapter.resolve_import(imp, file_map) == "include/foo.h"


def test_resolve_quoted_include_basename_fallback(adapter: CAdapter) -> None:
    """Header lives on a compiler -I search path, not next to the includer --
    the adapter has no build-system visibility, so basename matching is the
    best available fallback."""
    file_map = {"src/main.c": "src/main.c", "include/deep/foo.h": "include/deep/foo.h"}
    imp = ImportStatement(module="foo.h", file_path="src/main.c", line=1, is_relative=True)
    assert adapter.resolve_import(imp, file_map) == "include/deep/foo.h"


def test_resolve_quoted_include_unresolvable(adapter: CAdapter) -> None:
    file_map = {"main.c": "main.c"}
    imp = ImportStatement(module="missing.h", file_path="main.c", line=1, is_relative=True)
    assert adapter.resolve_import(imp, file_map) is None


def test_resolve_angle_bracket_always_external(adapter: CAdapter) -> None:
    file_map = {"stdio.h": "stdio.h"}  # even a coincidental name match
    imp = ImportStatement(module="stdio.h", file_path="main.c", line=1, is_relative=False)
    assert adapter.resolve_import(imp, file_map) is None


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_entry_point(adapter: CAdapter) -> None:
    files = [
        DiscoveredFile(path="main.c", absolute_path=str(FIXTURES_DIR / "main.c"), language="c"),
        DiscoveredFile(path="point.c", absolute_path=str(FIXTURES_DIR / "point.c"), language="c"),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert entry_points == ["main.c"]


def test_header_never_an_entry_point(adapter: CAdapter, tmp_path: Path) -> None:
    header = tmp_path / "fake_main.h"
    header.write_text("int main(void) {\n    return 0;\n}\n")
    files = [DiscoveredFile(path="fake_main.h", absolute_path=str(header), language="c")]
    assert adapter.detect_entry_points(files) == []


def test_main_prototype_is_not_an_entry_point(adapter: CAdapter, tmp_path: Path) -> None:
    source_file = tmp_path / "proto.c"
    source_file.write_text("int main(void);\n")
    files = [DiscoveredFile(path="proto.c", absolute_path=str(source_file), language="c")]
    assert adapter.detect_entry_points(files) == []


def test_non_main_file_not_entry_point(adapter: CAdapter, tmp_path: Path) -> None:
    lib_file = tmp_path / "lib.c"
    lib_file.write_text("int add(int a, int b) {\n    return a + b;\n}\n")
    files = [DiscoveredFile(path="lib.c", absolute_path=str(lib_file), language="c")]
    assert adapter.detect_entry_points(files) == []


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_visibility_public(adapter: CAdapter) -> None:
    s = Symbol(
        name="point_make",
        qualified_name="point_make",
        kind=SymbolKind.FUNCTION,
        file_path="point.c",
        start_line=1,
        end_line=1,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_visibility_private(adapter: CAdapter) -> None:
    s = Symbol(
        name="square",
        qualified_name="square",
        kind=SymbolKind.FUNCTION,
        file_path="point.c",
        start_line=1,
        end_line=1,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE
