from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import SymbolKind, Visibility
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
