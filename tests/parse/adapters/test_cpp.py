from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import (
    DiscoveredFile,
    ImportStatement,
    ParsedFile,
    Symbol,
    SymbolKind,
    Visibility,
)
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.cpp import CppAdapter
from archex.parse.engine import TreeSitterEngine
from archex.pipeline.chunker import ASTChunker

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "cpp_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> CppAdapter:
    return CppAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "cpp")


def extract(engine: TreeSitterEngine, adapter: CppAdapter, path: str) -> list[Symbol]:
    source = (FIXTURES_DIR / path).read_bytes()
    tree = parse(engine, source)
    return adapter.extract_symbols(tree, source, path)


def by_qname(symbols: list[Symbol], qualified_name: str) -> list[Symbol]:
    return [s for s in symbols if s.qualified_name == qualified_name]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: CppAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: CppAdapter) -> None:
    assert adapter.language_id == "cpp"


def test_file_extensions(adapter: CppAdapter) -> None:
    assert adapter.file_extensions == [".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"]


def test_tree_sitter_name(adapter: CppAdapter) -> None:
    assert adapter.tree_sitter_name == "cpp"


# ---------------------------------------------------------------------------
# extract_symbols: namespaces
# ---------------------------------------------------------------------------


def test_single_namespace_is_a_module_symbol(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    modules = by_qname(symbols, "geo")
    assert len(modules) == 1
    assert modules[0].kind == SymbolKind.MODULE


def test_cpp17_nested_namespace_syntax(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "shapes.hpp")
    modules = by_qname(symbols, "geo.shapes")
    assert len(modules) == 1
    assert modules[0].kind == SymbolKind.MODULE


def test_classic_nested_namespace_syntax(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b"namespace outer { namespace inner { class Widget {}; } }\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "nested.hpp")
    names = {s.qualified_name for s in symbols}
    assert {"outer", "outer.inner", "outer.inner.Widget"} <= names


def test_anonymous_namespace_has_no_module_symbol(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    source = b"namespace {\n    int helper() { return 1; }\n}\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "anon.cpp")
    assert [s.kind for s in symbols] == [SymbolKind.FUNCTION]


def test_anonymous_namespace_members_get_internal_linkage(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    source = b"namespace {\n    int helper() { return 1; }\n}\nint pub() { return 2; }\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "anon.cpp")
    helper = by_qname(symbols, "helper")[0]
    pub = by_qname(symbols, "pub")[0]
    assert helper.visibility == Visibility.PRIVATE
    assert pub.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# extract_symbols: classes and structs
# ---------------------------------------------------------------------------


def test_class_specifier_maps_to_class_kind(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    point = by_qname(symbols, "geo.Point")[0]
    assert point.kind == SymbolKind.CLASS


def test_struct_specifier_maps_to_type_kind(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "shapes.hpp")
    size = by_qname(symbols, "geo.shapes.Size")[0]
    assert size.kind == SymbolKind.TYPE


def test_top_level_class_is_public(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    point = by_qname(symbols, "geo.Point")[0]
    assert point.visibility == Visibility.PUBLIC


def test_nested_type_declaration(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = (
        b"class Outer {\npublic:\n    struct Meta { int version; };\n"
        b"private:\n    class Impl {};\n};\n"
    )
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "outer.hpp")
    meta = by_qname(symbols, "Outer.Meta")[0]
    impl = by_qname(symbols, "Outer.Impl")[0]
    assert meta.kind == SymbolKind.TYPE
    assert meta.visibility == Visibility.PUBLIC
    assert impl.kind == SymbolKind.CLASS
    assert impl.visibility == Visibility.PRIVATE


def test_kr_combined_struct_and_variable(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b"struct Vec {\n    int x;\n    int y;\n} origin;\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "vec.hpp")
    vec = by_qname(symbols, "Vec")[0]
    assert vec.kind == SymbolKind.TYPE


def test_forward_declaration_is_not_a_symbol(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b"struct Fwd;\nclass Fwd2;\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "fwd.hpp")
    assert symbols == []


def test_self_referential_struct_does_not_duplicate(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "list.hpp")
    nodes = by_qname(symbols, "geo.ListNode")
    assert len(nodes) == 1
    assert nodes[0].kind == SymbolKind.TYPE


# ---------------------------------------------------------------------------
# extract_symbols: functions (free, static, prototype vs definition)
# ---------------------------------------------------------------------------


def test_free_function_definition(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "shapes.cpp")
    funcs = by_qname(symbols, "geo.shapes.area")
    assert len(funcs) == 2
    assert all(s.kind == SymbolKind.FUNCTION for s in funcs)


def test_free_function_prototype_counts_as_symbol(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "shapes.hpp")
    funcs = by_qname(symbols, "geo.shapes.area")
    assert len(funcs) == 2
    assert all(s.signature and ";" not in s.signature for s in funcs)


def test_static_free_function_is_private(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b"static int helper(int x) { return x; }\nint pub(int x) { return x; }\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "util.cpp")
    helper = by_qname(symbols, "helper")[0]
    pub = by_qname(symbols, "pub")[0]
    assert helper.visibility == Visibility.PRIVATE
    assert pub.visibility == Visibility.PUBLIC


def test_function_pointer_variable_is_not_a_function(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    source = b"int (*fp)(int, int);\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "fp.hpp")
    assert symbols == []


# ---------------------------------------------------------------------------
# extract_symbols: overloads -- the milestone's core regression concern
# ---------------------------------------------------------------------------


def test_method_overloads_produce_distinct_symbols(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    moves = by_qname(symbols, "geo.Point.move")
    assert len(moves) == 2
    signatures = {m.signature for m in moves}
    assert signatures == {"void move(int dx, int dy)", "void move(double dx, double dy)"}
    lines = {m.start_line for m in moves}
    assert len(lines) == 2


def test_free_function_overloads_produce_distinct_symbols(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "shapes.hpp")
    areas = by_qname(symbols, "geo.shapes.area")
    assert len(areas) == 2
    signatures = {a.signature for a in areas}
    assert signatures == {
        "int area(int width, int height)",
        "double area(double width, double height)",
    }


def test_template_method_overloads_produce_distinct_symbols(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "container.hpp")
    adds = by_qname(symbols, "geo.Container.add")
    assert len(adds) == 2
    signatures = {a.signature for a in adds}
    assert signatures == {"void add(T item)", "void add(const T& item, int count)"}


def test_overloads_are_not_silently_dropped(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    """A dict/set keyed naively by name would silently collapse overloads to
    one entry -- assert the exact expected symbol count for a Point.hpp
    walk, not just that *some* symbols exist."""
    symbols = extract(engine, adapter, "point.hpp")
    method_names = [s.name for s in symbols if s.kind == SymbolKind.METHOD]
    assert method_names.count("move") == 2
    assert method_names.count("Point") == 2  # two overloaded constructors


def test_overload_symbol_ids_disambiguate_with_zero_collisions(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    """Overloads intentionally share one `qualified_name` (see
    GRAMMAR_EVALUATION.md); final `symbol_id` uniqueness across the whole
    file is proven end-to-end through the real chunker pipeline, the same
    `_disambiguate_symbol_ids` mechanism the Scala adapter's companion-object
    collision already relies on -- not a bespoke per-adapter mechanism."""
    source = (FIXTURES_DIR / "point.hpp").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.hpp")
    parsed = ParsedFile(
        path="point.hpp", language="cpp", symbols=symbols, lines=source.count(b"\n")
    )
    chunks = ASTChunker().chunk_file(parsed, source)

    all_ids = [c.symbol_id for c in chunks]
    assert len(all_ids) == len(set(all_ids))

    move_chunks = sorted((c for c in chunks if c.symbol_name == "move"), key=lambda c: c.start_line)
    assert len(move_chunks) == 2
    assert move_chunks[0].symbol_id != move_chunks[1].symbol_id
    assert move_chunks[1].symbol_id == f"{move_chunks[0].symbol_id}@2"


# ---------------------------------------------------------------------------
# extract_symbols: templates and specializations
# ---------------------------------------------------------------------------


def test_template_class_members_extracted(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "container.hpp")
    container = by_qname(symbols, "geo.Container")[0]
    size_method = by_qname(symbols, "geo.Container.size")[0]
    assert container.kind == SymbolKind.CLASS
    assert size_method.kind == SymbolKind.METHOD


def test_template_class_with_default_parameter(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "pair.hpp")
    pair = by_qname(symbols, "geo.Pair")
    assert len(pair) == 1
    assert pair[0].kind == SymbolKind.CLASS


def test_explicit_specialization_has_distinct_qualified_name(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    """The specialization's qualified name is genuinely distinct from the
    primary template's by construction (not by disambiguation) -- see
    GRAMMAR_EVALUATION.md."""
    symbols = extract(engine, adapter, "pair.hpp")
    primary = by_qname(symbols, "geo.Pair")
    specialization = by_qname(symbols, "geo.Pair<int>")
    assert len(primary) == 1
    assert len(specialization) == 1
    assert primary[0].qualified_name != specialization[0].qualified_name


def test_specialization_members_are_independent_of_primary(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    symbols = extract(engine, adapter, "pair.hpp")
    primary_ctor = by_qname(symbols, "geo.Pair.Pair")
    spec_ctor = by_qname(symbols, "geo.Pair<int>.Pair")
    assert len(primary_ctor) == 1
    assert len(spec_ctor) == 1
    assert primary_ctor[0].signature != spec_ctor[0].signature


# ---------------------------------------------------------------------------
# extract_symbols: constructors, destructors, operators
# ---------------------------------------------------------------------------


def test_overloaded_constructors(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    ctors = by_qname(symbols, "geo.Point.Point")
    assert len(ctors) == 2
    assert all(c.kind == SymbolKind.METHOD for c in ctors)


def test_destructor_is_a_method(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    dtor = by_qname(symbols, "geo.Point.~Point")[0]
    assert dtor.kind == SymbolKind.METHOD
    assert dtor.name == "~Point"


def test_operator_overload_is_a_method(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    op = by_qname(symbols, "geo.Point.operator+")[0]
    assert op.kind == SymbolKind.METHOD
    assert op.name == "operator+"


# ---------------------------------------------------------------------------
# extract_symbols: data members
# ---------------------------------------------------------------------------


def test_multiple_declarators_in_one_field_declaration(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    source = b"class Point {\npublic:\n    int x_, y_;\n};\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "point.hpp")
    names = {s.name for s in symbols if s.kind == SymbolKind.VARIABLE}
    assert names == {"x_", "y_"}


def test_pointer_and_reference_members(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b"class Foo {\npublic:\n    Foo* self_;\n    int& ref_;\n};\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "foo.hpp")
    names = {s.name for s in symbols if s.kind == SymbolKind.VARIABLE}
    assert names == {"self_", "ref_"}


def test_static_const_member_is_constant(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = (
        b"class Config {\npublic:\n    static const int kMaxSize;\n    int instanceField;\n};\n"
    )
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "config.hpp")
    max_size = by_qname(symbols, "Config.kMaxSize")[0]
    instance = by_qname(symbols, "Config.instanceField")[0]
    assert max_size.kind == SymbolKind.CONSTANT
    assert instance.kind == SymbolKind.VARIABLE


# ---------------------------------------------------------------------------
# extract_symbols: visibility (class default private, struct default public)
# ---------------------------------------------------------------------------


def test_class_members_default_private(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    x = by_qname(symbols, "geo.Point.x_")[0]
    assert x.visibility == Visibility.PRIVATE


def test_class_public_section_is_public(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "point.hpp")
    get_x = by_qname(symbols, "geo.Point.getX")[0]
    assert get_x.visibility == Visibility.PUBLIC


def test_struct_members_default_public(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "shapes.hpp")
    width = by_qname(symbols, "geo.shapes.Size.width")[0]
    assert width.visibility == Visibility.PUBLIC


def test_protected_maps_to_internal(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b"class Base {\nprotected:\n    int shared_;\n};\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "base.hpp")
    shared = by_qname(symbols, "Base.shared_")[0]
    assert shared.visibility == Visibility.INTERNAL


# ---------------------------------------------------------------------------
# extract_symbols: preprocessor conditionals and extern "C" (contained gap)
# ---------------------------------------------------------------------------


def test_extracts_through_extern_c_and_ifdef(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "platform.hpp")
    names = {s.name for s in symbols}
    assert "platform_name" in names


def test_ifdef_branch_boundaries_are_independent(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    """The two #ifdef/#else platform_sleep_ms prototypes must keep distinct,
    independent line boundaries despite the file's one documented,
    non-cascading is_missing diagnostic (see GRAMMAR_EVALUATION.md)."""
    symbols = extract(engine, adapter, "platform.hpp")
    sleeps = [s for s in symbols if s.name == "platform_sleep_ms"]
    lines = {s.start_line for s in sleeps}
    assert len(sleeps) == 2
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# Inline source: edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "empty.cpp") == []


def test_class_only_declarations_still_walk(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    symbols = extract(engine, adapter, "pair.hpp")
    assert len(symbols) > 0
    assert all(isinstance(s, Symbol) for s in symbols)


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_quoted_include(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = (FIXTURES_DIR / "point.cpp").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "point.cpp")
    assert [imp.module for imp in imports] == ["point.hpp"]
    assert imports[0].is_relative is True


def test_parse_angle_bracket_include(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = (FIXTURES_DIR / "list.cpp").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "list.cpp")
    modules = {imp.module: imp for imp in imports}
    assert modules["cstdlib"].is_relative is False


def test_parse_includes_through_extern_c_and_ifdef(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    source = (FIXTURES_DIR / "platform.hpp").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "platform.hpp")
    assert imports == []


def test_multiple_includes_in_order(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = (FIXTURES_DIR / "main.cpp").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "main.cpp")
    modules = [imp.module for imp in imports]
    assert modules == ["cstdio", "list.hpp", "platform.hpp", "point.hpp", "shapes.hpp"]


def test_include_not_present_inside_namespace(
    engine: TreeSitterEngine, adapter: CppAdapter
) -> None:
    source = b'namespace geo {\n#include "inner.hpp"\n}\n'
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "weird.cpp")
    assert imports == []


def test_empty_file_has_no_imports(engine: TreeSitterEngine, adapter: CppAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    assert adapter.parse_imports(tree, source, "empty.cpp") == []


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_quoted_include_same_directory(adapter: CppAdapter) -> None:
    file_map = {"list.hpp": "list.hpp", "point.hpp": "point.hpp"}
    imp = ImportStatement(module="point.hpp", file_path="list.hpp", line=5, is_relative=True)
    assert adapter.resolve_import(imp, file_map) == "point.hpp"


def test_resolve_quoted_include_subdirectory(adapter: CppAdapter) -> None:
    file_map = {"src/main.cpp": "src/main.cpp", "include/foo.hpp": "include/foo.hpp"}
    imp = ImportStatement(
        module="../include/foo.hpp", file_path="src/main.cpp", line=1, is_relative=True
    )
    assert adapter.resolve_import(imp, file_map) == "include/foo.hpp"


def test_resolve_quoted_include_basename_fallback(adapter: CppAdapter) -> None:
    file_map = {"main.cpp": "main.cpp", "include/deep/foo.hpp": "include/deep/foo.hpp"}
    imp = ImportStatement(module="foo.hpp", file_path="main.cpp", line=1, is_relative=True)
    assert adapter.resolve_import(imp, file_map) == "include/deep/foo.hpp"


def test_resolve_quoted_include_unresolvable(adapter: CppAdapter) -> None:
    file_map = {"main.cpp": "main.cpp"}
    imp = ImportStatement(module="missing.hpp", file_path="main.cpp", line=1, is_relative=True)
    assert adapter.resolve_import(imp, file_map) is None


def test_resolve_angle_bracket_always_external(adapter: CppAdapter) -> None:
    file_map = {"vector": "vector"}  # even a coincidental name match
    imp = ImportStatement(module="vector", file_path="main.cpp", line=1, is_relative=False)
    assert adapter.resolve_import(imp, file_map) is None


def test_resolve_include_of_c_tier_header(adapter: CppAdapter) -> None:
    """A .cpp file including a C-tier .h header resolves the same way --
    file_map spans the whole repo, not just cpp-tier files."""
    file_map = {"legacy.h": "legacy.h", "main.cpp": "main.cpp"}
    imp = ImportStatement(module="legacy.h", file_path="main.cpp", line=1, is_relative=True)
    assert adapter.resolve_import(imp, file_map) == "legacy.h"


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_entry_point(adapter: CppAdapter) -> None:
    files = [
        DiscoveredFile(
            path="main.cpp",
            absolute_path=str(FIXTURES_DIR / "main.cpp"),
            language="cpp",
        )
    ]
    assert adapter.detect_entry_points(files) == ["main.cpp"]


def test_header_never_an_entry_point(adapter: CppAdapter, tmp_path: Path) -> None:
    header = tmp_path / "fake_main.hpp"
    header.write_text("int main() { return 0; }\n")
    files = [DiscoveredFile(path="fake_main.hpp", absolute_path=str(header), language="cpp")]
    assert adapter.detect_entry_points(files) == []


def test_main_prototype_is_not_an_entry_point(adapter: CppAdapter, tmp_path: Path) -> None:
    source_file = tmp_path / "proto.cpp"
    source_file.write_text("int main();\n")
    files = [DiscoveredFile(path="proto.cpp", absolute_path=str(source_file), language="cpp")]
    assert adapter.detect_entry_points(files) == []


def test_non_main_file_not_entry_point(adapter: CppAdapter, tmp_path: Path) -> None:
    lib_file = tmp_path / "lib.cpp"
    lib_file.write_text("void helper() {}\n")
    files = [DiscoveredFile(path="lib.cpp", absolute_path=str(lib_file), language="cpp")]
    assert adapter.detect_entry_points(files) == []


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_visibility_public(adapter: CppAdapter) -> None:
    s = Symbol(
        name="getX",
        qualified_name="geo.Point.getX",
        kind=SymbolKind.METHOD,
        file_path="point.hpp",
        start_line=1,
        end_line=1,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_visibility_private(adapter: CppAdapter) -> None:
    s = Symbol(
        name="x_",
        qualified_name="geo.Point.x_",
        kind=SymbolKind.VARIABLE,
        file_path="point.hpp",
        start_line=1,
        end_line=1,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE
