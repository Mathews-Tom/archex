from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.scala import ScalaAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "scala_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> ScalaAdapter:
    return ScalaAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "scala")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: ScalaAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: ScalaAdapter) -> None:
    assert adapter.language_id == "scala"


def test_file_extensions(adapter: ScalaAdapter) -> None:
    assert adapter.file_extensions == [".scala", ".sc"]


def test_tree_sitter_name(adapter: ScalaAdapter) -> None:
    assert adapter.tree_sitter_name == "scala"


# ---------------------------------------------------------------------------
# extract_symbols: package qualification
# ---------------------------------------------------------------------------


def test_semicolon_style_package_qualifies_top_level_object(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "App.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "App.scala")
    main = next(s for s in symbols if s.name == "Main")
    assert main.qualified_name == "com.example.app.Main"
    assert main.parent == "com.example.app"


def test_chained_bodyless_package_accumulates_namespace(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # Two consecutive bodyless `package` clauses accumulate into a single
    # ambient namespace ("com.example" then "models" -> "com.example.models")
    # rather than only honoring the first or the last one seen.
    source = b"package com.example\npackage models\n\nclass Widget\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "widget.scala")
    widget = next(s for s in symbols if s.name == "Widget")
    assert widget.qualified_name == "com.example.models.Widget"
    assert widget.parent == "com.example.models"


def test_brace_style_package_recurses_into_body(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "legacy" / "Adjacent.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "legacy/Adjacent.scala")
    classes = {s.name: s for s in symbols if s.kind == SymbolKind.CLASS}
    assert classes["LegacyWidget"].qualified_name == "com.example.app.legacy.LegacyWidget"
    assert classes["LegacyGadget"].qualified_name == "com.example.app.legacy.LegacyGadget"
    assert classes["LegacyRegistry"].parent == "com.example.app.legacy"


def test_adjacent_classes_in_brace_package_have_independent_boundaries(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # Regression guard for the cascading-corruption failure mode probed in
    # GRAMMAR_EVALUATION.md: adjacent same-kind declarations inside a brace
    # package body must keep independently correct start/end lines.
    source = (FIXTURES_DIR / "legacy" / "Adjacent.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "legacy/Adjacent.scala")
    classes = {s.name: s for s in symbols if s.kind == SymbolKind.CLASS}
    assert (classes["LegacyWidget"].start_line, classes["LegacyWidget"].end_line) == (3, 5)
    assert (classes["LegacyGadget"].start_line, classes["LegacyGadget"].end_line) == (7, 9)
    assert (classes["LegacyRegistry"].start_line, classes["LegacyRegistry"].end_line) == (11, 13)


# ---------------------------------------------------------------------------
# extract_symbols: classes and objects
# ---------------------------------------------------------------------------


def test_extract_class(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "Address.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/Address.scala")
    address = next(s for s in symbols if s.name == "Address")
    assert address.kind == SymbolKind.CLASS
    assert address.qualified_name == "com.example.app.models.Address"
    assert address.visibility == Visibility.PUBLIC


def test_object_definition_reported_as_class_kind(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # object_definition has no dedicated Symbol kind; the adapter reports it
    # as CLASS, matching the existing Kotlin object_declaration -> CLASS
    # precedent (no dedicated singleton kind exists on the model).
    source = (FIXTURES_DIR / "util" / "StringUtils.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "util/StringUtils.scala")
    utils = next(s for s in symbols if s.name == "StringUtils")
    assert utils.kind == SymbolKind.CLASS
    assert utils.qualified_name == "com.example.app.util.StringUtils"


# ---------------------------------------------------------------------------
# extract_symbols: traits
# ---------------------------------------------------------------------------


def test_trait_reported_as_interface(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    # The Symbol model has no dedicated trait kind; a Scala trait is
    # structurally closest to an interface, so it reports as INTERFACE.
    source = (FIXTURES_DIR / "contracts" / "Greeter.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "contracts/Greeter.scala")
    greeter = next(s for s in symbols if s.name == "Greeter")
    assert greeter.kind == SymbolKind.INTERFACE
    assert greeter.qualified_name == "com.example.app.contracts.Greeter"
    assert greeter.visibility == Visibility.PUBLIC


def test_trait_abstract_method_member(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "contracts" / "Greeter.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "contracts/Greeter.scala")
    greet = next(s for s in symbols if s.name == "greet")
    assert greet.kind == SymbolKind.METHOD
    assert greet.parent == "com.example.app.contracts.Greeter"
    assert greet.signature == "def greet(name: String): String"


def test_trait_extending_trait_has_default_method(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "contracts" / "FriendlyGreeter.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "contracts/FriendlyGreeter.scala")
    traits = {s.name: s for s in symbols if s.kind == SymbolKind.INTERFACE}
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert traits["FriendlyGreeter"].qualified_name == "com.example.app.contracts.FriendlyGreeter"
    assert traits["LoudGreeter"].qualified_name == "com.example.app.contracts.LoudGreeter"
    assert methods["greet"].parent == "com.example.app.contracts.FriendlyGreeter"
    assert methods["shout"].parent == "com.example.app.contracts.LoudGreeter"
    assert methods["shout"].signature == "def shout(name: String): String"


# ---------------------------------------------------------------------------
# extract_symbols: companion object name/kind collision
# ---------------------------------------------------------------------------


def test_companion_class_and_object_share_qualified_name_and_kind(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # `case class User` (line 4) and `object User` (line 6) are independent
    # top-level siblings that intentionally collide on both qualified_name
    # and kind -- documented in GRAMMAR_EVALUATION.md as expected, resolved
    # downstream by pipeline/chunker.py's disambiguation. A dict keyed by
    # name would silently collapse this to one entry, so assert on the raw
    # list instead to prove both survive extraction.
    source = (FIXTURES_DIR / "models" / "User.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.scala")
    users = [s for s in symbols if s.name == "User"]
    assert len(users) == 2
    assert {s.qualified_name for s in users} == {"com.example.app.models.User"}
    assert {s.kind for s in users} == {SymbolKind.CLASS}
    assert sorted(s.start_line for s in users) == [4, 6]


# ---------------------------------------------------------------------------
# extract_symbols: methods
# ---------------------------------------------------------------------------


def test_method_visibility_default_public(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.scala")
    validate = next(s for s in symbols if s.name == "validate")
    assert validate.visibility == Visibility.PUBLIC
    assert validate.signature == "def validate(user: User): Boolean"


def test_method_visibility_private(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.scala")
    normalize_email = next(s for s in symbols if s.name == "normalizeEmail")
    assert normalize_email.visibility == Visibility.PRIVATE
    assert normalize_email.parent == "com.example.app.models.User"


def test_qualified_protected_maps_to_internal(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.scala")
    audit_log = next(s for s in symbols if s.name == "auditLog")
    assert audit_log.visibility == Visibility.INTERNAL  # protected[this] -> INTERNAL
    assert audit_log.signature == "def auditLog(message: String): Unit"


def test_qualified_private_bracket_maps_to_private(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # No fixture exercises the bracketed `private[X]` form (only
    # `protected[this]` in UserService.scala) -- confirm the bracket
    # qualifier is ignored for the PRIVATE mapping too, on both a method
    # and a val, and regardless of the qualifier's own contents.
    source = (
        b"package demo\n\n"
        b"class Widget {\n"
        b"  private[this] def secret(): Int = 42\n"
        b"  private[demo] val guarded: Int = 1\n"
        b"}\n"
    )
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "widget.scala")
    members = {s.name: s for s in symbols if s.parent == "demo.Widget"}
    assert members["secret"].visibility == Visibility.PRIVATE
    assert members["guarded"].visibility == Visibility.PRIVATE


def test_method_without_parameter_list_gets_synthesized_empty_parens(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # `def count: Int` has no `parameters` node at all in the grammar; the
    # adapter must not crash or omit the parens, it synthesizes "()".
    source = (FIXTURES_DIR / "services" / "UserService.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.scala")
    count = next(s for s in symbols if s.name == "count")
    assert count.signature == "def count(): Int"


# ---------------------------------------------------------------------------
# extract_symbols: top-level functions
# ---------------------------------------------------------------------------


def test_curried_top_level_function_signature_and_no_parent(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # Curried functions declare multiple sibling `parameters` nodes; both
    # parameter lists must appear in the signature. Top-level functions are
    # always parent=None even though the qualified_name carries the
    # namespace prefix.
    source = b"package demo\n\ndef add(a: Int)(b: Int): Int = a + b\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "add.scala")
    add = next(s for s in symbols if s.name == "add")
    assert add.kind == SymbolKind.FUNCTION
    assert add.parent is None
    assert add.qualified_name == "demo.add"
    assert add.signature == "def add(a: Int)(b: Int): Int"


# ---------------------------------------------------------------------------
# extract_symbols: constants and variables
# ---------------------------------------------------------------------------


def test_val_definition_is_constant(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "User.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/User.scala")
    default_role = next(s for s in symbols if s.name == "DEFAULT_ROLE")
    assert default_role.kind == SymbolKind.CONSTANT
    assert default_role.visibility == Visibility.PUBLIC
    assert default_role.parent == "com.example.app.models.User"


def test_var_definition_is_variable(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.scala")
    registered = next(s for s in symbols if s.name == "registered")
    assert registered.kind == SymbolKind.VARIABLE
    assert registered.visibility == Visibility.PRIVATE
    assert registered.parent == "com.example.app.services.UserService"


def test_destructuring_val_pattern_is_silently_skipped(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    # Only the plain-identifier `pattern` form is extracted; a destructuring
    # binding has no single name to report and must be skipped without
    # crashing, while the plain val alongside it still gets extracted.
    source = b"package demo\n\nobject Foo {\n  val (a, b) = (1, 2)\n  val plain = 5\n}\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "foo.scala")
    assert {s.name for s in symbols} == {"Foo", "plain"}


# ---------------------------------------------------------------------------
# extract_symbols: nested types
# ---------------------------------------------------------------------------


def test_nested_class_inside_object_recurses_with_qualified_parent(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "util" / "StringUtils.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "util/StringUtils.scala")
    builder = next(s for s in symbols if s.name == "Builder")
    assert builder.kind == SymbolKind.CLASS
    assert builder.qualified_name == "com.example.app.util.StringUtils.Builder"
    assert builder.parent == "com.example.app.util.StringUtils"
    members = {s.name: s for s in symbols if s.parent == "com.example.app.util.StringUtils.Builder"}
    assert members["append"].signature == "def append(part: String): Builder"
    assert members["build"].signature == "def build(): String"
    assert members["parts"].kind == SymbolKind.VARIABLE
    assert members["parts"].visibility == Visibility.PRIVATE


def test_nested_private_object_inside_class(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.scala").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/UserService.scala")
    metrics = next(s for s in symbols if s.name == "Metrics")
    assert metrics.kind == SymbolKind.CLASS
    assert metrics.visibility == Visibility.PRIVATE
    assert metrics.qualified_name == "com.example.app.services.UserService.Metrics"
    record_lookup = next(s for s in symbols if s.name == "recordLookup")
    assert record_lookup.parent == "com.example.app.services.UserService.Metrics"
    assert record_lookup.signature == "def recordLookup(): Unit"


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_simple_import(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "App.scala").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "App.scala")
    service_import = next(i for i in imports if i.module == "com.example.app.services.UserService")
    assert service_import.alias is None
    assert service_import.line == 6
    assert service_import.is_relative is False


def test_parse_wildcard_import(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "UserService.scala").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "services/UserService.scala")
    wildcard = next(i for i in imports if i.module.endswith("._"))
    assert wildcard.module == "com.example.app.models._"
    assert wildcard.line == 5


def test_parse_two_name_group_produces_one_statement_per_name(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "App.scala").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "App.scala")
    util_group = [i for i in imports if i.line == 3]
    assert {i.module for i in util_group} == {"scala.util.Failure", "scala.util.Success"}
    assert all(i.alias is None for i in util_group)


def test_parse_four_name_group_produces_one_statement_per_name(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "App.scala").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "App.scala")
    shapes_group = [i for i in imports if i.line == 7]
    assert {i.module for i in shapes_group} == {
        "com.example.app.shapes.Circle",
        "com.example.app.shapes.Empty",
        "com.example.app.shapes.Shape",
        "com.example.app.shapes.Square",
    }
    assert all(i.alias is None for i in shapes_group)


def test_parse_arrow_renamed_selector_sets_alias_but_keeps_original_module(
    engine: TreeSitterEngine, adapter: ScalaAdapter
) -> None:
    source = (FIXTURES_DIR / "App.scala").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "App.scala")
    renamed = next(i for i in imports if i.alias == "Greetable")
    assert renamed.module == "com.example.app.contracts.Greeter"
    assert renamed.line == 4


def test_imports_not_relative(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = (FIXTURES_DIR / "App.scala").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "App.scala")
    assert imports
    for imp in imports:
        assert imp.is_relative is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "empty.scala") == []
    assert adapter.parse_imports(tree, source, "empty.scala") == []


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.scala"):
        source = f.read_bytes()
        tree = parse(engine, source)
        for s in adapter.extract_symbols(tree, source, str(f.relative_to(FIXTURES_DIR))):
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_members_have_parent(engine: TreeSitterEngine, adapter: ScalaAdapter) -> None:
    member_kinds = {SymbolKind.METHOD, SymbolKind.CONSTANT, SymbolKind.VARIABLE}
    for f in FIXTURES_DIR.rglob("*.scala"):
        source = f.read_bytes()
        tree = parse(engine, source)
        for s in adapter.extract_symbols(tree, source, str(f.relative_to(FIXTURES_DIR))):
            if s.kind in member_kinds:
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"
