from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.php import PHPAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "php_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> PHPAdapter:
    return PHPAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "php")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: PHPAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: PHPAdapter) -> None:
    assert adapter.language_id == "php"


def test_file_extensions(adapter: PHPAdapter) -> None:
    assert adapter.file_extensions == [".php"]


def test_tree_sitter_name(adapter: PHPAdapter) -> None:
    assert adapter.tree_sitter_name == "php"


# ---------------------------------------------------------------------------
# extract_symbols: namespace qualification
# ---------------------------------------------------------------------------


def test_semicolon_namespace_qualifies_class(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    user = next(s for s in symbols if s.kind == SymbolKind.CLASS)
    # `\`-separated PHP namespace becomes a `.`-joined qualified_name.
    assert user.qualified_name == "App.Models.User"


def test_brace_namespace_qualifies_declarations(
    engine: TreeSitterEngine, adapter: PHPAdapter
) -> None:
    source = (FIXTURES_DIR / "Legacy" / "BraceNamespace.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Legacy/BraceNamespace.php")
    loggable = next(s for s in symbols if s.kind == SymbolKind.INTERFACE)
    logger = next(s for s in symbols if s.kind == SymbolKind.CLASS)
    assert loggable.qualified_name == "App.Legacy.Loggable"
    assert logger.qualified_name == "App.Legacy.FileLogger"


# ---------------------------------------------------------------------------
# extract_symbols: classes
# ---------------------------------------------------------------------------


def test_extract_class(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    assert any(s.name == "User" for s in classes)


def test_class_visibility_defaults_public(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    # PHP has no top-level visibility modifier for classes; the adapter defaults to PUBLIC.
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    user = next(s for s in symbols if s.kind == SymbolKind.CLASS)
    assert user.visibility == Visibility.PUBLIC


def test_class_members_nested_under_qualified_parent(
    engine: TreeSitterEngine, adapter: PHPAdapter
) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    members = [s for s in symbols if s.kind != SymbolKind.CLASS]
    assert members
    for m in members:
        assert m.parent == "App.Models.User"


# ---------------------------------------------------------------------------
# extract_symbols: interfaces
# ---------------------------------------------------------------------------


def test_extract_interface(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Contracts" / "Arrayable.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Contracts/Arrayable.php")
    iface = next(s for s in symbols if s.kind == SymbolKind.INTERFACE)
    assert iface.name == "Arrayable"
    assert iface.visibility == Visibility.PUBLIC


def test_interface_method_member(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Contracts" / "Arrayable.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Contracts/Arrayable.php")
    to_array = next(s for s in symbols if s.kind == SymbolKind.METHOD)
    assert to_array.name == "toArray"
    assert to_array.parent == "App.Contracts.Arrayable"
    assert to_array.visibility == Visibility.PUBLIC


def test_interface_method_omitted_modifier_defaults_public(
    engine: TreeSitterEngine, adapter: PHPAdapter
) -> None:
    # None of the fixtures omit the visibility modifier entirely, so this
    # exercises the fallback path directly against an inline source.
    source = b"""<?php
namespace App\\Test;

class Foo
{
    function bar(): void
    {
    }
}
"""
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Foo.php")
    bar = next(s for s in symbols if s.kind == SymbolKind.METHOD)
    assert bar.visibility == Visibility.PUBLIC


# ---------------------------------------------------------------------------
# extract_symbols: traits
# ---------------------------------------------------------------------------


def test_trait_reported_as_class(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    # The Symbol model has no dedicated trait kind; a trait's body has the
    # same member shape as a class, so the adapter reports it as CLASS.
    source = (FIXTURES_DIR / "Traits" / "HasTimestamps.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Traits/HasTimestamps.php")
    trait = next(s for s in symbols if s.name == "HasTimestamps")
    assert trait.kind == SymbolKind.CLASS
    assert trait.qualified_name == "App.Traits.HasTimestamps"


def test_trait_members(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Traits" / "HasTimestamps.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Traits/HasTimestamps.php")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["touch"].parent == "App.Traits.HasTimestamps"
    assert methods["createdAt"].parent == "App.Traits.HasTimestamps"
    prop = next(s for s in symbols if s.kind == SymbolKind.VARIABLE)
    assert prop.name == "createdAt"
    assert prop.visibility == Visibility.INTERNAL  # protected -> INTERNAL


# ---------------------------------------------------------------------------
# extract_symbols: enums
# ---------------------------------------------------------------------------


def test_extract_enum(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "Status.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/Status.php")
    status = next(s for s in symbols if s.name == "Status")
    assert status.kind == SymbolKind.ENUM
    assert status.qualified_name == "App.Models.Status"


def test_enum_cases_are_constants(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "Status.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/Status.php")
    cases = {s.name: s for s in symbols if s.kind == SymbolKind.CONSTANT}
    assert set(cases) == {"Active", "Inactive"}
    for c in cases.values():
        assert c.qualified_name.startswith("App.Models.Status.")
        assert c.parent == "App.Models.Status"


def test_enum_method(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "Status.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/Status.php")
    label = next(s for s in symbols if s.kind == SymbolKind.METHOD)
    assert label.name == "label"
    assert label.parent == "App.Models.Status"


# ---------------------------------------------------------------------------
# extract_symbols: methods
# ---------------------------------------------------------------------------


def test_method_visibility_mapping(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["toArray"].visibility == Visibility.PUBLIC
    assert methods["validate"].visibility == Visibility.PRIVATE


def test_protected_method_maps_to_internal(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.php")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["limit"].visibility == Visibility.INTERNAL  # protected -> INTERNAL


def test_static_method(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.php")
    make = next(s for s in symbols if s.kind == SymbolKind.METHOD and s.name == "make")
    assert make.visibility == Visibility.PUBLIC
    assert make.parent == "App.Services.UserService"


def test_method_signature(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    create = next(s for s in symbols if s.kind == SymbolKind.METHOD and s.name == "create")
    assert create.signature is not None
    assert "string $name" in create.signature
    assert "self" in create.signature


# ---------------------------------------------------------------------------
# extract_symbols: constructor property promotion
# ---------------------------------------------------------------------------


def test_constructor_extracted_as_method(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    ctor = next(s for s in symbols if s.kind == SymbolKind.METHOD and s.name == "__construct")
    assert ctor.parent == "App.Models.User"
    assert ctor.visibility == Visibility.PUBLIC


def test_constructor_promoted_property_becomes_variable(
    engine: TreeSitterEngine, adapter: PHPAdapter
) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    ctor = next(s for s in symbols if s.kind == SymbolKind.METHOD and s.name == "__construct")
    promoted = next(s for s in symbols if s.kind == SymbolKind.VARIABLE and s.name == "name")
    # `int $id = 0` is a plain (non-promoted) constructor parameter and never
    # produces a symbol of its own; only the promoted
    # `private readonly string $name` parameter does. The class also
    # separately declares a genuine `private int $id;` field (line 14) whose
    # own VARIABLE symbol is anchored at its own declaration, not the
    # constructor's line range -- proving the two are not conflated.
    assert promoted.qualified_name == "App.Models.User.name"
    assert promoted.parent == "App.Models.User"
    assert promoted.visibility == Visibility.PRIVATE
    # Promoted properties have no declaration site of their own; they are
    # anchored at the constructor's line range.
    assert promoted.start_line == ctor.start_line
    assert promoted.end_line == ctor.end_line
    id_field = next(s for s in symbols if s.kind == SymbolKind.VARIABLE and s.name == "id")
    assert id_field.start_line != ctor.start_line


# ---------------------------------------------------------------------------
# extract_symbols: fields and constants
# ---------------------------------------------------------------------------


def test_field_visibility(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    fields = {s.name: s for s in symbols if s.kind == SymbolKind.VARIABLE}
    assert fields["id"].visibility == Visibility.PRIVATE
    assert fields["email"].visibility == Visibility.INTERNAL  # protected -> INTERNAL


def test_class_constant(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.php")
    max_len = next(s for s in symbols if s.kind == SymbolKind.CONSTANT)
    assert max_len.name == "MAX_NAME_LEN"
    assert max_len.qualified_name == "App.Models.User.MAX_NAME_LEN"
    assert max_len.visibility == Visibility.PUBLIC


def test_private_static_constant(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.php")
    default_limit = next(s for s in symbols if s.kind == SymbolKind.CONSTANT)
    assert default_limit.name == "DEFAULT_LIMIT"
    assert default_limit.visibility == Visibility.PRIVATE


def test_private_static_property(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.php")
    registry = next(s for s in symbols if s.kind == SymbolKind.VARIABLE and s.name == "registry")
    assert registry.visibility == Visibility.PRIVATE
    assert registry.parent == "App.Services.UserService"


# ---------------------------------------------------------------------------
# extract_symbols: top-level namespaced functions
# ---------------------------------------------------------------------------


def test_extract_top_level_function(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Helpers" / "functions.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Helpers/functions.php")
    format_date = next(s for s in symbols if s.name == "format_date")
    assert format_date.kind == SymbolKind.FUNCTION
    assert format_date.qualified_name == "App.Helpers.format_date"
    assert format_date.parent is None
    assert format_date.signature is not None


def test_top_level_function_with_default_param(
    engine: TreeSitterEngine, adapter: PHPAdapter
) -> None:
    source = (FIXTURES_DIR / "Helpers" / "functions.php").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Helpers/functions.php")
    slugify = next(s for s in symbols if s.name == "slugify")
    assert slugify.kind == SymbolKind.FUNCTION
    assert slugify.qualified_name == "App.Helpers.slugify"
    assert slugify.signature is not None
    assert "$separator" in slugify.signature


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_simple_import(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.php").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Models/User.php")
    modules = {i.module for i in imports}
    assert "App.Traits.HasTimestamps" in modules
    trait_import = next(i for i in imports if i.module == "App.Traits.HasTimestamps")
    assert trait_import.alias is None
    assert trait_import.is_relative is False


def test_parse_aliased_import(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Services/UserService.php")
    aliased = next(i for i in imports if i.alias == "Arr")
    assert aliased.module == "App.Contracts.Arrayable"


def test_parse_grouped_import_produces_two_statements(
    engine: TreeSitterEngine, adapter: PHPAdapter
) -> None:
    # `use App\Models\{Status, User};` must yield two independent
    # ImportStatement entries, one per name in the group.
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Services/UserService.php")
    grouped = [i for i in imports if i.module in ("App.Models.Status", "App.Models.User")]
    assert len(grouped) == 2
    assert {i.module for i in grouped} == {"App.Models.Status", "App.Models.User"}


def test_parse_use_function_import(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.php").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Services/UserService.php")
    fn_import = next(i for i in imports if i.module == "App.Helpers.format_date")
    assert fn_import.is_relative is False


def test_imports_not_relative(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "index.php").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "index.php")
    assert imports
    for imp in imports:
        assert imp.is_relative is False


def test_imports_use_dot_separators(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = (FIXTURES_DIR / "index.php").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "index.php")
    modules = {i.module for i in imports}
    assert modules == {"App.Models.Status", "App.Models.User", "App.Services.UserService"}
    for module in modules:
        assert "\\" not in module


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    source = b"<?php\n"
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "empty.php") == []
    assert adapter.parse_imports(tree, source, "empty.php") == []


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.php"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_members_have_parent(engine: TreeSitterEngine, adapter: PHPAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.php"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            if s.kind in (SymbolKind.METHOD, SymbolKind.VARIABLE, SymbolKind.CONSTANT):
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"
