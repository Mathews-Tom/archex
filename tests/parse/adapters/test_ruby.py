from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.ruby import RubyAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "ruby_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> RubyAdapter:
    return RubyAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "ruby")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: RubyAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: RubyAdapter) -> None:
    assert adapter.language_id == "ruby"


def test_file_extensions(adapter: RubyAdapter) -> None:
    assert adapter.file_extensions == [".rb"]


def test_tree_sitter_name(adapter: RubyAdapter) -> None:
    assert adapter.tree_sitter_name == "ruby"


# ---------------------------------------------------------------------------
# extract_symbols: module/class qualification with '.' separators
# ---------------------------------------------------------------------------


def test_nested_module_qualifies_class(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/user.rb")

    user = next(s for s in symbols if s.name == "User")
    assert user.kind == SymbolKind.CLASS
    assert user.qualified_name == "StoreFront.Models.User"
    assert user.parent == "StoreFront.Models"


def test_deeply_nested_modules_use_dot_separators(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "mixins" / "trackable.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "mixins/trackable.rb")

    class_methods = next(s for s in symbols if s.name == "ClassMethods")
    assert class_methods.qualified_name == "StoreFront.Mixins.Trackable.ClassMethods"

    tracked_events = next(s for s in symbols if s.name == "tracked_events")
    assert (
        tracked_events.qualified_name == "StoreFront.Mixins.Trackable.ClassMethods.tracked_events"
    )
    assert "::" not in tracked_events.qualified_name


# ---------------------------------------------------------------------------
# extract_symbols: modules
# ---------------------------------------------------------------------------


def test_module_extracted_as_module_kind(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "support" / "slugger.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "support/slugger.rb")

    slugger = next(s for s in symbols if s.name == "Slugger")
    assert slugger.kind == SymbolKind.MODULE
    assert slugger.visibility == Visibility.PUBLIC


def test_top_level_module_has_no_parent(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "support" / "slugger.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "support/slugger.rb")

    top = next(s for s in symbols if s.name == "StoreFront")
    assert top.parent is None


def test_sibling_classes_share_module_parent(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "legacy" / "admin.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "legacy/admin.rb")

    classes = {s.name: s for s in symbols if s.kind == SymbolKind.CLASS}
    assert classes["Admin"].parent == "StoreFront.Legacy"
    assert classes["Guest"].parent == "StoreFront.Legacy"
    assert classes["Admin"].qualified_name == "StoreFront.Legacy.Admin"
    assert classes["Guest"].qualified_name == "StoreFront.Legacy.Guest"


# ---------------------------------------------------------------------------
# extract_symbols: instance, module, and singleton methods
# ---------------------------------------------------------------------------


def test_instance_methods_default_to_public(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "services" / "user_service.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "services/user_service.rb")

    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["initialize"].visibility == Visibility.PUBLIC
    assert methods["register"].visibility == Visibility.PUBLIC
    assert methods["serialize"].visibility == Visibility.PUBLIC
    for m in methods.values():
        assert m.parent == "StoreFront.Services.UserService"


def test_protected_marker_maps_to_internal(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/user.rb")

    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["normalized_role"].visibility == Visibility.INTERNAL  # protected -> INTERNAL


def test_private_marker_applies_only_to_subsequent_methods(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/user.rb")

    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["normalize_email"].visibility == Visibility.PRIVATE
    # Declared before the `private` marker -- must stay public.
    assert methods["display_name"].visibility == Visibility.PUBLIC


def test_explicit_public_marker_resets_visibility_after_private(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    # Ruby lets a bare `public` call re-open public visibility after a
    # `private` marker; the adapter's per-body visibility state must track
    # that toggle rather than sticking once it flips to private.
    source = b"public\ndef a; end\nprivate\ndef b; end\npublic\ndef c; end\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "vis.rb")

    methods = {s.name: s for s in symbols}
    assert methods["a"].visibility == Visibility.PUBLIC
    assert methods["b"].visibility == Visibility.PRIVATE
    assert methods["c"].visibility == Visibility.PUBLIC


def test_visibility_call_with_symbol_arguments_updates_existing_methods(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (
        b"class Foo\n"
        b"  def hidden; end\n"
        b"  def semi_hidden; end\n"
        b"  private :hidden\n"
        b"  protected :semi_hidden\n"
        b"end\n"
    )
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "foo.rb")

    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["hidden"].visibility == Visibility.PRIVATE
    assert methods["semi_hidden"].visibility == Visibility.INTERNAL


def test_visibility_call_wrapping_method_extracts_method(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = b"class Foo\n  private def secret(x)\n    x\n  end\nend\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "foo.rb")

    secret = next(s for s in symbols if s.name == "secret")
    assert secret.qualified_name == "Foo.secret"
    assert secret.visibility == Visibility.PRIVATE
    assert secret.parent == "Foo"


def test_visibility_call_mixes_symbol_arguments_and_wrapped_method(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = b"class Foo\n  def hidden; end\n  private :hidden, def secret(x)\n    x\n  end\nend\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "foo.rb")

    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["hidden"].visibility == Visibility.PRIVATE
    assert methods["secret"].visibility == Visibility.PRIVATE


def test_empty_visibility_call_resets_subsequent_method_visibility(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = b"class Foo\nprivate()\ndef hidden; end\npublic()\ndef shown; end\nend\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "foo.rb")

    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["hidden"].visibility == Visibility.PRIVATE
    assert methods["shown"].visibility == Visibility.PUBLIC


def test_singleton_method_with_self_receiver_stays_nested(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/user.rb")

    finder = next(s for s in symbols if s.name == "find_by_email")
    assert finder.qualified_name == "StoreFront.Models.User.find_by_email"
    assert finder.parent == "StoreFront.Models.User"
    assert finder.signature is not None
    assert "self" in finder.signature


def test_singleton_method_with_explicit_constant_receiver_reparents(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    # `def Helper.bar` inside module M rebinds the method's parent to the
    # named receiver instead of the lexically enclosing module -- this is
    # what lets a mixin attach "class methods" to whatever module/class
    # extends it, decoupled from where the def itself is written.
    source = b"module M\n  def Helper.bar(x)\n  end\nend\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "reparent.rb")

    bar = next(s for s in symbols if s.name == "bar")
    assert bar.qualified_name == "Helper.bar"
    assert bar.parent == "Helper"


def test_module_function_via_self_receiver(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "support" / "slugger.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "support/slugger.rb")

    slugify = next(s for s in symbols if s.name == "slugify")
    assert slugify.kind == SymbolKind.METHOD
    assert slugify.qualified_name == "StoreFront.Slugger.slugify"
    assert slugify.parent == "StoreFront.Slugger"


def test_singleton_class_methods_stay_nested(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (
        b"class Widget\n"
        b"  class << self\n"
        b"    def build(name)\n"
        b"      new(name)\n"
        b"    end\n"
        b"  end\n"
        b"end\n"
    )
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "widget.rb")

    build = next(s for s in symbols if s.name == "build")
    assert build.qualified_name == "Widget.build"
    assert build.parent == "Widget"
    assert build.signature == "def self.build(name)"


def test_singleton_class_with_constant_receiver_reparents_methods(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (
        b"class Widget\n"
        b"end\n"
        b"class Gadget\n"
        b"  class << Widget\n"
        b"    def build(name)\n"
        b"      name\n"
        b"    end\n"
        b"  end\n"
        b"end\n"
    )
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "widget.rb")

    build = next(s for s in symbols if s.name == "build")
    assert build.qualified_name == "Widget.build"
    assert build.parent == "Widget"
    assert build.signature == "def self.build(name)"


def test_bang_method_name_preserved(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "store_front" / "auditable.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "store_front/auditable.rb")

    audit = next(s for s in symbols if s.name == "audit!")
    assert audit.qualified_name == "StoreFront.Auditable.audit!"


# ---------------------------------------------------------------------------
# extract_symbols: constants
# ---------------------------------------------------------------------------


def test_class_constant(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/user.rb")

    default_role = next(s for s in symbols if s.name == "DEFAULT_ROLE")
    assert default_role.kind == SymbolKind.CONSTANT
    assert default_role.qualified_name == "StoreFront.Models.User.DEFAULT_ROLE"
    assert default_role.parent == "StoreFront.Models.User"
    assert default_role.visibility == Visibility.PUBLIC


def test_module_level_constant(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "support" / "slugger.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "support/slugger.rb")

    separator = next(s for s in symbols if s.name == "SEPARATOR")
    assert separator.kind == SymbolKind.CONSTANT
    assert separator.parent == "StoreFront.Slugger"


def test_attr_reader_produces_no_symbols(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    # `attr_reader :email, :role` is a method call, not an assignment or a
    # `def` -- it must not be mistaken for a field/constant declaration.
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "models/user.rb")

    names = {s.name for s in symbols}
    assert "email" not in names
    assert "role" not in names
    assert "attr_reader" not in names


# ---------------------------------------------------------------------------
# extract_symbols: top-level scripts without declarations
# ---------------------------------------------------------------------------


def test_top_level_script_without_declarations_has_no_symbols(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    # app.rb only contains `require`/`require_relative` calls and a local
    # variable assignment at the top level -- none of those are declarations
    # and must not be misreported as symbols.
    source = (FIXTURES_DIR / "app.rb").read_bytes()
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "app.rb") == []


# ---------------------------------------------------------------------------
# parse_imports: require / require_relative
# ---------------------------------------------------------------------------


def test_parse_require_is_not_relative(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "app.rb").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.rb")

    json_import = next(i for i in imports if i.module == "json")
    assert json_import.is_relative is False
    assert json_import.line == 3


def test_parse_require_relative_is_relative(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = (FIXTURES_DIR / "app.rb").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.rb")

    modules = {i.module: i for i in imports}
    assert modules["store_front/auditable"].is_relative is True
    assert modules["services/user_service"].is_relative is True


def test_parse_imports_excludes_non_require_calls(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    # app.rb also calls `StoreFront::Services::UserService.new`,
    # `JSON.generate`, and `puts` -- none of those are require calls and
    # must not leak into the import list.
    source = (FIXTURES_DIR / "app.rb").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "app.rb")

    assert len(imports) == 5
    assert {i.module for i in imports} == {
        "json",
        "store_front/auditable",
        "support/slugger",
        "models/user",
        "services/user_service",
    }


def test_parse_require_relative_preserves_parent_dir_prefix(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "models" / "user.rb").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "models/user.rb")

    modules = {i.module for i in imports}
    assert "../store_front/auditable" in modules
    assert "../support/slugger" in modules
    for i in imports:
        assert i.is_relative is True


def test_parse_require_without_relative_flag(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "services" / "user_service.rb").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "services/user_service.rb")

    set_import = next(i for i in imports if i.module == "set")
    assert set_import.is_relative is False


def test_file_with_no_requires_has_no_imports(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = (FIXTURES_DIR / "legacy" / "admin.rb").read_bytes()
    tree = parse(engine, source)
    assert adapter.parse_imports(tree, source, "legacy/admin.rb") == []


def test_interpolated_require_string_falls_back_to_literal_suffix(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    # A dynamically interpolated require target has no single literal module
    # name. The adapter walks the string for the first plain string_content
    # segment rather than crashing or silently dropping the call -- pin the
    # documented (if imperfect) result so a regression is caught either way.
    source = b'require_relative "#{dir}/foo"\n'
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "dyn.rb")

    assert len(imports) == 1
    assert imports[0].module == "/foo"
    assert imports[0].is_relative is True


def test_require_call_without_arguments_produces_no_import(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    source = b"require()\n"
    tree = parse(engine, source)
    assert adapter.parse_imports(tree, source, "bare.rb") == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    assert adapter.extract_symbols(tree, source, "empty.rb") == []
    assert adapter.parse_imports(tree, source, "empty.rb") == []


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: RubyAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.rb"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_methods_and_constants_have_parent(
    engine: TreeSitterEngine, adapter: RubyAdapter
) -> None:
    for f in FIXTURES_DIR.rglob("*.rb"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            if s.kind in (SymbolKind.METHOD, SymbolKind.CONSTANT):
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"
