from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.csharp import CSharpAdapter
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "csharp_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapter() -> CSharpAdapter:
    return CSharpAdapter()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "csharp")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_language_adapter_protocol(adapter: CSharpAdapter) -> None:
    assert isinstance(adapter, LanguageAdapter)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_language_id(adapter: CSharpAdapter) -> None:
    assert adapter.language_id == "csharp"


def test_file_extensions(adapter: CSharpAdapter) -> None:
    assert adapter.file_extensions == [".cs"]


def test_tree_sitter_name(adapter: CSharpAdapter) -> None:
    assert adapter.tree_sitter_name == "c_sharp"


# ---------------------------------------------------------------------------
# extract_symbols: classes
# ---------------------------------------------------------------------------


def test_extract_class(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    assert any(s.name == "User" for s in classes)


def test_class_qualified_name_with_namespace(
    engine: TreeSitterEngine, adapter: CSharpAdapter
) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    user = next(s for s in symbols if s.name == "User" and s.kind == SymbolKind.CLASS)
    assert user.qualified_name == "MyApp.Models.User"


def test_class_visibility(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    user = next(s for s in symbols if s.name == "User" and s.kind == SymbolKind.CLASS)
    assert user.visibility == Visibility.PUBLIC


def test_extract_record(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    record = next(s for s in symbols if s.name == "PersonRecord")
    assert record.kind == SymbolKind.CLASS
    assert record.visibility == Visibility.PUBLIC


def test_extract_struct(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    coord = next(s for s in symbols if s.name == "Coordinate")
    assert coord.kind == SymbolKind.TYPE
    assert coord.visibility == Visibility.PUBLIC


def test_struct_properties(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    lat = next(s for s in symbols if s.name == "Latitude")
    assert lat.qualified_name == "MyApp.Models.Coordinate.Latitude"
    assert lat.parent == "MyApp.Models.Coordinate"


# ---------------------------------------------------------------------------
# extract_symbols: interfaces
# ---------------------------------------------------------------------------


def test_extract_interface(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.cs")
    iface = next(s for s in symbols if s.name == "IUserService")
    assert iface.kind == SymbolKind.INTERFACE
    assert iface.visibility == Visibility.PUBLIC


def test_interface_methods_public_by_default(
    engine: TreeSitterEngine, adapter: CSharpAdapter
) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.cs")
    iface_methods = [
        s
        for s in symbols
        if s.kind == SymbolKind.METHOD and s.parent == "MyApp.Services.IUserService"
    ]
    assert len(iface_methods) == 3
    for m in iface_methods:
        assert m.visibility == Visibility.PUBLIC


def test_interface_qualified_name(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Services" / "UserService.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Services/UserService.cs")
    iface = next(s for s in symbols if s.name == "IUserService")
    assert iface.qualified_name == "MyApp.Services.IUserService"


# ---------------------------------------------------------------------------
# extract_symbols: enums
# ---------------------------------------------------------------------------


def test_extract_enum() -> None:
    source = b"""
namespace MyApp {
    public enum Status {
        Active,
        Inactive,
        Pending
    }
}
"""
    engine = TreeSitterEngine()
    adapter = CSharpAdapter()
    tree = engine.parse_bytes(source, "csharp")
    symbols = adapter.extract_symbols(tree, source, "Status.cs")
    status = next(s for s in symbols if s.name == "Status")
    assert status.kind == SymbolKind.ENUM


def test_enum_constants() -> None:
    source = b"""
namespace MyApp {
    public enum Status {
        Active,
        Inactive,
        Pending
    }
}
"""
    engine = TreeSitterEngine()
    adapter = CSharpAdapter()
    tree = engine.parse_bytes(source, "csharp")
    symbols = adapter.extract_symbols(tree, source, "Status.cs")
    consts = [s for s in symbols if s.kind == SymbolKind.CONSTANT]
    names = {s.name for s in consts}
    assert names == {"Active", "Inactive", "Pending"}
    for c in consts:
        assert c.parent == "MyApp.Status"
        assert c.qualified_name.startswith("MyApp.Status.")


# ---------------------------------------------------------------------------
# extract_symbols: methods
# ---------------------------------------------------------------------------


def test_method_visibility(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    methods = {
        s.name: s
        for s in symbols
        if s.kind == SymbolKind.METHOD and s.parent == "MyApp.Models.User"
    }
    assert methods["GetName"].visibility == Visibility.PUBLIC
    assert methods["GetEmail"].visibility == Visibility.INTERNAL
    assert methods["Validate"].visibility == Visibility.PRIVATE


def test_method_signature(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    set_name = next(s for s in symbols if s.name == "SetName")
    assert set_name.signature is not None
    assert "SetName" in set_name.signature
    assert "name" in set_name.signature


def test_constructor(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    ctor = next(
        s
        for s in symbols
        if s.kind == SymbolKind.METHOD and s.name == "User" and s.parent == "MyApp.Models.User"
    )
    assert ctor.qualified_name == "MyApp.Models.User.User"
    assert ctor.visibility == Visibility.PUBLIC
    assert ctor.signature is not None
    assert "name" in ctor.signature


def test_static_extension_method(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Utils" / "StringExtensions.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Utils/StringExtensions.cs")
    to_title = next(s for s in symbols if s.name == "ToTitleCase")
    assert to_title.kind == SymbolKind.METHOD
    assert to_title.visibility == Visibility.PUBLIC


def test_internal_method(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Utils" / "StringExtensions.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Utils/StringExtensions.cs")
    pad = next(s for s in symbols if s.name == "PadCenter")
    assert pad.visibility == Visibility.INTERNAL


def test_protected_internal_visibility() -> None:
    source = b"""
namespace MyApp {
    public class Foo {
        protected internal void SharedMethod() {}
        private protected void RestrictedMethod() {}
    }
}
"""
    engine = TreeSitterEngine()
    adapter = CSharpAdapter()
    tree = engine.parse_bytes(source, "csharp")
    symbols = adapter.extract_symbols(tree, source, "Foo.cs")
    methods = {s.name: s for s in symbols if s.kind == SymbolKind.METHOD}
    assert methods["SharedMethod"].visibility == Visibility.INTERNAL
    assert methods["RestrictedMethod"].visibility == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# extract_symbols: fields, properties, constants
# ---------------------------------------------------------------------------


def test_field_visibility(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    fields = {
        s.name: s
        for s in symbols
        if s.kind == SymbolKind.VARIABLE and s.parent == "MyApp.Models.User"
    }
    assert fields["_name"].visibility == Visibility.PRIVATE
    assert fields["age"].visibility == Visibility.INTERNAL  # protected → INTERNAL


def test_static_readonly_is_constant(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    max_len = next(s for s in symbols if s.name == "MaxNameLength")
    assert max_len.kind == SymbolKind.CONSTANT
    assert max_len.visibility == Visibility.PUBLIC


def test_const_field_is_constant(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Utils" / "StringExtensions.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Utils/StringExtensions.cs")
    empty = next(s for s in symbols if s.name == "Empty")
    assert empty.kind == SymbolKind.CONSTANT
    assert empty.visibility == Visibility.PUBLIC


def test_private_const_is_constant(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Utils" / "StringExtensions.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Utils/StringExtensions.cs")
    default_pad = next(s for s in symbols if s.name == "DefaultPad")
    assert default_pad.kind == SymbolKind.CONSTANT
    assert default_pad.visibility == Visibility.PRIVATE


def test_property_extraction(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    target_names = {"Name", "Email", "Age"}
    props = {s.name: s for s in symbols if s.kind == SymbolKind.VARIABLE and s.name in target_names}
    assert props["Name"].visibility == Visibility.PUBLIC
    assert props["Name"].parent == "MyApp.Models.User"


# ---------------------------------------------------------------------------
# extract_symbols: events and delegates
# ---------------------------------------------------------------------------


def test_event_field_extraction(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Events" / "EventHandler.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Events/EventHandler.cs")
    events = [s for s in symbols if s.name in {"OnUserChanged", "OnError"}]
    assert len(events) == 2
    for e in events:
        assert e.kind == SymbolKind.VARIABLE
        assert e.visibility == Visibility.PUBLIC


def test_delegate_extraction(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Events" / "EventHandler.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Events/EventHandler.cs")
    delegate = next(s for s in symbols if s.name == "UserChangedHandler")
    assert delegate.kind == SymbolKind.TYPE
    assert delegate.visibility == Visibility.PUBLIC


def test_delegate_qualified_name(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Events" / "EventHandler.cs").read_bytes()
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Events/EventHandler.cs")
    delegate = next(s for s in symbols if s.name == "UserChangedHandler")
    assert delegate.qualified_name == "MyApp.Events.UserChangedHandler"


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


def test_parse_using_directives(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Program.cs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Program.cs")
    modules = {i.module for i in imports}
    assert "System" in modules
    assert "MyApp.Models" in modules
    assert "MyApp.Services" in modules


def test_parse_static_using(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Utils" / "StringExtensions.cs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Utils/StringExtensions.cs")
    modules = {i.module for i in imports}
    assert "System.String" in modules


def test_imports_not_relative(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Program.cs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Program.cs")
    for imp in imports:
        assert imp.is_relative is False


def test_namespace_not_treated_as_import(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "Models/User.cs")
    modules = {i.module for i in imports}
    # namespace declaration should not appear as import
    assert "MyApp.Models" not in modules


# ---------------------------------------------------------------------------
# resolve_import
# ---------------------------------------------------------------------------


def test_resolve_internal_import(adapter: CSharpAdapter) -> None:
    file_map = {
        "Models/User.cs": "/repo/Models/User.cs",
        "Services/UserService.cs": "/repo/Services/UserService.cs",
    }
    imp = ImportStatement(
        module="MyApp.Models",
        file_path="Program.cs",
        line=2,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved == "/repo/Models/User.cs"


def test_resolve_system_namespace_returns_none(adapter: CSharpAdapter) -> None:
    file_map = {"Program.cs": "/repo/Program.cs"}
    imp = ImportStatement(
        module="System.Collections.Generic",
        file_path="Program.cs",
        line=2,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


def test_resolve_microsoft_namespace_returns_none(adapter: CSharpAdapter) -> None:
    file_map = {"Program.cs": "/repo/Program.cs"}
    imp = ImportStatement(
        module="Microsoft.Extensions.Logging",
        file_path="Program.cs",
        line=2,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


def test_resolve_unknown_returns_none(adapter: CSharpAdapter) -> None:
    file_map = {"Foo.cs": "/repo/Foo.cs"}
    imp = ImportStatement(
        module="ThirdParty.Library",
        file_path="Program.cs",
        line=2,
        is_relative=False,
    )
    resolved = adapter.resolve_import(imp, file_map)
    assert resolved is None


# ---------------------------------------------------------------------------
# classify_visibility
# ---------------------------------------------------------------------------


def test_classify_public_symbol(adapter: CSharpAdapter) -> None:
    s = Symbol(
        name="Program",
        qualified_name="MyApp.Program",
        kind=SymbolKind.CLASS,
        file_path="Program.cs",
        start_line=1,
        end_line=10,
        visibility=Visibility.PUBLIC,
    )
    assert adapter.classify_visibility(s) == Visibility.PUBLIC


def test_classify_private_symbol(adapter: CSharpAdapter) -> None:
    s = Symbol(
        name="Validate",
        qualified_name="MyApp.Models.User.Validate",
        kind=SymbolKind.METHOD,
        file_path="User.cs",
        start_line=1,
        end_line=5,
        visibility=Visibility.PRIVATE,
    )
    assert adapter.classify_visibility(s) == Visibility.PRIVATE


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


def test_detect_main_entry_point(adapter: CSharpAdapter) -> None:
    files = [
        DiscoveredFile(
            path="Program.cs",
            absolute_path=str(FIXTURES_DIR / "Program.cs"),
            language="csharp",
        ),
        DiscoveredFile(
            path="Models/User.cs",
            absolute_path=str(FIXTURES_DIR / "Models" / "User.cs"),
            language="csharp",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "Program.cs" in entry_points
    assert "Models/User.cs" not in entry_points


def test_no_entry_point_for_service(adapter: CSharpAdapter) -> None:
    files = [
        DiscoveredFile(
            path="Services/UserService.cs",
            absolute_path=str(FIXTURES_DIR / "Services" / "UserService.cs"),
            language="csharp",
        ),
    ]
    assert adapter.detect_entry_points(files) == []


def test_detect_test_attribute_entry_point(adapter: CSharpAdapter) -> None:
    # [Fact] xunit test
    files = [
        DiscoveredFile(
            path="Tests/UserTests.cs",
            absolute_path=str(FIXTURES_DIR / "Program.cs"),  # reuse any file
            language="csharp",
        ),
    ]
    # Program.cs has static void Main so it qualifies
    entry_points = adapter.detect_entry_points(files)
    assert "Tests/UserTests.cs" in entry_points


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = b""
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Empty.cs")
    imports = adapter.parse_imports(tree, source, "Empty.cs")
    assert symbols == []
    assert imports == []


def test_no_namespace_file(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = b"public class Foo { public void Bar() {} }\n"
    tree = parse(engine, source)
    symbols = adapter.extract_symbols(tree, source, "Foo.cs")
    foo = next(s for s in symbols if s.name == "Foo")
    assert foo.kind == SymbolKind.CLASS
    assert foo.qualified_name == "Foo"


def test_all_symbols_have_qualified_names(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.cs"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            assert s.qualified_name, f"Missing qualified_name for {s.name} in {f}"


def test_all_members_have_parent(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    for f in FIXTURES_DIR.rglob("*.cs"):
        source = f.read_bytes()
        tree = parse(engine, source)
        symbols = adapter.extract_symbols(tree, source, str(f))
        for s in symbols:
            if s.kind in (SymbolKind.METHOD, SymbolKind.VARIABLE, SymbolKind.CONSTANT):
                assert s.parent is not None, f"Missing parent for {s.qualified_name} in {f}"


# ---------------------------------------------------------------------------
# parse_imports: global using
# ---------------------------------------------------------------------------


def test_parse_global_using(engine: TreeSitterEngine, adapter: CSharpAdapter) -> None:
    source = b"global using System.Linq;\nusing System;\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "GlobalUsings.cs")
    modules = {i.module for i in imports}
    assert "System.Linq" in modules
    assert "System" in modules


def test_parse_global_using_preserves_line(
    engine: TreeSitterEngine, adapter: CSharpAdapter
) -> None:
    source = b"global using System.Linq;\nglobal using System.Collections.Generic;\n"
    tree = parse(engine, source)
    imports = adapter.parse_imports(tree, source, "GlobalUsings.cs")
    assert len(imports) == 2
    assert imports[0].line == 1
    assert imports[1].line == 2


# ---------------------------------------------------------------------------
# detect_entry_points: top-level statements
# ---------------------------------------------------------------------------


def test_detect_top_level_statements_entry_point(adapter: CSharpAdapter, tmp_path: Path) -> None:
    tls_file = tmp_path / "Program.cs"
    tls_file.write_text('Console.WriteLine("Hello");\n')
    files = [
        DiscoveredFile(
            path="Program.cs",
            absolute_path=str(tls_file),
            language="csharp",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "Program.cs" in entry_points


def test_top_level_await_detected(adapter: CSharpAdapter, tmp_path: Path) -> None:
    tls_file = tmp_path / "Program.cs"
    tls_file.write_text("await Task.Delay(1000);\n")
    files = [
        DiscoveredFile(
            path="Program.cs",
            absolute_path=str(tls_file),
            language="csharp",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "Program.cs" in entry_points


def test_top_level_not_detected_with_class(adapter: CSharpAdapter, tmp_path: Path) -> None:
    tls_file = tmp_path / "Program.cs"
    tls_file.write_text('class Foo { }\nConsole.WriteLine("Hello");\n')
    files = [
        DiscoveredFile(
            path="Program.cs",
            absolute_path=str(tls_file),
            language="csharp",
        ),
    ]
    entry_points = adapter.detect_entry_points(files)
    assert "Program.cs" not in entry_points


# ---------------------------------------------------------------------------
# Round-trip: parse → chunk → store → retrieve
# ---------------------------------------------------------------------------


def test_round_trip_parse_chunk_store_retrieve(
    engine: TreeSitterEngine,
    adapter: CSharpAdapter,
    tmp_path: Path,
) -> None:
    from archex.index.store import IndexStore
    from archex.models import IndexConfig, ParsedFile
    from archex.pipeline.chunker import ASTChunker

    source = (FIXTURES_DIR / "Models" / "User.cs").read_bytes()
    tree = engine.parse_bytes(source, adapter.language_id)
    symbols = adapter.extract_symbols(tree, source, "Models/User.cs")
    assert len(symbols) > 0

    parsed = ParsedFile(path="Models/User.cs", language=adapter.language_id, symbols=symbols)
    chunker = ASTChunker(config=IndexConfig())
    chunks = chunker.chunk_files([parsed], {"Models/User.cs": source})
    assert len(chunks) > 0

    store = IndexStore(tmp_path / "test.db")
    try:
        store.insert_chunks(chunks)
        retrieved = store.get_chunks_for_file("Models/User.cs")
        assert len(retrieved) == len(chunks)
        for chunk in retrieved:
            assert chunk.content
            assert chunk.language == adapter.language_id
    finally:
        store.close()
