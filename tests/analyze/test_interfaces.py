"""Tests for analyze/interfaces.py: interface surface extraction."""

from __future__ import annotations

from pathlib import Path

from archex.analyze.interfaces import (
    _parse_parameters,  # pyright: ignore[reportPrivateUsage]
    extract_interfaces,
)
from archex.index.graph import DependencyGraph
from archex.models import SymbolKind
from archex.parse import (
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.parse.adapters.python import PythonAdapter

SIMPLE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "python_simple"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_parsed_and_graph(repo_path: Path):
    from archex.acquire import discover_files

    files = discover_files(repo_path, languages=["python"])
    engine = TreeSitterEngine()
    adapters = {"python": PythonAdapter()}

    parsed_files = extract_symbols(files, engine, adapters)
    import_map = parse_imports(files, engine, adapters)
    file_map = build_file_map(files)
    file_languages = {f.path: f.language for f in files}
    resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

    graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)
    return parsed_files, graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_interfaces_returns_interfaces(python_simple_repo: Path) -> None:
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    assert len(interfaces) > 0


def test_extract_interfaces_only_top_level(python_simple_repo: Path) -> None:
    """Methods (symbols with a parent) must not appear as interfaces."""
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    for iface in interfaces:
        # Methods have a parent; top-level functions/classes do not
        assert iface.symbol.kind != SymbolKind.METHOD, (
            f"Method '{iface.symbol.name}' should not be in interface surface"
        )


def test_extract_interfaces_no_private_symbols(python_simple_repo: Path) -> None:
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    for iface in interfaces:
        name = iface.symbol.name
        is_dunder = name.startswith("__") and name.endswith("__")
        is_private = name.startswith("_") and not is_dunder
        assert not is_private, f"Private symbol '{name}' should not be in interface surface"


def test_extract_interfaces_contains_public_functions(python_simple_repo: Path) -> None:
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    # python_simple has: run() in main.py, hash_password() and validate_email() in utils.py
    function_names = {i.symbol.name for i in interfaces if i.symbol.kind == SymbolKind.FUNCTION}
    assert "validate_email" in function_names
    assert "hash_password" in function_names


def test_extract_interfaces_contains_public_classes(python_simple_repo: Path) -> None:
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    class_names = {i.symbol.name for i in interfaces if i.symbol.kind == SymbolKind.CLASS}
    # python_simple has: Role, User, AuthService
    assert "User" in class_names
    assert "AuthService" in class_names


def test_extract_interfaces_used_by_populated(python_simple_repo: Path) -> None:
    """Files that import another file should appear in that file's used_by."""
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)

    # Build a map: file_path -> used_by lists from all interfaces in that file
    file_used_by: dict[str, set[str]] = {}
    for iface in interfaces:
        fp = iface.symbol.file_path
        file_used_by.setdefault(fp, set()).update(iface.used_by)

    # At least one file should have non-empty used_by (some file is imported)
    any_used_by = any(len(v) > 0 for v in file_used_by.values())
    assert any_used_by, "No interfaces have used_by populated despite graph edges existing"


def test_extract_interfaces_signature_not_empty(python_simple_repo: Path) -> None:
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    for iface in interfaces:
        assert iface.signature, f"Interface '{iface.symbol.name}' has empty signature"


def test_extract_interfaces_function_has_parameters(python_simple_repo: Path) -> None:
    """validate_email(email: str) should have one parsed parameter."""
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    validate = next((i for i in interfaces if i.symbol.name == "validate_email"), None)
    assert validate is not None, "validate_email not found in interfaces"
    assert len(validate.parameters) == 1
    assert validate.parameters[0].name == "email"


def test_extract_interfaces_return_type_parsed(python_simple_repo: Path) -> None:
    """validate_email has return type bool."""
    parsed_files, graph = _build_parsed_and_graph(python_simple_repo)
    interfaces = extract_interfaces(parsed_files, graph)
    validate = next((i for i in interfaces if i.symbol.name == "validate_email"), None)
    assert validate is not None
    assert validate.return_type == "bool"


def test_extract_interfaces_empty_input() -> None:
    graph = DependencyGraph()
    result = extract_interfaces([], graph)
    assert result == []


# ---------------------------------------------------------------------------
# _parse_parameters edge-case tests
# ---------------------------------------------------------------------------


def test_parse_parameters_with_defaults() -> None:
    params = _parse_parameters("def foo(a: int, b: str = 'hello')")
    assert len(params) == 2
    a = next(p for p in params if p.name == "a")
    b = next(p for p in params if p.name == "b")
    assert a.required is True
    assert a.default is None
    assert b.required is False
    assert b.default == "'hello'"


def test_parse_parameters_self_cls_skipped() -> None:
    params = _parse_parameters("def method(self, x: int)")
    assert len(params) == 1
    assert params[0].name == "x"

    params_cls = _parse_parameters("def method(cls, y: str)")
    assert len(params_cls) == 1
    assert params_cls[0].name == "y"


def test_parse_parameters_no_parens() -> None:
    params = _parse_parameters("some_var")
    assert params == []


def test_parse_parameters_empty_parens() -> None:
    params = _parse_parameters("def foo()")
    assert params == []


def test_parse_parameters_nested_generics() -> None:
    # The depth-tracking splitter correctly isolates each param even when the
    # annotation contains nested brackets with inner commas.  The _PARAM_RE
    # annotation pattern excludes literal commas, so a: dict[str, list[int]]
    # fails the regex and is skipped; b: int is matched normally.
    params = _parse_parameters("def foo(a: dict[str, list[int]], b: int)")
    # b is always parsed
    b = next((p for p in params if p.name == "b"), None)
    assert b is not None
    assert b.type_annotation == "int"
    assert b.required is True
    # a is dropped by the regex due to the comma inside the annotation
    a = next((p for p in params if p.name == "a"), None)
    assert a is None


def test_parse_parameters_var_positional() -> None:
    params = _parse_parameters("def foo(*args: int)")
    assert len(params) == 1
    assert params[0].name == "args"
    assert params[0].required is False


def test_parse_parameters_var_keyword() -> None:
    params = _parse_parameters("def foo(**kwargs: str)")
    assert len(params) == 1
    assert params[0].name == "kwargs"
    assert params[0].required is False


def test_extract_interfaces_symbol_no_signature() -> None:
    from archex.models import ParsedFile, Symbol, SymbolKind, Visibility

    sym = Symbol(
        name="bare_func",
        qualified_name="bare_func",
        kind=SymbolKind.FUNCTION,
        file_path="fake.py",
        start_line=1,
        end_line=1,
        visibility=Visibility.PUBLIC,
        signature=None,
    )
    pf = ParsedFile(path="fake.py", language="python", symbols=[sym])
    graph = DependencyGraph()
    interfaces = extract_interfaces([pf], graph)
    assert len(interfaces) == 1
    iface = interfaces[0]
    assert iface.parameters == []
    assert iface.return_type is None
