from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from archex.models import DiscoveredFile, SymbolKind
from archex.parse.adapters import get_adapter
from archex.parse.engine import TreeSitterEngine
from archex.parse.symbols import extract_symbols

if TYPE_CHECKING:
    from archex.parse.adapters.base import LanguageAdapter

FIXTURE_DIR = "tests/fixtures/python_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapters(engine: TreeSitterEngine) -> dict[str, LanguageAdapter]:
    adapter = get_adapter("python", engine)
    assert adapter is not None
    return {"python": adapter}


@pytest.fixture()
def python_simple_files() -> list[DiscoveredFile]:
    """All python_simple fixture files as DiscoveredFile instances."""
    files: list[DiscoveredFile] = []
    for rel_path in [
        "main.py",
        "models.py",
        "utils.py",
        "services/__init__.py",
        "services/auth.py",
    ]:
        abs_path = os.path.join(FIXTURE_DIR, rel_path)
        files.append(
            DiscoveredFile(
                path=rel_path,
                absolute_path=abs_path,
                language="python",
            )
        )
    return files


def test_extract_symbols_returns_parsed_files(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    assert len(parsed) == len(python_simple_files)


def test_all_files_have_correct_language(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    for pf in parsed:
        assert pf.language == "python"


def test_models_py_has_symbols(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    models_file = next(pf for pf in parsed if pf.path == "models.py")
    class_names = {s.name for s in models_file.symbols if s.kind == SymbolKind.CLASS}
    assert "Role" in class_names
    assert "User" in class_names


def test_utils_py_has_functions(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    utils_file = next(pf for pf in parsed if pf.path == "utils.py")
    func_names = {s.name for s in utils_file.symbols if s.kind == SymbolKind.FUNCTION}
    assert "hash_password" in func_names
    assert "validate_email" in func_names


def test_auth_py_has_methods(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    auth_file = next(pf for pf in parsed if pf.path == "services/auth.py")
    method_qnames = {s.qualified_name for s in auth_file.symbols if s.kind == SymbolKind.METHOD}
    assert "AuthService.login" in method_qnames
    assert "AuthService.logout" in method_qnames
    assert "AuthService.verify_token" in method_qnames


def test_line_counts_are_positive(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    for pf in parsed:
        assert pf.lines > 0, f"{pf.path} has zero line count"


def test_skips_files_with_no_adapter(
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    files = [
        DiscoveredFile(
            path="main.go",
            absolute_path="/nonexistent/main.go",
            language="go",
        )
    ]
    parsed = extract_symbols(files, engine, adapters)
    assert parsed == []


def test_main_py_has_run_function(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    parsed = extract_symbols(python_simple_files, engine, adapters)
    main_file = next(pf for pf in parsed if pf.path == "main.py")
    func_names = {s.name for s in main_file.symbols if s.kind == SymbolKind.FUNCTION}
    assert "run" in func_names


def test_parse_file_worker_logs_warning_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    """_parse_file_worker logs a warning and returns None when parsing fails."""
    import logging

    from archex.parse.symbols import _parse_file_worker  # pyright: ignore[reportPrivateUsage]

    with caplog.at_level(logging.WARNING, logger="archex.parse.symbols"):
        result = _parse_file_worker("/nonexistent/bad.py", "bad.py", "python")

    assert result is None
    assert any("Failed to parse" in r.message for r in caplog.records)
    assert any("bad.py" in r.message for r in caplog.records)


def test_parse_file_worker_returns_none_on_missing_file() -> None:
    """_parse_file_worker returns None (not raises) for a missing file."""
    from archex.parse.symbols import _parse_file_worker  # pyright: ignore[reportPrivateUsage]

    result = _parse_file_worker("/nonexistent/ghost.py", "ghost.py", "python")
    assert result is None
