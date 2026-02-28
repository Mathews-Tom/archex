from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from archex.models import DiscoveredFile
from archex.parse.adapters import get_adapter
from archex.parse.engine import TreeSitterEngine
from archex.parse.imports import build_file_map, parse_imports, resolve_imports

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


# --- build_file_map ---


def test_build_file_map_simple(python_simple_files: list[DiscoveredFile]) -> None:
    file_map = build_file_map(python_simple_files)
    assert "models" in file_map
    assert file_map["models"] == "models.py"
    assert "utils" in file_map
    assert file_map["utils"] == "utils.py"


def test_build_file_map_nested_module(python_simple_files: list[DiscoveredFile]) -> None:
    file_map = build_file_map(python_simple_files)
    assert "services.auth" in file_map
    assert file_map["services.auth"] == "services/auth.py"


def test_build_file_map_package_init(python_simple_files: list[DiscoveredFile]) -> None:
    file_map = build_file_map(python_simple_files)
    assert "services" in file_map
    assert file_map["services"] == "services/__init__.py"


# --- parse_imports ---


def test_parse_imports_returns_all_files(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    # All Python files should be in the result
    for f in python_simple_files:
        assert f.path in import_map


def test_main_py_import_count(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    main_imports = import_map["main.py"]
    # main.py has: from models import Role, User; from services.auth import AuthService;
    #              from utils import validate_email
    assert len(main_imports) >= 3


def test_utils_py_has_stdlib_imports(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    utils_modules = {imp.module for imp in import_map["utils.py"]}
    assert "hashlib" in utils_modules
    assert "re" in utils_modules


def test_auth_py_has_utils_import(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    auth_modules = {imp.module for imp in import_map["services/auth.py"]}
    assert "utils" in auth_modules


# --- resolve_imports ---


def test_resolve_internal_imports(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    file_map = build_file_map(python_simple_files)
    file_languages = {f.path: f.language for f in python_simple_files}
    resolve_imports(import_map, file_map, adapters, file_languages)

    main_imports = import_map["main.py"]
    # "from models import Role, User" should resolve to models.py
    models_imp = next((i for i in main_imports if i.module == "models"), None)
    assert models_imp is not None
    assert models_imp.resolved_path == "models.py"


def test_resolve_external_imports_are_none(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    file_map = build_file_map(python_simple_files)
    file_languages = {f.path: f.language for f in python_simple_files}
    resolve_imports(import_map, file_map, adapters, file_languages)

    # hashlib and re in utils.py are external
    utils_imports = import_map["utils.py"]
    external_modules = {"hashlib", "re"}
    for imp in utils_imports:
        if imp.module in external_modules:
            assert imp.resolved_path is None, (
                f"Expected {imp.module} to be external (None), got {imp.resolved_path}"
            )


def test_resolve_nested_module(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    import_map = parse_imports(python_simple_files, engine, adapters)
    file_map = build_file_map(python_simple_files)
    file_languages = {f.path: f.language for f in python_simple_files}
    resolve_imports(import_map, file_map, adapters, file_languages)

    main_imports = import_map["main.py"]
    auth_imp = next((i for i in main_imports if "auth" in i.module or "services" in i.module), None)
    assert auth_imp is not None
    assert auth_imp.resolved_path == "services/auth.py"


def test_parse_imports_worker_logs_warning_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    """_parse_imports_worker logs a warning and returns None when parsing fails."""
    import logging

    from archex.parse.imports import _parse_imports_worker  # pyright: ignore[reportPrivateUsage]

    with caplog.at_level(logging.WARNING, logger="archex.parse.imports"):
        result = _parse_imports_worker("/nonexistent/bad.py", "bad.py", "python")

    assert result is None
    assert any("Failed to parse" in r.message for r in caplog.records)
    assert any("bad.py" in r.message for r in caplog.records)


def test_parse_imports_worker_returns_none_on_missing_file() -> None:
    """_parse_imports_worker returns None (not raises) for a missing file."""
    from archex.parse.imports import _parse_imports_worker  # pyright: ignore[reportPrivateUsage]

    result = _parse_imports_worker("/nonexistent/ghost.py", "ghost.py", "python")
    assert result is None
