from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from archex.models import DiscoveredFile
from archex.parse.adapters import default_adapter_registry
from archex.parse.engine import TreeSitterEngine
from archex.parse.imports import build_file_map, parse_imports, resolve_imports

if TYPE_CHECKING:
    from pathlib import Path

    from archex.parse.adapters.base import LanguageAdapter

FIXTURE_DIR = "tests/fixtures/python_simple"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture()
def adapters() -> dict[str, LanguageAdapter]:
    cls = default_adapter_registry.get("python")
    assert cls is not None
    return {"python": cls()}


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


def test_parse_imports_worker_raises_on_missing_file() -> None:
    """_parse_imports_worker propagates ParseError for a missing file."""
    from archex.exceptions import ParseError
    from archex.parse.imports import _parse_imports_worker  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ParseError):
        _parse_imports_worker("/nonexistent/ghost.py", "ghost.py", "python")


# --- skip / continue branches ---


def test_parse_imports_skips_unknown_language(engine: TreeSitterEngine) -> None:
    """parse_imports skips files whose language has no matching adapter."""
    files = [
        DiscoveredFile(
            path="test.xyz",
            absolute_path="/tmp/test.xyz",
            language="unknown_lang",
        )
    ]
    result = parse_imports(files, engine, {})
    assert result == {}


def test_resolve_imports_skips_missing_language() -> None:
    """resolve_imports skips files not present in file_languages."""
    from archex.models import ImportStatement

    import_map: dict[str, list[ImportStatement]] = {
        "unknown.py": [ImportStatement(module="foo", file_path="unknown.py", line=1)],
    }
    # file_languages is empty — "unknown.py" has no language entry
    result = resolve_imports(import_map, {}, {}, {})
    assert result["unknown.py"][0].resolved_path is None


def test_resolve_imports_skips_missing_adapter() -> None:
    """resolve_imports skips files whose language has no adapter."""
    from archex.models import ImportStatement

    import_map: dict[str, list[ImportStatement]] = {
        "test.rb": [ImportStatement(module="foo", file_path="test.rb", line=1)],
    }
    file_languages = {"test.rb": "ruby"}
    # adapters does not contain "ruby"
    result = resolve_imports(import_map, {}, {}, file_languages)
    assert result["test.rb"][0].resolved_path is None


# --- _parse_imports_worker direct tests ---


def test_parse_imports_worker_unsupported_language() -> None:
    """_parse_imports_worker returns None for a language with no adapter."""
    from archex.parse.imports import _parse_imports_worker  # pyright: ignore[reportPrivateUsage]

    result = _parse_imports_worker("/fake/file.xyz", "file.xyz", "brainfuck")
    assert result is None


def test_parse_imports_worker_success(tmp_path: Path) -> None:
    """_parse_imports_worker reads a real file and returns parsed imports."""
    from archex.parse.imports import _parse_imports_worker  # pyright: ignore[reportPrivateUsage]

    py_file = tmp_path / "sample.py"
    py_file.write_text("import os\nimport sys\n")
    result = _parse_imports_worker(str(py_file), "sample.py", "python")
    assert result is not None
    path, imports = result
    assert path == "sample.py"
    assert len(imports) >= 2


# --- parallel path tests ---


def test_parse_imports_parallel_path(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """parse_imports uses ProcessPoolExecutor when parallel=True and >10 files."""
    files: list[DiscoveredFile] = []
    for i in range(12):
        f = tmp_path / f"mod_{i}.py"
        f.write_text("import os\n")
        files.append(
            DiscoveredFile(
                path=f"mod_{i}.py",
                absolute_path=str(f),
                language="python",
            )
        )
    result = parse_imports(files, engine, adapters, parallel=True)
    assert len(result) == 12
    for path, imports in result.items():
        assert path.startswith("mod_")
        assert len(imports) >= 1


def test_parse_imports_parallel_fallback_on_error(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """parse_imports falls back to sequential when ProcessPoolExecutor raises."""
    files: list[DiscoveredFile] = []
    for i in range(12):
        f = tmp_path / f"mod_{i}.py"
        f.write_text("import os\n")
        files.append(
            DiscoveredFile(
                path=f"mod_{i}.py",
                absolute_path=str(f),
                language="python",
            )
        )
    with patch("archex.parse.imports.ProcessPoolExecutor", side_effect=RuntimeError("pool failed")):
        result = parse_imports(files, engine, adapters, parallel=True)
    assert len(result) == 12


def test_build_file_map_non_python_extension() -> None:
    """build_file_map uses path-without-extension as key for non-.py files."""
    files = [
        DiscoveredFile(
            path="src/utils.ts",
            absolute_path="/tmp/src/utils.ts",
            language="typescript",
        ),
        DiscoveredFile(
            path="cmd/main.go",
            absolute_path="/tmp/cmd/main.go",
            language="go",
        ),
    ]
    result = build_file_map(files)
    assert result["src.utils"] == "src/utils.ts"
    assert result["cmd.main"] == "cmd/main.go"


def test_strict_parallel_raises_on_bad_file(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """parse_imports raises ParseError when strict=True and a file fails in parallel mode."""
    from archex.exceptions import ParseError

    files: list[DiscoveredFile] = []
    for i in range(11):
        f = tmp_path / f"good_{i}.py"
        f.write_text("import os\n")
        files.append(
            DiscoveredFile(
                path=f"good_{i}.py",
                absolute_path=str(f),
                language="python",
            )
        )
    # 12th file points to a nonexistent path — worker will raise
    files.append(
        DiscoveredFile(
            path="missing.py",
            absolute_path=str(tmp_path / "missing.py"),
            language="python",
        )
    )
    with pytest.raises(ParseError, match="Parallel import parsing failed"):
        parse_imports(files, engine, adapters, parallel=True, strict=True)


def test_nonstrict_parallel_skips_bad_file(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """parse_imports returns good results and skips bad files when strict=False in parallel."""
    files: list[DiscoveredFile] = []
    for i in range(11):
        f = tmp_path / f"good_{i}.py"
        f.write_text("import os\n")
        files.append(
            DiscoveredFile(
                path=f"good_{i}.py",
                absolute_path=str(f),
                language="python",
            )
        )
    # 12th file points to a nonexistent path — worker will raise, should be skipped
    files.append(
        DiscoveredFile(
            path="missing.py",
            absolute_path=str(tmp_path / "missing.py"),
            language="python",
        )
    )
    result = parse_imports(files, engine, adapters, parallel=True, strict=False)
    # Bad file is skipped, good 11 files return results
    assert len(result) == 11
