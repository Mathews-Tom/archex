from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from archex.models import Config, DiscoveredFile, SymbolKind
from archex.parse.adapters import default_adapter_registry
from archex.parse.engine import TreeSitterEngine
from archex.parse.imports import parse_imports
from archex.parse.symbols import extract_symbols, extract_symbols_and_imports

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


def test_extract_symbols_and_imports_matches_separate_passes(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    combined = extract_symbols_and_imports(python_simple_files, engine, adapters)

    separate_symbols = extract_symbols(python_simple_files, TreeSitterEngine(), adapters)
    separate_imports = parse_imports(python_simple_files, TreeSitterEngine(), adapters)

    assert [parsed.model_dump() for parsed in combined.parsed_files] == [
        parsed.model_dump() for parsed in separate_symbols
    ]
    assert {
        path: [imp.model_dump() for imp in imports]
        for path, imports in combined.imports_by_path.items()
    } == {path: [imp.model_dump() for imp in imports] for path, imports in separate_imports.items()}


def test_extract_symbols_and_imports_reads_each_file_exactly_once(
    python_simple_files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """Symbol extraction, chunk-range extraction, and import parsing all reuse
    one read of a file's bytes — no per-file double read via a separate tree
    parse and a second manual read."""
    from pathlib import Path

    with patch("pathlib.Path.read_bytes", wraps=Path.read_bytes, autospec=True) as mock_read_bytes:
        extract_symbols_and_imports(python_simple_files, engine, adapters, parallel=False)
    assert mock_read_bytes.call_count == len(python_simple_files)


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


# --- _parse_file_worker direct tests ---


def test_parse_file_worker_unsupported_language() -> None:
    """_parse_file_worker returns None for a language with no adapter."""
    from archex.parse.symbols import _parse_file_worker  # pyright: ignore[reportPrivateUsage]

    result = _parse_file_worker("/fake/file.xyz", "file.xyz", "brainfuck")
    assert result is None


def test_parse_file_worker_success(tmp_path: Path) -> None:
    """_parse_file_worker reads a real file and returns a ParsedFile."""
    from archex.parse.symbols import _parse_file_worker  # pyright: ignore[reportPrivateUsage]

    py_file = tmp_path / "sample.py"
    py_file.write_text("def hello():\n    pass\n")
    result = _parse_file_worker(str(py_file), "sample.py", "python")
    assert result is not None
    assert result.path == "sample.py"
    assert result.language == "python"
    assert len(result.symbols) >= 1
    assert result.lines > 0


def test_parse_file_worker_raises_on_missing_file() -> None:
    """_parse_file_worker propagates ParseError for a missing file."""
    from archex.exceptions import ParseError
    from archex.parse.symbols import _parse_file_worker  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ParseError):
        _parse_file_worker("/nonexistent/ghost.py", "ghost.py", "python")


# --- parallel-by-default config tests ---


def test_config_parallel_defaults_to_true() -> None:
    """Config.parallel defaults to True so parsing uses all available cores by default."""
    assert Config().parallel is True


def test_parallel_default_config_uses_process_pool(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """A fresh Config's default parallel=True drives ProcessPoolExecutor for a >10-file batch."""
    files: list[DiscoveredFile] = []
    for i in range(12):
        f = tmp_path / f"mod_{i}.py"
        f.write_text(f"def func_{i}():\n    pass\n")
        files.append(DiscoveredFile(path=f"mod_{i}.py", absolute_path=str(f), language="python"))

    config = Config()

    with patch("archex.parse.symbols.ProcessPoolExecutor", wraps=ProcessPoolExecutor) as pool_spy:
        result = extract_symbols_and_imports(files, engine, adapters, parallel=config.parallel)

    pool_spy.assert_called_once()
    assert len(result.parsed_files) == 12


# --- parallel path tests ---


def test_extract_symbols_parallel_path(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """extract_symbols uses ProcessPoolExecutor when parallel=True and >10 files."""
    files: list[DiscoveredFile] = []
    for i in range(12):
        f = tmp_path / f"mod_{i}.py"
        f.write_text(f"def func_{i}():\n    pass\n")
        files.append(
            DiscoveredFile(
                path=f"mod_{i}.py",
                absolute_path=str(f),
                language="python",
            )
        )
    result = extract_symbols(files, engine, adapters, parallel=True)
    assert len(result) == 12
    for pf in result:
        assert pf.language == "python"
        assert len(pf.symbols) >= 1


def test_extract_symbols_parallel_fallback_on_error(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """extract_symbols falls back to sequential when ProcessPoolExecutor raises."""
    files: list[DiscoveredFile] = []
    for i in range(12):
        f = tmp_path / f"mod_{i}.py"
        f.write_text("def hello():\n    pass\n")
        files.append(
            DiscoveredFile(
                path=f"mod_{i}.py",
                absolute_path=str(f),
                language="python",
            )
        )
    with patch("archex.parse.symbols.ProcessPoolExecutor", side_effect=RuntimeError("fail")):
        result = extract_symbols(files, engine, adapters, parallel=True)
    assert len(result) == 12


def test_strict_parallel_raises_on_bad_file(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """extract_symbols raises ParseError when strict=True and a file fails in parallel mode."""
    from archex.exceptions import ParseError

    files: list[DiscoveredFile] = []
    for i in range(11):
        f = tmp_path / f"good_{i}.py"
        f.write_text(f"def func_{i}():\n    pass\n")
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
    with pytest.raises(ParseError, match="Parallel parsing failed"):
        extract_symbols(files, engine, adapters, parallel=True, strict=True)


def test_parallel_total_worker_failure_raises(
    tmp_path: Path, engine: TreeSitterEngine, adapters: dict[str, LanguageAdapter]
) -> None:
    """When all files fail to parse, extract_symbols raises ParseError even without strict."""
    files: list[DiscoveredFile] = []
    for i in range(12):
        f = tmp_path / f"mod_{i}.py"
        files.append(
            DiscoveredFile(
                path=f.name,
                absolute_path=str(f),  # file doesn't exist, will raise OSError
                language="python",
            )
        )
    from archex.exceptions import ParseError

    with pytest.raises(ParseError, match="Total worker failure: all 12 files failed to parse"):
        extract_symbols(files, engine, adapters, parallel=True, strict=False)


def test_sequential_total_worker_failure_raises(
    tmp_path: Path, engine: TreeSitterEngine, adapters: dict[str, LanguageAdapter]
) -> None:
    """All parse failures raise ParseError without strict mode."""
    files: list[DiscoveredFile] = []
    for i in range(2):  # less than 10 falls back to sequential
        f = tmp_path / f"mod_{i}.py"
        files.append(
            DiscoveredFile(
                path=f.name,
                absolute_path=str(f),  # file doesn't exist, will raise OSError/ParseError
                language="python",
            )
        )
    from archex.exceptions import ParseError

    with pytest.raises(ParseError, match="Total parser failure: all 2 files failed to parse"):
        extract_symbols_and_imports(files, engine, adapters, parallel=False, strict=False)


def test_sequential_fault_isolation_skips_oversized_file(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """A single oversized file raises ParseError from the engine but does not abort the batch."""
    good = tmp_path / "good.py"
    good.write_text("def hello():\n    pass\n")
    oversized = tmp_path / "oversized.py"
    oversized.write_bytes(b"x" * 10_000_001)  # exceeds TreeSitterEngine's default max_file_size

    files = [
        DiscoveredFile(path="good.py", absolute_path=str(good), language="python"),
        DiscoveredFile(path="oversized.py", absolute_path=str(oversized), language="python"),
    ]

    result = extract_symbols(files, engine, adapters, parallel=False)
    assert [pf.path for pf in result] == ["good.py"]


def test_sequential_fault_isolation_and_imports_skips_bad_file(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """extract_symbols_and_imports isolates a per-file ParseError in the sequential path."""
    good = tmp_path / "good.py"
    good.write_text("def hello():\n    pass\n")

    files = [
        DiscoveredFile(path="good.py", absolute_path=str(good), language="python"),
        DiscoveredFile(
            path="missing.py",
            absolute_path=str(tmp_path / "missing.py"),
            language="python",
        ),
    ]

    result = extract_symbols_and_imports(files, engine, adapters, parallel=False)
    assert [pf.path for pf in result.parsed_files] == ["good.py"]
    assert set(result.imports_by_path) == {"good.py"}


def test_sequential_fault_isolation_strict_raises(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """extract_symbols raises ParseError when strict=True and a file fails sequentially."""
    from archex.exceptions import ParseError

    good = tmp_path / "good.py"
    good.write_text("def hello():\n    pass\n")

    files = [
        DiscoveredFile(path="good.py", absolute_path=str(good), language="python"),
        DiscoveredFile(
            path="missing.py",
            absolute_path=str(tmp_path / "missing.py"),
            language="python",
        ),
    ]

    with pytest.raises(ParseError, match="Sequential parsing failed"):
        extract_symbols(files, engine, adapters, parallel=False, strict=True)


def test_nonstrict_parallel_skips_bad_file(
    tmp_path: Path,
    engine: TreeSitterEngine,
    adapters: dict[str, LanguageAdapter],
) -> None:
    """extract_symbols returns good results and skips bad files when strict=False in parallel."""
    files: list[DiscoveredFile] = []
    for i in range(11):
        f = tmp_path / f"good_{i}.py"
        f.write_text(f"def func_{i}():\n    pass\n")
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
    result = extract_symbols(files, engine, adapters, parallel=True, strict=False)
    # Bad file is skipped, good 11 files return results
    assert len(result) == 11
    for pf in result:
        assert pf.language == "python"
