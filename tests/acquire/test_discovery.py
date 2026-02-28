from __future__ import annotations

from pathlib import Path

import pytest

from archex.acquire.discovery import (
    DEFAULT_IGNORES,
    discover_files,
)
from archex.acquire.discovery import (
    _detect_language as _detect_language,  # pyright: ignore[reportPrivateUsage]
)
from archex.acquire.discovery import (
    _matches_ignore as _matches_ignore,  # pyright: ignore[reportPrivateUsage]
)
from archex.exceptions import AcquireError


def test_discover_file_count(python_simple_repo: Path) -> None:
    files = discover_files(python_simple_repo)
    # python_simple has: main.py, models.py, utils.py, services/__init__.py, services/auth.py
    assert len(files) == 5


def test_discover_all_python(python_simple_repo: Path) -> None:
    files = discover_files(python_simple_repo)
    assert all(f.language == "python" for f in files)


def test_discover_language_filter(python_simple_repo: Path) -> None:
    files = discover_files(python_simple_repo, languages=["python"])
    assert len(files) == 5

    files_ts = discover_files(python_simple_repo, languages=["typescript"])
    assert len(files_ts) == 0


def test_discover_relative_paths(python_simple_repo: Path) -> None:
    files = discover_files(python_simple_repo)
    paths = {f.path for f in files}
    assert "main.py" in paths
    assert "models.py" in paths
    assert "utils.py" in paths


def test_discover_absolute_paths(python_simple_repo: Path) -> None:
    files = discover_files(python_simple_repo)
    for f in files:
        assert Path(f.absolute_path).is_absolute()
        assert Path(f.absolute_path).exists()


def test_discover_size_bytes(python_simple_repo: Path) -> None:
    files = discover_files(python_simple_repo)
    for f in files:
        assert f.size_bytes >= 0


def test_discover_ignore_rules(tmp_path: Path) -> None:
    # Create a non-git directory so rglob fallback is used
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "index.py").write_text("x = 1")
    node_mod = repo / "node_modules"
    node_mod.mkdir()
    (node_mod / "something.py").write_text("y = 2")

    files = discover_files(repo)
    paths = [f.path for f in files]
    assert "index.py" in paths
    assert not any("node_modules" in p for p in paths)


def test_discover_custom_ignores(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text("x = 1")
    skip_dir = repo / "skip_me"
    skip_dir.mkdir()
    (skip_dir / "ignored.py").write_text("y = 2")

    files = discover_files(repo, ignores=["skip_me/"])
    paths = [f.path for f in files]
    assert "keep.py" in paths
    assert not any("skip_me" in p for p in paths)


def test_discover_nonexistent_path() -> None:
    with pytest.raises(AcquireError, match="does not exist"):
        discover_files(Path("/nonexistent/repo/path"))


def test_detect_language_python() -> None:
    assert _detect_language(Path("foo.py")) == "python"


def test_detect_language_typescript() -> None:
    assert _detect_language(Path("bar.ts")) == "typescript"
    assert _detect_language(Path("bar.tsx")) == "typescript"


def test_detect_language_javascript() -> None:
    assert _detect_language(Path("baz.js")) == "javascript"
    assert _detect_language(Path("baz.jsx")) == "javascript"


def test_detect_language_go() -> None:
    assert _detect_language(Path("main.go")) == "go"


def test_detect_language_rust() -> None:
    assert _detect_language(Path("lib.rs")) == "rust"


def test_detect_language_unknown() -> None:
    assert _detect_language(Path("README.md")) is None
    assert _detect_language(Path("Makefile")) is None


def test_matches_ignore_directory() -> None:
    assert _matches_ignore("node_modules/index.js", ["node_modules/"])
    assert _matches_ignore("a/b/__pycache__/x.pyc", ["__pycache__/"])
    assert not _matches_ignore("src/main.py", ["node_modules/"])


def test_matches_ignore_file() -> None:
    assert _matches_ignore("exact_file.py", ["exact_file.py"])
    assert not _matches_ignore("other.py", ["exact_file.py"])


def test_default_ignores_content() -> None:
    assert "node_modules/" in DEFAULT_IGNORES
    assert ".git/" in DEFAULT_IGNORES
    assert "__pycache__/" in DEFAULT_IGNORES
    assert ".venv/" in DEFAULT_IGNORES


def test_discover_files_skips_oversized_files(tmp_path: Path) -> None:
    """Files exceeding max_file_size are excluded from discovery results."""
    repo = tmp_path / "repo"
    repo.mkdir()
    small = repo / "small.py"
    small.write_bytes(b"x = 1\n")
    big = repo / "big.py"
    big.write_bytes(b"x = 1\n" * 1000)

    small_size = small.stat().st_size
    big_size = big.stat().st_size
    # max_file_size set between small and big
    limit = (small_size + big_size) // 2

    files = discover_files(repo, max_file_size=limit)
    paths = [f.path for f in files]
    assert "small.py" in paths
    assert "big.py" not in paths


def test_discover_files_includes_file_at_size_limit(tmp_path: Path) -> None:
    """A file exactly at max_file_size is included."""
    repo = tmp_path / "repo"
    repo.mkdir()
    content = b"x = 1\n"
    exact = repo / "exact.py"
    exact.write_bytes(content)

    files = discover_files(repo, max_file_size=len(content))
    paths = [f.path for f in files]
    assert "exact.py" in paths


def test_discover_files_default_max_size_allows_normal_files(python_simple_repo: Path) -> None:
    """Default max_file_size (10MB) allows all fixture files through."""
    files = discover_files(python_simple_repo)
    assert len(files) == 5
