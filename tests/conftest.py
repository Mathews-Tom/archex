from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _init_fixture_repo(tmp_path: Path, fixture_name: str) -> Path:
    """Copy a fixture directory to tmp_path and initialise a git repo."""
    dest = tmp_path / fixture_name
    shutil.copytree(FIXTURES_DIR / fixture_name, dest)
    subprocess.run(["git", "init"], cwd=dest, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@archex.test"],
        cwd=dest,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "archex-test"],
        cwd=dest,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=dest, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=dest,
        check=True,
        capture_output=True,
    )
    return dest


@pytest.fixture
def fixture_path():
    def _fixture_path(name: str) -> Path:
        path = FIXTURES_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Fixture '{name}' not found at {path}")
        return path

    return _fixture_path


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def python_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/python_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "python_simple")


@pytest.fixture
def java_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/java_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "java_simple")


@pytest.fixture
def kotlin_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/kotlin_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "kotlin_simple")


@pytest.fixture
def csharp_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/csharp_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "csharp_simple")


@pytest.fixture
def swift_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/swift_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "swift_simple")


@pytest.fixture
def typescript_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/typescript_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "typescript_simple")


@pytest.fixture
def rust_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/rust_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "rust_simple")


@pytest.fixture
def go_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/go_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "go_simple")


@pytest.fixture
def monorepo_simple_repo(tmp_path: Path) -> Path:
    """Copy tests/fixtures/monorepo_simple to a temp dir and initialise a git repo."""
    return _init_fixture_repo(tmp_path, "monorepo_simple")
