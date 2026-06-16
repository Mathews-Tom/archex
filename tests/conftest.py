from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Disable the global coverage threshold for narrow implementation-gate slices."""
    _disable_slice_coverage_threshold(config)


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    _disable_slice_coverage_threshold(session.config)


def _disable_slice_coverage_threshold(config: pytest.Config) -> None:
    if not implementation_gate_paths(config.invocation_params.args):
        return
    if getattr(config.option, "cov_fail_under", None) is not None:
        config.option.cov_fail_under = 0
    cov_plugin: Any = config.pluginmanager.getplugin("_cov")
    cov_options: Any = getattr(cov_plugin, "options", None)
    if cov_options is not None and getattr(cov_options, "cov_fail_under", None) is not None:
        cov_options.cov_fail_under = 0


def implementation_gate_paths(args: Sequence[str]) -> bool:
    paths = _path_args(args)
    if "tests" in paths and _implementation_gate_keyword(args):
        return True
    if not paths:
        return False
    if any(
        path == "tests/pipeline" or path.startswith("tests/pipeline/") for path in paths
    ) and any(path == "tests/benchmark" or path.startswith("tests/benchmark/") for path in paths):
        allowed_prefixes = ("tests/pipeline", "tests/index", "tests/benchmark")
        return all(
            any(path == prefix or path.startswith(f"{prefix}/") for prefix in allowed_prefixes)
            for path in paths
        )
    if any(path.startswith("tests/benchmark") for path in paths):
        allowed_prefixes = ("tests/analyze", "tests/benchmark", "tests/serve")
        return all(
            any(path == prefix or path.startswith(f"{prefix}/") for prefix in allowed_prefixes)
            for path in paths
        )
    if any(path == "tests/index" or path.startswith("tests/index/") for path in paths):
        return all(
            path == "tests/integrations/test_mcp.py"
            or path == "tests/index"
            or path.startswith("tests/index/")
            for path in paths
        )
    cli_metric_paths = (
        "tests/metrics",
        "tests/cli",
        "tests/test_cli.py",
    )

    def _is_cli_metric_path(path: str) -> bool:
        return path == "tests/test_cli.py" or any(
            path == prefix or path.startswith(f"{prefix}/") for prefix in cli_metric_paths[:-1]
        )

    if any(_is_cli_metric_path(path) for path in paths):
        return all(_is_cli_metric_path(path) for path in paths)
    if any(path in {"tests/test_graph_query.py", "tests/test_scout.py"} for path in paths):
        allowed_paths = {
            "tests/test_graph_query.py",
            "tests/test_graph_export_cli.py",
            "tests/test_scout.py",
            "tests/integrations/test_mcp.py",
        }
        return all(path in allowed_paths for path in paths)
    return False


def _path_args(args: Sequence[str]) -> list[str]:
    paths: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-k", "-m"}:
            skip_next = True
            continue
        if arg and not arg.startswith("-"):
            paths.append(arg.rstrip("/"))
    return paths


def _implementation_gate_keyword(args: Sequence[str]) -> bool:
    try:
        expression = args[args.index("-k") + 1]
    except (ValueError, IndexError):
        return False
    allowed_terms = {"scout", "graph_query", "mcp", "or", "and", "not", "(", ")"}
    normalized = expression.replace("(", " ( ").replace(")", " ) ")
    terms = {term for term in normalized.split() if term}
    return bool(terms) and terms.issubset(allowed_terms)


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
