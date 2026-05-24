from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from archex.project import DEFAULT_SETTINGS_TOML, ProjectState, init_project


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@archex.test")
    _git(repo, "config", "user.name", "archex-test")
    (repo / "README.md").write_text("# repo\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_project_state_resolves_git_root_from_subdirectory(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    subdir = repo / "src" / "pkg"
    subdir.mkdir(parents=True)

    state = ProjectState.resolve(subdir)

    assert state.repo_root == repo.resolve()
    assert state.settings_path == repo / ".archex" / "settings.toml"


def test_project_state_rejects_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not resolve repository root"):
        ProjectState.resolve(tmp_path)


def test_init_project_creates_settings_metadata_dogfood_and_gitignore(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    result = init_project(repo)

    assert result.created is True
    assert result.settings_written is True
    assert result.gitignore_updated is True
    assert (repo / ".archex" / "settings.toml").read_text(encoding="utf-8") == (
        DEFAULT_SETTINGS_TOML
    )
    assert (repo / ".archex" / "dogfood" / "history").is_dir()
    metadata = json.loads((repo / ".archex" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["archex_version"]
    assert metadata["created_at"]
    assert ".archex/" in (repo / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_init_project_is_idempotent_and_does_not_duplicate_gitignore(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    init_project(repo)
    settings = repo / ".archex" / "settings.toml"
    settings.write_text("custom = true\n", encoding="utf-8")

    result = init_project(repo)

    assert result.created is False
    assert result.settings_written is False
    assert result.gitignore_updated is False
    assert settings.read_text(encoding="utf-8") == "custom = true\n"
    gitignore_lines = (repo / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert gitignore_lines.count(".archex/") == 1


def test_init_project_force_rewrites_settings_but_preserves_generated_files(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    init_project(repo)
    settings = repo / ".archex" / "settings.toml"
    generated = repo / ".archex" / "index.db"
    settings.write_text("custom = true\n", encoding="utf-8")
    generated.write_text("db", encoding="utf-8")

    result = init_project(repo, force=True)

    assert result.created is False
    assert result.settings_written is True
    assert settings.read_text(encoding="utf-8") == DEFAULT_SETTINGS_TOML
    assert generated.read_text(encoding="utf-8") == "db"


def test_init_project_reset_requires_force(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    with pytest.raises(ValueError, match="--reset requires --force"):
        init_project(repo, reset=True)


def test_init_project_force_reset_removes_existing_state(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    init_project(repo)
    generated = repo / ".archex" / "index.db"
    generated.write_text("db", encoding="utf-8")

    result = init_project(repo, force=True, reset=True)

    assert result.created is True
    assert result.settings_written is True
    assert not generated.exists()
    assert (repo / ".archex" / "settings.toml").exists()
