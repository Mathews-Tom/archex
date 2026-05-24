"""Repo-local archex project state helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from archex import __version__

PROJECT_DIR_NAME = ".archex"
PROJECT_GITIGNORE_ENTRY = f"{PROJECT_DIR_NAME}/"

DEFAULT_SETTINGS_TOML = """[project]
mode = "local"

[index]
cache_dir = ".archex"
languages = []
vector = false
delta_threshold = 0.5

[dogfood]
tasks_dir = "benchmarks/tasks"
output_dir = ".archex/dogfood"
default_task_prefix = "archex_"
strategies = ["raw_files", "raw_grepped", "archex_query"]
"""


@dataclass(frozen=True)
class ProjectState:
    """Resolved repo-local archex state paths."""

    repo_root: Path

    @property
    def project_dir(self) -> Path:
        return self.repo_root / PROJECT_DIR_NAME

    @property
    def settings_path(self) -> Path:
        return self.project_dir / "settings.toml"

    @property
    def metadata_path(self) -> Path:
        return self.project_dir / "metadata.json"

    @property
    def dogfood_dir(self) -> Path:
        return self.project_dir / "dogfood"

    @property
    def dogfood_history_dir(self) -> Path:
        return self.dogfood_dir / "history"

    @property
    def gitignore_path(self) -> Path:
        return self.repo_root / ".gitignore"

    @classmethod
    def resolve(cls, source: str | Path) -> ProjectState:
        """Resolve a source path to its Git repository root."""
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Source path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Source path is not a directory: {path}")

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "not a Git repository"
            raise ValueError(f"Could not resolve repository root for {path}: {detail}")
        root_text = result.stdout.strip()
        if not root_text:
            raise ValueError(f"git rev-parse returned an empty repository root for {path}")
        return cls(repo_root=Path(root_text).resolve())

    def initialized(self) -> bool:
        return self.settings_path.exists()


@dataclass(frozen=True)
class ProjectInitResult:
    """Result of initializing repo-local project state."""

    state: ProjectState
    created: bool
    settings_written: bool
    gitignore_updated: bool


def init_project(
    source: str | Path,
    *,
    force: bool = False,
    reset: bool = False,
) -> ProjectInitResult:
    """Create or refresh repo-local archex state."""
    if reset and not force:
        raise ValueError("--reset requires --force")

    state = ProjectState.resolve(source)
    existed = state.initialized()

    if reset and state.project_dir.exists():
        shutil.rmtree(state.project_dir)
        existed = False

    state.dogfood_history_dir.mkdir(parents=True, exist_ok=True)

    settings_written = False
    if force or not state.settings_path.exists():
        state.settings_path.write_text(DEFAULT_SETTINGS_TOML, encoding="utf-8")
        settings_written = True

    metadata = {
        "archex_version": __version__,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    state.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    gitignore_updated = ensure_project_gitignore(state.gitignore_path)

    return ProjectInitResult(
        state=state,
        created=not existed,
        settings_written=settings_written,
        gitignore_updated=gitignore_updated,
    )


def ensure_project_gitignore(gitignore_path: Path) -> bool:
    """Ensure `.archex/` is ignored exactly once."""
    if gitignore_path.exists():
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    if PROJECT_GITIGNORE_ENTRY in {line.strip() for line in lines}:
        return False

    if lines and lines[-1] != "":
        lines.append("")
    lines.append("# archex project state")
    lines.append(PROJECT_GITIGNORE_ENTRY)
    gitignore_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
