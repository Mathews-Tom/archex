"""Source file discovery: language detection, gitignore filtering, and file enumeration."""

from __future__ import annotations

import subprocess
from pathlib import Path

from archex.exceptions import AcquireError
from archex.models import DiscoveredFile

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
}

DEFAULT_IGNORES: list[str] = [
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".venv/",
    "venv/",
    "vendor/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "dist/",
    "build/",
    ".eggs/",
    "*.egg-info/",
]


def _detect_language(path: Path) -> str | None:
    return EXTENSION_MAP.get(path.suffix.lower())


def _matches_ignore(rel_path: str, ignores: list[str]) -> bool:
    parts = Path(rel_path).parts
    for pattern in ignores:
        stripped = pattern.rstrip("/")
        if pattern.endswith("/"):
            # directory segment match
            if stripped in parts:
                return True
        else:
            if rel_path == pattern or Path(rel_path).name == pattern:
                return True
    return False


def discover_files(
    repo_path: Path,
    languages: list[str] | None = None,
    ignores: list[str] | None = None,
    max_file_size: int = 10_000_000,
) -> list[DiscoveredFile]:
    """Enumerate source files in repo_path.

    Uses `git ls-files` when a .git directory is present, otherwise falls back
    to Path.rglob. Filters by language and ignore patterns.

    Raises AcquireError if repo_path does not exist.
    """
    if not repo_path.exists():
        raise AcquireError(f"Repository path does not exist: {repo_path}")

    effective_ignores = list(ignores) if ignores is not None else list(DEFAULT_IGNORES)

    raw_paths: list[str]
    if (repo_path / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            raw_paths = [line for line in result.stdout.splitlines() if line.strip()]
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            raise AcquireError(f"git ls-files failed: {stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AcquireError("git ls-files timed out") from exc
    else:
        raw_paths = [str(p.relative_to(repo_path)) for p in repo_path.rglob("*") if p.is_file()]

    discovered: list[DiscoveredFile] = []
    for rel in raw_paths:
        if _matches_ignore(rel, effective_ignores):
            continue

        file_path = repo_path / rel
        if not file_path.is_file():
            continue

        lang = _detect_language(file_path)
        if lang is None:
            continue

        if languages is not None and lang not in languages:
            continue

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        if size > max_file_size:
            continue

        discovered.append(
            DiscoveredFile(
                path=rel,
                absolute_path=str(file_path),
                language=lang,
                size_bytes=size,
            )
        )

    return discovered
