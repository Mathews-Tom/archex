"""YAML task loading and validation for benchmark definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from archex.benchmark.models import BenchmarkTask

if TYPE_CHECKING:
    from pathlib import Path


def load_task(path: Path) -> BenchmarkTask:
    """Parse a single YAML file into a BenchmarkTask."""
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return BenchmarkTask.model_validate(data)


def load_tasks(directory: Path) -> list[BenchmarkTask]:
    """Load all *.yaml files from a directory into BenchmarkTasks."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {directory}")
    tasks: list[BenchmarkTask] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        tasks.append(load_task(yaml_file))
    return tasks


def validate_task(task: BenchmarkTask, repo_path: Path) -> list[str]:
    """Validate a task against a cloned repo. Returns list of error strings."""
    errors: list[str] = []

    if not repo_path.is_dir():
        errors.append(f"Repo path does not exist: {repo_path}")
        return errors

    for expected in task.expected_files:
        full = repo_path / expected
        if not full.is_file():
            errors.append(f"Expected file not found: {expected}")

    if not task.expected_files:
        errors.append("No expected_files defined")

    if not task.question.strip():
        errors.append("Empty question")

    return errors
