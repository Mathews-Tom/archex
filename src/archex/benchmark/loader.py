"""YAML task loading and validation for benchmark definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import yaml
from pydantic import BaseModel, ValidationError

from archex.benchmark.models import ArchitectureBenchmarkTask, BenchmarkTask, DeltaBenchmarkTask

if TYPE_CHECKING:
    from pathlib import Path

BenchmarkSpec = TypeVar("BenchmarkSpec", bound=BaseModel)


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return cast("dict[str, object]", data)


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        field = loc or "<root>"
        if error["type"] == "extra_forbidden":
            messages.append(f"unknown field {field!r}: {error['msg']}")
        else:
            messages.append(f"{field}: {error['msg']}")
    return f"Invalid benchmark YAML in {path}: " + "; ".join(messages)


def _validate_yaml_model(path: Path, model: type[BenchmarkSpec]) -> BenchmarkSpec:
    try:
        return model.model_validate(_load_yaml_mapping(path))
    except ValidationError as exc:
        raise ValueError(_format_validation_error(path, exc)) from exc


LoadedTask = TypeVar("LoadedTask", BenchmarkTask, ArchitectureBenchmarkTask, DeltaBenchmarkTask)


def _append_unique_task(
    tasks: list[LoadedTask],
    task: LoadedTask,
    path: Path,
    seen_paths: dict[str, Path],
) -> None:
    previous_path = seen_paths.get(task.task_id)
    if previous_path is not None:
        msg = f"Duplicate task_id {task.task_id!r} in {previous_path} and {path}"
        raise ValueError(msg)
    seen_paths[task.task_id] = path
    tasks.append(task)


def load_task(path: Path) -> BenchmarkTask:
    """Parse a single YAML file into a BenchmarkTask."""
    return _validate_yaml_model(path, BenchmarkTask)


def load_tasks(directory: Path) -> list[BenchmarkTask]:
    """Load all *.yaml files from a directory into BenchmarkTasks."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {directory}")
    tasks: list[BenchmarkTask] = []
    seen_paths: dict[str, Path] = {}
    for yaml_file in sorted(directory.glob("*.yaml")):
        _append_unique_task(tasks, load_task(yaml_file), yaml_file, seen_paths)
    return tasks


def load_arch_task(path: Path) -> ArchitectureBenchmarkTask:
    """Parse a single YAML file into an ArchitectureBenchmarkTask."""
    return _validate_yaml_model(path, ArchitectureBenchmarkTask)


def load_arch_tasks(directory: Path) -> list[ArchitectureBenchmarkTask]:
    """Load all *.yaml files from a directory into ArchitectureBenchmarkTasks."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Architecture tasks directory not found: {directory}")
    tasks: list[ArchitectureBenchmarkTask] = []
    seen_paths: dict[str, Path] = {}
    for yaml_file in sorted(directory.glob("*.yaml")):
        _append_unique_task(tasks, load_arch_task(yaml_file), yaml_file, seen_paths)
    return tasks


def load_delta_task(path: Path) -> DeltaBenchmarkTask:
    """Parse a single YAML file into a DeltaBenchmarkTask."""
    return _validate_yaml_model(path, DeltaBenchmarkTask)


def load_delta_tasks(directory: Path) -> list[DeltaBenchmarkTask]:
    """Load all *.yaml files from a directory into DeltaBenchmarkTasks."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {directory}")
    tasks: list[DeltaBenchmarkTask] = []
    seen_paths: dict[str, Path] = {}
    for yaml_file in sorted(directory.glob("*.yaml")):
        _append_unique_task(tasks, load_delta_task(yaml_file), yaml_file, seen_paths)
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
