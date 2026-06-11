"""Head-to-head benchmark manifest loading and task selection."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

import yaml
from pydantic import BaseModel, ValidationError

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkTask, HeadToHeadManifest, Strategy, TaskCategory

ManifestModel = TypeVar("ManifestModel", bound=BaseModel)


class HeadToHeadManifestError(ValueError):
    """Raised when the public comparison manifest is invalid."""


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise HeadToHeadManifestError(f"Failed to parse YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise HeadToHeadManifestError(
            f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        )
    return cast("dict[str, object]", data)


def _format_manifest_validation_error(path: Path, exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        field = loc or "<root>"
        if error["type"] == "extra_forbidden":
            messages.append(f"unknown field {field!r}: {error['msg']}")
        else:
            messages.append(f"{field}: {error['msg']}")
    return f"Invalid head-to-head manifest in {path}: " + "; ".join(messages)


def _validate_yaml_model(path: Path, model: type[ManifestModel]) -> ManifestModel:
    try:
        return model.model_validate(_load_yaml_mapping(path))
    except ValidationError as exc:
        raise HeadToHeadManifestError(_format_manifest_validation_error(path, exc)) from exc


def _reject_empty_text(path: Path, field: str, value: str) -> None:
    if not value.strip():
        raise HeadToHeadManifestError(f"Invalid head-to-head manifest in {path}: {field}: empty")


def _reject_unpinned_version(path: Path, tool_name: str, version: str) -> None:
    normalized = version.strip().lower()
    if normalized in {"", "latest", "head", "main", "operator-pinned"}:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: external_tools.{tool_name}.version "
            "must pin an exact released version"
        )


def _validate_manifest_shape(path: Path, manifest: HeadToHeadManifest) -> None:
    if manifest.manifest_version != 1:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: manifest_version must be 1"
        )
    _reject_empty_text(path, "name", manifest.name)
    _reject_empty_text(path, "hardware_notes", manifest.hardware_notes)
    if not manifest.task_subset:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: task_subset must not be empty"
        )
    if len(set(manifest.task_subset)) != len(manifest.task_subset):
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: task_subset contains duplicate task ids"
        )
    if not manifest.archex.local_models_only:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: archex.local_models_only must be true"
        )
    if manifest.archex.strategy is not Strategy.ARCHEX_QUERY:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: archex.strategy must be archex_query"
        )
    if manifest.raw_read_strategy is not Strategy.RAW_GREPPED:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: raw_read_strategy must be raw_grepped"
        )
    if not manifest.external_tools:
        raise HeadToHeadManifestError(
            f"Invalid head-to-head manifest in {path}: external_tools must not be empty"
        )
    seen_tools: set[str] = set()
    for tool in manifest.external_tools:
        _reject_empty_text(path, f"external_tools.{tool.name}.name", tool.name)
        _reject_empty_text(path, f"external_tools.{tool.name}.command", tool.command)
        _reject_empty_text(path, f"external_tools.{tool.name}.embedder", tool.embedder)
        _reject_unpinned_version(path, tool.name, tool.version)
        if tool.name in seen_tools:
            raise HeadToHeadManifestError(
                f"Invalid head-to-head manifest in {path}: duplicate external tool {tool.name!r}"
            )
        seen_tools.add(tool.name)


def load_headtohead_manifest(path: Path) -> HeadToHeadManifest:
    """Load and strictly validate a public head-to-head comparison manifest."""
    manifest = _validate_yaml_model(path, HeadToHeadManifest)
    _validate_manifest_shape(path, manifest)
    return manifest


def select_headtohead_tasks(
    manifest: HeadToHeadManifest,
    tasks_dir: Path,
) -> list[BenchmarkTask]:
    """Select the manifest-pinned external task subset in manifest order."""
    all_tasks = {task.task_id: task for task in load_tasks(tasks_dir)}
    missing = [task_id for task_id in manifest.task_subset if task_id not in all_tasks]
    if missing:
        raise HeadToHeadManifestError(
            "Head-to-head manifest references unknown task ids: " + ", ".join(missing)
        )

    selected = [all_tasks[task_id] for task_id in manifest.task_subset]
    self_tasks = [
        task.task_id for task in selected if task.repo == "." or task.category == TaskCategory.SELF
    ]
    if self_tasks:
        raise HeadToHeadManifestError(
            "Head-to-head task subset must exclude self-repo tasks: " + ", ".join(self_tasks)
        )
    return selected


def load_headtohead_tasks(
    manifest_path: Path, tasks_dir: Path
) -> tuple[HeadToHeadManifest, list[BenchmarkTask]]:
    """Load a manifest and its external benchmark tasks."""
    manifest = load_headtohead_manifest(manifest_path)
    return manifest, select_headtohead_tasks(manifest, tasks_dir)
