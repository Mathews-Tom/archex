"""Head-to-head benchmark manifest loading and task selection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypeVar, cast

import yaml
from pydantic import BaseModel, ValidationError

from archex.benchmark.external_mcp import reset_external_tool_config, set_external_tool_config
from archex.benchmark.loader import load_tasks
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    HeadToHeadManifest,
    Strategy,
    TaskCategory,
)
from archex.benchmark.runner import run_all

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


HEADTOHEAD_REPORT_STRATEGIES: tuple[Strategy, ...] = (
    Strategy.ARCHEX_QUERY,
    Strategy.EXTERNAL_MCP,
    Strategy.RAW_GREPPED,
)


def run_headtohead(
    manifest_path: Path,
    output_dir: Path,
    tasks_dir: Path,
) -> list[BenchmarkReport]:
    """Run the manifest-pinned comparison across raw, archex, and one external lane."""
    manifest, tasks = load_headtohead_tasks(manifest_path, tasks_dir)
    if len(manifest.external_tools) != 1:
        raise HeadToHeadManifestError(
            "Head-to-head runner currently requires exactly one external tool"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_result in output_dir.glob("*.json"):
        stale_result.unlink()
    shutil.copy2(manifest_path, output_dir / "manifest.yaml")
    external_tool = manifest.external_tools[0]
    token = set_external_tool_config(external_tool)
    try:
        return run_all(
            tasks_dir=tasks_dir,
            output_dir=output_dir,
            strategies=[
                Strategy.RAW_FILES,
                manifest.raw_read_strategy,
                manifest.archex.strategy,
                Strategy.EXTERNAL_MCP,
            ],
            tasks=tasks,
            retrieval_options=BenchmarkRetrievalOptions(embedder=manifest.archex.embedder),
        )
    finally:
        reset_external_tool_config(token)


def load_headtohead_results(input_dir: Path) -> list[BenchmarkReport]:
    """Load result JSON files from a head-to-head output directory."""
    reports: list[BenchmarkReport] = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))
    return reports


def _lane_label(result: BenchmarkResult) -> str:
    if result.strategy is Strategy.ARCHEX_QUERY:
        return "archex"
    if result.strategy is Strategy.RAW_GREPPED:
        return "raw-grep/read"
    if result.strategy is Strategy.EXTERNAL_MCP:
        return result.strategy_label or result.provenance.get("external_tool", "external")
    return result.strategy.value


def _result_provenance(result: BenchmarkResult, manifest: HeadToHeadManifest) -> str:
    if result.strategy is Strategy.ARCHEX_QUERY:
        return f"manifest={manifest.name}; lane=archex; embedder={manifest.archex.embedder}"
    if result.strategy is Strategy.RAW_GREPPED:
        return f"manifest={manifest.name}; lane=raw-grep/read; source=repo files"
    tool = result.provenance.get("external_tool", result.strategy_label or "external")
    version = result.provenance.get("external_tool_version", "")
    embedder = result.provenance.get("external_tool_embedder", "")
    return f"manifest={manifest.name}; lane={tool}; version={version}; embedder={embedder}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_cell(value: float, provenance: str, field: str, *, digits: int = 2) -> str:
    return f"{value:.{digits}f}<br><sub>prov: {provenance}; field={field}</sub>"


def _integer_metric_cell(value: float, provenance: str, field: str) -> str:
    return f"{value:.0f}<br><sub>prov: {provenance}; field={field}</sub>"


def _results_by_lane(
    reports: list[BenchmarkReport],
) -> dict[str, list[BenchmarkResult]]:
    lanes: dict[str, list[BenchmarkResult]] = {}
    for report in reports:
        for result in report.results:
            if result.strategy in HEADTOHEAD_REPORT_STRATEGIES:
                lanes.setdefault(_lane_label(result), []).append(result)
    return lanes


def format_headtohead_markdown(
    manifest: HeadToHeadManifest,
    reports: list[BenchmarkReport],
) -> str:
    """Render the public three-way comparison without hiding losing cells."""
    if not reports:
        return "No head-to-head benchmark results."

    lanes = _results_by_lane(reports)
    required_lanes = {"archex", manifest.external_tools[0].name, "raw-grep/read"}
    missing_lanes = sorted(required_lanes.difference(lanes))
    if missing_lanes:
        raise HeadToHeadManifestError(
            "Head-to-head report is missing lane(s): " + ", ".join(missing_lanes)
        )
    lines: list[str] = [
        "# archex Head-to-Head Benchmark",
        "",
        f"Manifest: `{manifest.name}`",
        f"Tasks: `{len(reports)}` external-repo tasks",
        f"Hardware notes: {manifest.hardware_notes}",
        "",
        "Every metric cell includes its provenance. No winner filtering is applied.",
        "",
        "| Lane | Recall | Precision | F1 | Token efficiency | Completion penalty tokens "
        "| Efficiency after completion | Warm latency ms | Cold-start ms | Edit-to-correct ms "
        "| Freshness correct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for lane in sorted(lanes):
        results = lanes[lane]
        provenance = _result_provenance(results[0], manifest) + f"; tasks={len(results)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    lane,
                    _metric_cell(_mean([r.recall for r in results]), provenance, "recall"),
                    _metric_cell(_mean([r.precision for r in results]), provenance, "precision"),
                    _metric_cell(_mean([r.f1_score for r in results]), provenance, "f1_score"),
                    _metric_cell(
                        _mean([r.token_efficiency for r in results]),
                        provenance,
                        "token_efficiency",
                    ),
                    _integer_metric_cell(
                        _mean([float(r.bundle_completion_tokens) for r in results]),
                        provenance,
                        "bundle_completion_tokens",
                    ),
                    _metric_cell(
                        _mean([r.token_efficiency_with_completion for r in results]),
                        provenance,
                        "token_efficiency_with_completion",
                    ),
                    _integer_metric_cell(
                        _mean([r.warm_latency_ms or r.wall_time_ms for r in results]),
                        provenance,
                        "warm_latency_ms",
                    ),
                    _integer_metric_cell(
                        _mean([r.cold_start_ms for r in results]),
                        provenance,
                        "cold_start_ms",
                    ),
                    _integer_metric_cell(
                        _mean([r.freshness_latency_ms for r in results]),
                        provenance,
                        "freshness_latency_ms",
                    ),
                    _metric_cell(
                        _mean([1.0 if r.freshness_correct else 0.0 for r in results]),
                        provenance,
                        "freshness_correct",
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv tool install cocoindex-code  # operator choice: [full] for local embeddings",
            "uv run archex benchmark headtohead run --manifest "
            "benchmarks/headtohead/manifest.yaml --output .archex/headtohead",
            "uv run archex benchmark headtohead report --input .archex/headtohead "
            "--format markdown",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
