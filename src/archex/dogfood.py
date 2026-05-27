"""Dogfood benchmark orchestration for repo-local lifecycle runs."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from archex.benchmark.baseline import BaselineComparison, compare_baseline, load_baseline
from archex.benchmark.loader import load_tasks
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    Strategy,
    TaskCategory,
)
from archex.benchmark.reporter import format_markdown, format_summary
from archex.benchmark.runner import DEFAULT_STRATEGIES, run_benchmark
from archex.project import ProjectState

DOGFOOD_BASELINE_PATH = Path("benchmarks/dogfood_baseline.json")
DEFAULT_TASKS_DIR = Path("benchmarks/tasks")
DEFAULT_OUTPUT_DIR = Path(".archex/dogfood")
DEFAULT_TASK_PREFIX = "archex_"
DOGFOOD_DIAGNOSTIC_STRATEGIES = frozenset({Strategy.RAW_FILES.value, Strategy.RAW_GREPPED.value})


@dataclass(frozen=True)
class DogfoodRunResult:
    """Persisted dogfood run artifacts and baseline comparison."""

    repo_root: Path
    tasks: list[BenchmarkTask]
    reports: list[BenchmarkReport]
    comparisons: list[BaselineComparison]
    baseline_path: Path | None
    output_dir: Path
    latest_json_path: Path
    latest_markdown_path: Path
    history_json_path: Path

    @property
    def regressions(self) -> list[BaselineComparison]:
        return [comparison for comparison in self.comparisons if comparison.regression]


def run_dogfood(
    source: str | Path = ".",
    *,
    task_id: str | None = None,
    include_external: bool = False,
    include_query_fusion: bool = False,
    baseline_path: str | Path | None = None,
) -> DogfoodRunResult:
    """Run dogfood benchmark tasks against a local repository."""
    state = ProjectState.resolve(source)
    settings = _load_dogfood_settings(state)
    tasks_dir = _resolve_repo_path(settings.tasks_dir, state.repo_root)
    output_dir = _resolve_repo_path(settings.output_dir, state.repo_root)
    selected_tasks = _select_tasks(
        tasks_dir,
        task_id=task_id,
        include_external=include_external,
        default_task_prefix=settings.default_task_prefix,
    )
    strategies = _selected_strategies(
        settings.strategies,
        include_query_fusion=include_query_fusion,
    )

    reports = [
        run_benchmark(task, strategies=strategies, repo_path=state.repo_root)
        for task in selected_tasks
    ]
    baseline_file = _resolve_baseline_path(
        state,
        explicit_baseline=baseline_path,
        output_dir=output_dir,
    )
    comparisons = _compare_to_baseline(reports, baseline_file)
    latest_json_path, latest_markdown_path, history_json_path = _write_reports(
        output_dir,
        selected_tasks,
        reports,
        comparisons,
        baseline_file,
    )

    return DogfoodRunResult(
        repo_root=state.repo_root,
        tasks=selected_tasks,
        reports=reports,
        comparisons=comparisons,
        baseline_path=baseline_file,
        output_dir=output_dir,
        latest_json_path=latest_json_path,
        latest_markdown_path=latest_markdown_path,
        history_json_path=history_json_path,
    )


@dataclass(frozen=True)
class _DogfoodSettings:
    tasks_dir: Path
    output_dir: Path
    default_task_prefix: str
    strategies: list[Strategy]


def _load_dogfood_settings(state: ProjectState) -> _DogfoodSettings:
    if not state.settings_path.exists():
        return _DogfoodSettings(
            tasks_dir=DEFAULT_TASKS_DIR,
            output_dir=DEFAULT_OUTPUT_DIR,
            default_task_prefix=DEFAULT_TASK_PREFIX,
            strategies=list(DEFAULT_STRATEGIES),
        )

    data = tomllib.loads(state.settings_path.read_text(encoding="utf-8"))
    dogfood_data = data.get("dogfood", {})
    if not isinstance(dogfood_data, dict):
        raise ValueError(f"Invalid dogfood settings in {state.settings_path}")
    dogfood_settings = cast("dict[str, Any]", dogfood_data)

    tasks_dir = Path(_string_setting(dogfood_settings, "tasks_dir", str(DEFAULT_TASKS_DIR)))
    output_dir = Path(_string_setting(dogfood_settings, "output_dir", str(DEFAULT_OUTPUT_DIR)))
    default_task_prefix = _string_setting(
        dogfood_settings,
        "default_task_prefix",
        DEFAULT_TASK_PREFIX,
    )
    raw_strategies: object = dogfood_settings.get(
        "strategies",
        [strategy.value for strategy in DEFAULT_STRATEGIES],
    )
    if not isinstance(raw_strategies, list):
        raise ValueError(f"Invalid dogfood strategies in {state.settings_path}")
    strategy_values: list[str] = []
    for raw_strategy in cast("list[object]", raw_strategies):
        if not isinstance(raw_strategy, str):
            raise ValueError(f"Invalid dogfood strategies in {state.settings_path}")
        strategy_values.append(raw_strategy)
    strategies = [Strategy(value) for value in strategy_values]
    return _DogfoodSettings(
        tasks_dir=tasks_dir,
        output_dir=output_dir,
        default_task_prefix=default_task_prefix,
        strategies=strategies,
    )


def _string_setting(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Invalid dogfood setting {key}: expected string")
    return value


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return repo_root / expanded


def _select_tasks(
    tasks_dir: Path,
    *,
    task_id: str | None,
    include_external: bool,
    default_task_prefix: str,
) -> list[BenchmarkTask]:
    tasks = load_tasks(tasks_dir)
    if not include_external:
        tasks = [
            task
            for task in tasks
            if task.category == TaskCategory.SELF or task.task_id.startswith(default_task_prefix)
        ]
    if task_id is not None:
        tasks = [task for task in tasks if task.task_id == task_id]
    if not tasks:
        target = task_id if task_id is not None else f"{default_task_prefix}*"
        raise ValueError(f"No dogfood tasks found for {target}")
    return tasks


def _selected_strategies(
    configured_strategies: list[Strategy],
    *,
    include_query_fusion: bool,
) -> list[Strategy]:
    strategies = list(configured_strategies)
    if include_query_fusion and Strategy.ARCHEX_QUERY_FUSION not in strategies:
        strategies.append(Strategy.ARCHEX_QUERY_FUSION)
    return strategies


def _resolve_baseline_path(
    state: ProjectState,
    *,
    explicit_baseline: str | Path | None,
    output_dir: Path,
) -> Path | None:
    del output_dir
    if explicit_baseline is not None:
        path = Path(explicit_baseline).expanduser()
        if not path.is_absolute():
            path = state.repo_root / path
        if not path.is_file():
            raise ValueError(f"Dogfood baseline not found: {path}")
        return path

    raise ValueError("Dogfood requires an explicit --baseline path")


def _compare_to_baseline(
    reports: list[BenchmarkReport],
    baseline_path: Path | None,
) -> list[BaselineComparison]:
    if baseline_path is None:
        return []
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = load_baseline(baseline_data)
    return compare_baseline(
        reports,
        baseline,
        excluded_strategies=DOGFOOD_DIAGNOSTIC_STRATEGIES,
    )


def _write_reports(
    output_dir: Path,
    tasks: list[BenchmarkTask],
    reports: list[BenchmarkReport],
    comparisons: list[BaselineComparison],
    baseline_path: Path | None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = _json_payload(timestamp, tasks, reports, comparisons, baseline_path)

    latest_json_path = output_dir / "latest.json"
    latest_markdown_path = output_dir / "latest.md"
    history_json_path = history_dir / f"{timestamp.replace(':', '-')}.json"

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    latest_json_path.write_text(serialized, encoding="utf-8")
    history_json_path.write_text(serialized, encoding="utf-8")
    latest_markdown_path.write_text(
        _markdown_report(tasks, reports, comparisons, baseline_path),
        encoding="utf-8",
    )
    return latest_json_path, latest_markdown_path, history_json_path


def _json_payload(
    timestamp: str,
    tasks: list[BenchmarkTask],
    reports: list[BenchmarkReport],
    comparisons: list[BaselineComparison],
    baseline_path: Path | None,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "baseline_path": str(baseline_path) if baseline_path is not None else "",
        "tasks": [task.task_id for task in tasks],
        "reports": [report.model_dump(mode="json") for report in reports],
        "comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
        "regressions": [
            comparison.model_dump(mode="json")
            for comparison in comparisons
            if comparison.regression
        ],
        "retrieval_diagnostics": _retrieval_diagnostics(tasks, reports),
    }


def _retrieval_diagnostics(
    tasks: list[BenchmarkTask],
    reports: list[BenchmarkReport],
) -> list[dict[str, object]]:
    expected_by_task = {task.task_id: set(task.expected_files) for task in tasks}
    diagnostics: list[dict[str, object]] = []
    for report in reports:
        expected = expected_by_task[report.task_id]
        for result in report.results:
            ranked_files = _ranked_files(result)
            missing = _missing_expected_files(expected, ranked_files, result)
            unexpected = [path for path in ranked_files if path not in expected]
            failure_classes = _failure_classes(
                expected,
                ranked_files,
                missing,
                result,
            )
            diagnostics.append(
                {
                    "task_id": report.task_id,
                    "strategy": result.strategy.value,
                    "failure_classes": failure_classes,
                    "missing_expected_files": missing,
                    "top_unexpected_files": unexpected[:5],
                    "seed_files": result.seed_files,
                    "expanded_files": result.expanded_files,
                    "seed_recall": result.seed_recall,
                    "recall": result.recall,
                }
            )
    return diagnostics


def _ranked_files(result: BenchmarkResult) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for path in [*result.seed_files, *result.expanded_files]:
        if path in seen:
            continue
        seen.add(path)
        ranked.append(path)
    return ranked


def _missing_expected_files(
    expected: set[str],
    ranked_files: list[str],
    result: BenchmarkResult,
) -> list[str]:
    if ranked_files:
        return sorted(expected - set(ranked_files))
    if result.recall >= 1.0:
        return []
    return sorted(expected)


def _failure_classes(
    expected: set[str],
    ranked_files: list[str],
    missing: list[str],
    result: BenchmarkResult,
) -> list[str]:
    classes: list[str] = []
    if result.recall <= 0:
        classes.append("zero_recall")
    elif result.recall < 1.0 or missing:
        classes.append("partial_recall")

    if result.mrr < 1.0 and any(path in expected for path in ranked_files):
        classes.append("ranking_miss")

    if result.seed_files:
        seed_missing = expected - set(result.seed_files)
        if seed_missing:
            classes.append("seed_miss")

    if result.seed_recall > result.recall:
        classes.append("packing_miss")

    if result.expanded_files and missing:
        classes.append("expansion_miss")

    return classes


def _markdown_report(
    tasks: list[BenchmarkTask],
    reports: list[BenchmarkReport],
    comparisons: list[BaselineComparison],
    baseline_path: Path | None,
) -> str:
    lines: list[str] = ["# Dogfood Report", ""]
    if baseline_path is None:
        lines.append("Baseline: none")
    else:
        lines.append(f"Baseline: `{baseline_path}`")
    lines.append("")
    lines.append(format_summary(reports))
    lines.append("")
    lines.append("## Baseline Regressions")
    lines.append("")
    regressions = [comparison for comparison in comparisons if comparison.regression]
    if not comparisons:
        lines.append("No baseline comparison was run.")
    elif not regressions:
        lines.append("No baseline regressions detected.")
    else:
        lines.append("| Task | Strategy | Metric | Baseline | Current | Delta |")
        lines.append("|------|----------|--------|----------|---------|-------|")
        for regression in regressions:
            lines.append(
                f"| {regression.task_id} | {regression.strategy} | {regression.metric} "
                f"| {regression.baseline_value:.3f} | {regression.current_value:.3f} "
                f"| {regression.delta:+.3f} |"
            )
    lines.append("")
    lines.extend(_markdown_retrieval_diagnostics(tasks, reports))
    lines.append("")
    for report in reports:
        lines.append(format_markdown(report))
    return "\n".join(lines)


def _markdown_retrieval_diagnostics(
    tasks: list[BenchmarkTask],
    reports: list[BenchmarkReport],
) -> list[str]:
    diagnostics = _retrieval_diagnostics(tasks, reports)
    lines: list[str] = ["## Retrieval Diagnostics", ""]
    lines.append("| Task | Strategy | Failure Classes | Missing Expected | Top Unexpected |")
    lines.append("|------|----------|-----------------|------------------|----------------|")
    for diagnostic in diagnostics:
        classes = _join_markdown_list(cast("list[str]", diagnostic["failure_classes"]))
        missing = _join_markdown_list(cast("list[str]", diagnostic["missing_expected_files"]))
        unexpected = _join_markdown_list(cast("list[str]", diagnostic["top_unexpected_files"]))
        lines.append(
            f"| {diagnostic['task_id']} | {diagnostic['strategy']} "
            f"| {classes} | {missing} | {unexpected} |"
        )
    lines.append("")
    for diagnostic in diagnostics:
        lines.append(f"### {diagnostic['task_id']} / {diagnostic['strategy']}")
        lines.append("")
        lines.append("Seed files:")
        lines.extend(_markdown_bullets(cast("list[str]", diagnostic["seed_files"])))
        lines.append("")
        lines.append("Expanded files:")
        lines.extend(_markdown_bullets(cast("list[str]", diagnostic["expanded_files"])))
        lines.append("")
    return lines


def _join_markdown_list(values: list[str]) -> str:
    if not values:
        return "none"
    return "<br>".join(f"`{value}`" for value in values)


def _markdown_bullets(values: list[str]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{value}`" for value in values]
