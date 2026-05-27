"""Product-readiness reporting for persisted benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass

from archex.benchmark.models import BenchmarkReport, BenchmarkResult, BenchmarkTask, Strategy
from archex.benchmark.triage import TriageFinding, triage_failures

TARGET_MEAN_RECALL = 0.80
TARGET_MEAN_PRECISION = 0.60
TARGET_MEAN_F1 = 0.70
TARGET_ZERO_RECALL_TASKS = 1
LOW_F1_THRESHOLD = 0.40
LOW_PRECISION_THRESHOLD = 0.35


@dataclass(frozen=True)
class ReadinessTargetStatus:
    """Pass/fail status for one readiness target."""

    name: str
    actual: float | int
    target: float | int
    passed: bool
    comparator: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "actual": self.actual,
            "target": self.target,
            "passed": self.passed,
            "comparator": self.comparator,
        }


@dataclass(frozen=True)
class CategoryReadiness:
    """Aggregated metrics for one benchmark category."""

    category: str
    task_count: int
    mean_recall: float
    mean_precision: float
    mean_f1_score: float
    mean_mrr: float

    def to_json(self) -> dict[str, object]:
        return {
            "category": self.category,
            "task_count": self.task_count,
            "mean_recall": self.mean_recall,
            "mean_precision": self.mean_precision,
            "mean_f1_score": self.mean_f1_score,
            "mean_mrr": self.mean_mrr,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Non-blocking product-readiness report for one strategy."""

    strategy: str
    task_count: int
    mean_recall: float
    mean_precision: float
    mean_f1_score: float
    mean_mrr: float
    median_latency_ms: float
    p95_latency_ms: float
    zero_recall_tasks: int
    low_f1_tasks: int
    low_precision_tasks: int
    targets: list[ReadinessTargetStatus]
    categories: list[CategoryReadiness]
    blocking_tasks: list[TriageFinding]

    @property
    def ready(self) -> bool:
        return all(target.passed for target in self.targets)

    def to_json(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "ready": self.ready,
            "task_count": self.task_count,
            "mean_recall": self.mean_recall,
            "mean_precision": self.mean_precision,
            "mean_f1_score": self.mean_f1_score,
            "mean_mrr": self.mean_mrr,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "zero_recall_tasks": self.zero_recall_tasks,
            "low_f1_tasks": self.low_f1_tasks,
            "low_precision_tasks": self.low_precision_tasks,
            "targets": [target.to_json() for target in self.targets],
            "categories": [category.to_json() for category in self.categories],
            "blocking_tasks": [finding.to_json() for finding in self.blocking_tasks],
        }


def build_readiness_report(
    reports: list[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy = Strategy.ARCHEX_QUERY,
) -> ReadinessReport:
    """Build a non-blocking readiness report for persisted benchmark results."""
    results: list[tuple[BenchmarkReport, BenchmarkResult]] = []
    for report in reports:
        result = _find_result(report, strategy)
        if result is not None:
            results.append((report, result))

    if not results:
        return ReadinessReport(
            strategy=strategy.value,
            task_count=0,
            mean_recall=0.0,
            mean_precision=0.0,
            mean_f1_score=0.0,
            mean_mrr=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            zero_recall_tasks=0,
            low_f1_tasks=0,
            low_precision_tasks=0,
            targets=[],
            categories=[],
            blocking_tasks=[],
        )

    metric_results = [result for _, result in results]
    mean_recall = _mean([result.recall for result in metric_results])
    mean_precision = _mean([result.precision for result in metric_results])
    mean_f1 = _mean([result.f1_score for result in metric_results])
    mean_mrr = _mean([result.mrr for result in metric_results])
    median_latency_ms = _percentile([result.wall_time_ms for result in metric_results], 0.50)
    p95_latency_ms = _percentile([result.wall_time_ms for result in metric_results], 0.95)
    zero_recall_tasks = sum(1 for result in metric_results if result.recall <= 0.0)
    low_f1_tasks = sum(1 for result in metric_results if result.f1_score < LOW_F1_THRESHOLD)
    low_precision_tasks = sum(
        1 for result in metric_results if result.precision < LOW_PRECISION_THRESHOLD
    )
    targets = [
        ReadinessTargetStatus(
            name="mean_recall",
            actual=mean_recall,
            target=TARGET_MEAN_RECALL,
            passed=mean_recall >= TARGET_MEAN_RECALL,
            comparator=">=",
        ),
        ReadinessTargetStatus(
            name="mean_precision",
            actual=mean_precision,
            target=TARGET_MEAN_PRECISION,
            passed=mean_precision >= TARGET_MEAN_PRECISION,
            comparator=">=",
        ),
        ReadinessTargetStatus(
            name="mean_f1_score",
            actual=mean_f1,
            target=TARGET_MEAN_F1,
            passed=mean_f1 >= TARGET_MEAN_F1,
            comparator=">=",
        ),
        ReadinessTargetStatus(
            name="zero_recall_tasks",
            actual=zero_recall_tasks,
            target=TARGET_ZERO_RECALL_TASKS,
            passed=zero_recall_tasks <= TARGET_ZERO_RECALL_TASKS,
            comparator="<=",
        ),
    ]
    categories = _category_readiness(results, tasks_by_id)
    blocking_tasks = triage_failures(reports, tasks_by_id, strategy=strategy)[:10]
    return ReadinessReport(
        strategy=strategy.value,
        task_count=len(results),
        mean_recall=mean_recall,
        mean_precision=mean_precision,
        mean_f1_score=mean_f1,
        mean_mrr=mean_mrr,
        median_latency_ms=median_latency_ms,
        p95_latency_ms=p95_latency_ms,
        zero_recall_tasks=zero_recall_tasks,
        low_f1_tasks=low_f1_tasks,
        low_precision_tasks=low_precision_tasks,
        targets=targets,
        categories=categories,
        blocking_tasks=blocking_tasks,
    )


def format_readiness_markdown(report: ReadinessReport) -> str:
    """Render a readiness report as Markdown."""
    if report.task_count == 0:
        return f"# Benchmark Readiness\n\nNo `{report.strategy}` results found."

    lines = ["# Benchmark Readiness", ""]
    lines.append(f"Strategy: `{report.strategy}`")
    lines.append(f"Tasks: `{report.task_count}`")
    lines.append(f"Ready: `{'yes' if report.ready else 'no'}`")
    lines.append("")
    lines.append("| Metric | Actual | Target | Status |")
    lines.append("|---|---:|---:|---|")
    for target in report.targets:
        status = "pass" if target.passed else "fail"
        lines.append(
            f"| {target.name} | {_format_number(target.actual)} "
            f"| {target.comparator} {_format_number(target.target)} | {status} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Mean recall: `{report.mean_recall:.3f}`\n"
        f"- Mean precision: `{report.mean_precision:.3f}`\n"
        f"- Mean F1: `{report.mean_f1_score:.3f}`\n"
        f"- Mean MRR: `{report.mean_mrr:.3f}`\n"
        f"- Median latency: `{report.median_latency_ms:.0f} ms`\n"
        f"- P95 latency: `{report.p95_latency_ms:.0f} ms`\n"
        f"- Zero-recall tasks: `{report.zero_recall_tasks}`\n"
        f"- Tasks below F1 {LOW_F1_THRESHOLD:.2f}: `{report.low_f1_tasks}`\n"
        f"- Tasks below precision {LOW_PRECISION_THRESHOLD:.2f}: `{report.low_precision_tasks}`"
    )
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    lines.append("| Category | Tasks | Recall | Precision | F1 | MRR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category in report.categories:
        lines.append(
            f"| {category.category} | {category.task_count} | {category.mean_recall:.3f} "
            f"| {category.mean_precision:.3f} | {category.mean_f1_score:.3f} "
            f"| {category.mean_mrr:.3f} |"
        )
    lines.append("")
    lines.append("## Top Blocking Tasks")
    lines.append("")
    if not report.blocking_tasks:
        lines.append("No blocking tasks matched the triage thresholds.")
    else:
        lines.append("| Rank | Task | Category | Bucket | Recall | Precision | F1 |")
        lines.append("|---:|---|---|---|---:|---:|---:|")
        for index, finding in enumerate(report.blocking_tasks, start=1):
            lines.append(
                f"| {index} | `{finding.task_id}` | {finding.category} "
                f"| {finding.failure_bucket} | {finding.recall:.3f} "
                f"| {finding.precision:.3f} | {finding.f1_score:.3f} |"
            )
    return "\n".join(lines)


def format_readiness_json(report: ReadinessReport) -> str:
    """Render a readiness report as stable JSON."""
    return json.dumps(report.to_json(), indent=2, sort_keys=True)


def _find_result(report: BenchmarkReport, strategy: Strategy) -> BenchmarkResult | None:
    for result in report.results:
        if result.strategy == strategy:
            return result
    return None


def _category_readiness(
    results: list[tuple[BenchmarkReport, BenchmarkResult]],
    tasks_by_id: dict[str, BenchmarkTask],
) -> list[CategoryReadiness]:
    by_category: dict[str, list[BenchmarkResult]] = {}
    for report, result in results:
        category = _category(report, result, tasks_by_id)
        by_category.setdefault(category, []).append(result)

    categories: list[CategoryReadiness] = []
    for category, category_results in sorted(by_category.items()):
        categories.append(
            CategoryReadiness(
                category=category,
                task_count=len(category_results),
                mean_recall=_mean([result.recall for result in category_results]),
                mean_precision=_mean([result.precision for result in category_results]),
                mean_f1_score=_mean([result.f1_score for result in category_results]),
                mean_mrr=_mean([result.mrr for result in category_results]),
            )
        )
    return categories


def _category(
    report: BenchmarkReport,
    result: BenchmarkResult,
    tasks_by_id: dict[str, BenchmarkTask],
) -> str:
    if result.category is not None:
        return result.category.value
    task = tasks_by_id.get(report.task_id)
    if task is not None and task.category is not None:
        return task.category.value
    return "uncategorized"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"
