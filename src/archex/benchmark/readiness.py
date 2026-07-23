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
TARGET_REQUIRED_FILE_RECALL = 0.80
TARGET_MISSED_REQUIRED_TASK_RATE = 0.00
TARGET_RECEIPT_ACCURACY = 0.95
TARGET_TOKEN_EFFICIENCY_WITH_COMPLETION = 0.08


def _receipt_accuracy_score(value: bool | None) -> float:
    return 0.0 if value is None else 1.0 if value else 0.0


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return (sum(present) / len(present)) if present else None


def _format_optional(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f}"


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
    tokens_total: int
    token_efficiency: float
    token_efficiency_with_completion: float
    savings_vs_raw: float
    mean_required_file_recall: float
    mean_missed_required_file_rate: float
    missed_required_task_rate: float
    all_required_present_rate: float
    receipt_accuracy_rate: float
    zero_recall_tasks: int
    low_f1_tasks: int
    low_precision_tasks: int
    targets: list[ReadinessTargetStatus]
    categories: list[CategoryReadiness]
    blocking_tasks: list[TriageFinding]
    # Optional region/line-level retrieval-quality means over region-labelled
    # tasks for this strategy; None when no labelled tasks ran.
    mean_region_recall: float | None = None
    mean_region_precision: float | None = None
    mean_line_recall: float | None = None
    mean_ranked_region_ndcg: float | None = None
    mean_context_noise_ratio: float | None = None
    mean_relevance_per_1k_tokens: float | None = None

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
            "tokens_total": self.tokens_total,
            "token_efficiency": self.token_efficiency,
            "token_efficiency_with_completion": self.token_efficiency_with_completion,
            "savings_vs_raw": self.savings_vs_raw,
            "mean_required_file_recall": self.mean_required_file_recall,
            "mean_missed_required_file_rate": self.mean_missed_required_file_rate,
            "missed_required_task_rate": self.missed_required_task_rate,
            "all_required_present_rate": self.all_required_present_rate,
            "receipt_accuracy_rate": self.receipt_accuracy_rate,
            "zero_recall_tasks": self.zero_recall_tasks,
            "low_f1_tasks": self.low_f1_tasks,
            "low_precision_tasks": self.low_precision_tasks,
            "mean_region_recall": self.mean_region_recall,
            "mean_region_precision": self.mean_region_precision,
            "mean_line_recall": self.mean_line_recall,
            "mean_ranked_region_ndcg": self.mean_ranked_region_ndcg,
            "mean_context_noise_ratio": self.mean_context_noise_ratio,
            "mean_relevance_per_1k_tokens": self.mean_relevance_per_1k_tokens,
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
            tokens_total=0,
            token_efficiency=0.0,
            token_efficiency_with_completion=0.0,
            savings_vs_raw=0.0,
            mean_required_file_recall=0.0,
            mean_missed_required_file_rate=0.0,
            missed_required_task_rate=0.0,
            all_required_present_rate=0.0,
            receipt_accuracy_rate=0.0,
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
    median_latency_ms = _percentile(
        [result.wall_time_ms for result in metric_results if result.wall_time_ms is not None],
        0.50,
    )
    p95_latency_ms = _percentile(
        [result.wall_time_ms for result in metric_results if result.wall_time_ms is not None],
        0.95,
    )
    tokens_total = sum(result.tokens_total for result in metric_results)
    token_efficiency = _mean([result.token_efficiency for result in metric_results])
    savings_vs_raw = _mean([result.savings_vs_raw for result in metric_results])
    token_efficiency_with_completion = _mean(
        [result.token_efficiency_with_completion for result in metric_results]
    )
    mean_required_file_recall = _mean([result.required_file_recall for result in metric_results])
    mean_missed_required_file_rate = _mean(
        [result.missed_required_file_rate for result in metric_results]
    )
    missed_required_task_rate = _mean(
        [result.missed_required_task_rate for result in metric_results]
    )
    all_required_present_rate = _mean(
        [1.0 if result.all_required_files_present else 0.0 for result in metric_results]
    )
    receipt_accuracy_rate = _mean(
        [_receipt_accuracy_score(result.receipt_accuracy) for result in metric_results]
    )
    zero_recall_tasks = sum(1 for result in metric_results if result.recall <= 0.0)
    low_f1_tasks = sum(1 for result in metric_results if result.f1_score < LOW_F1_THRESHOLD)
    low_precision_tasks = sum(
        1 for result in metric_results if result.precision < LOW_PRECISION_THRESHOLD
    )
    region_results = [result for result in metric_results if result.region_recall is not None]
    mean_region_recall = _mean_optional([result.region_recall for result in region_results])
    mean_region_precision = _mean_optional([result.region_precision for result in region_results])
    mean_line_recall = _mean_optional([result.line_recall for result in region_results])
    mean_ranked_region_ndcg = _mean_optional(
        [result.ranked_region_ndcg for result in region_results]
    )
    mean_context_noise_ratio = _mean_optional(
        [result.context_noise_ratio for result in region_results]
    )
    mean_relevance_per_1k_tokens = _mean_optional(
        [result.relevance_per_1k_tokens for result in region_results]
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
        ReadinessTargetStatus(
            name="mean_required_file_recall",
            actual=mean_required_file_recall,
            target=TARGET_REQUIRED_FILE_RECALL,
            passed=mean_required_file_recall >= TARGET_REQUIRED_FILE_RECALL,
            comparator=">=",
        ),
        ReadinessTargetStatus(
            name="missed_required_task_rate",
            actual=missed_required_task_rate,
            target=TARGET_MISSED_REQUIRED_TASK_RATE,
            passed=missed_required_task_rate <= TARGET_MISSED_REQUIRED_TASK_RATE,
            comparator="<=",
        ),
        ReadinessTargetStatus(
            name="receipt_accuracy_rate",
            actual=receipt_accuracy_rate,
            target=TARGET_RECEIPT_ACCURACY,
            passed=receipt_accuracy_rate >= TARGET_RECEIPT_ACCURACY,
            comparator=">=",
        ),
        ReadinessTargetStatus(
            name="token_efficiency_with_completion",
            actual=token_efficiency_with_completion,
            target=TARGET_TOKEN_EFFICIENCY_WITH_COMPLETION,
            passed=token_efficiency_with_completion >= TARGET_TOKEN_EFFICIENCY_WITH_COMPLETION,
            comparator=">=",
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
        tokens_total=tokens_total,
        token_efficiency=token_efficiency,
        savings_vs_raw=savings_vs_raw,
        token_efficiency_with_completion=token_efficiency_with_completion,
        zero_recall_tasks=zero_recall_tasks,
        mean_required_file_recall=mean_required_file_recall,
        mean_missed_required_file_rate=mean_missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        all_required_present_rate=all_required_present_rate,
        receipt_accuracy_rate=receipt_accuracy_rate,
        low_f1_tasks=low_f1_tasks,
        low_precision_tasks=low_precision_tasks,
        targets=targets,
        categories=categories,
        blocking_tasks=blocking_tasks,
        mean_region_recall=mean_region_recall,
        mean_region_precision=mean_region_precision,
        mean_line_recall=mean_line_recall,
        mean_ranked_region_ndcg=mean_ranked_region_ndcg,
        mean_context_noise_ratio=mean_context_noise_ratio,
        mean_relevance_per_1k_tokens=mean_relevance_per_1k_tokens,
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
    lines.append("## Strategy Metrics")
    lines.append("")
    lines.append(
        "| Strategy | Tasks | Recall | Required Recall | Missed Task Rate "
        "| Receipt Accuracy | F1 | Tokens Total | Token Efficiency "
        "| Efficiency after Completion | Savings vs Raw | Median Latency | P95 Latency |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| `{report.strategy}` | {report.task_count} | {report.mean_recall:.3f} "
        f"| {report.mean_required_file_recall:.3f} "
        f"| {report.missed_required_task_rate:.3f} "
        f"| {report.receipt_accuracy_rate:.3f} "
        f"| {report.mean_f1_score:.3f} | {report.tokens_total:,} "
        f"| {report.token_efficiency:.3f} "
        f"| {report.token_efficiency_with_completion:.3f} "
        f"| {report.savings_vs_raw:.1f}% "
        f"| {report.median_latency_ms:.0f} ms | {report.p95_latency_ms:.0f} ms |"
    )
    if report.mean_region_recall is not None:
        lines.append("")
        lines.append("## Region & Context Efficiency")
        lines.append("")
        lines.append(
            f"- Mean region recall: `{_format_optional(report.mean_region_recall)}`\n"
            f"- Mean region precision: `{_format_optional(report.mean_region_precision)}`\n"
            f"- Mean line recall: `{_format_optional(report.mean_line_recall)}`\n"
            f"- Mean ranked region nDCG: `{_format_optional(report.mean_ranked_region_ndcg)}`\n"
            f"- Mean context noise ratio: `{_format_optional(report.mean_context_noise_ratio)}`\n"
            "- Mean relevance per 1k tokens: "
            f"`{_format_optional(report.mean_relevance_per_1k_tokens)}`"
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
        f"- Tokens total: `{report.tokens_total:,}`\n"
        f"- Token efficiency: `{report.token_efficiency:.3f}`\n"
        f"- Token efficiency after completion: `{report.token_efficiency_with_completion:.3f}`\n"
        f"- Required-file recall: `{report.mean_required_file_recall:.3f}`\n"
        f"- Missed required-file rate: `{report.mean_missed_required_file_rate:.3f}`\n"
        f"- Missed required-task rate: `{report.missed_required_task_rate:.3f}`\n"
        f"- All-required-present rate: `{report.all_required_present_rate:.3f}`\n"
        f"- Receipt accuracy: `{report.receipt_accuracy_rate:.3f}`\n"
        f"- Savings vs raw: `{report.savings_vs_raw:.1f}%`\n"
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
        lines.append(
            "| Rank | Task | Category | Bucket | Recall | Precision | F1 | Expansion Reasons |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---|")
        for index, finding in enumerate(report.blocking_tasks, start=1):
            reason_summary = ", ".join(
                f"{reason}:{count}"
                for reason, count in sorted(finding.expansion_reason_counts.items())
            )
            lines.append(
                f"| {index} | `{finding.task_id}` | {finding.category} "
                f"| {finding.failure_bucket} | {finding.recall:.3f} "
                f"| {finding.precision:.3f} | {finding.f1_score:.3f} "
                f"| {reason_summary or 'none'} |"
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
