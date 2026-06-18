"""Report generation for benchmark results: markdown tables, JSON, summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkReport, DeltaBenchmarkResult


_SUMMARY_FIELDS = (
    "tokens_total",
    "savings_vs_raw",
    "token_efficiency",
    "recall",
    "required_file_recall",
    "missed_required_file_rate",
    "missed_required_task_rate",
    "f1_score",
    "mrr",
    "ndcg",
    "map_score",
)
_BUCKET_FIELDS = (
    "recall",
    "precision",
    "required_file_recall",
    "missed_required_file_rate",
    "missed_required_task_rate",
    "f1_score",
    "mrr",
    "ndcg",
    "map_score",
    "seed_recall",
    "seed_precision",
)
_COMPARISON_METRICS = (
    "recall",
    "precision",
    "required_file_recall",
    "f1_score",
    "mrr",
    "ndcg",
    "map_score",
    "token_efficiency",
)
_COMPARISON_LABELS = {
    "recall": "Recall",
    "precision": "Precision",
    "required_file_recall": "Required Recall",
    "f1_score": "F1",
    "mrr": "MRR",
    "ndcg": "nDCG",
    "map_score": "MAP",
    "token_efficiency": "Efficiency",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


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


def _receipt_accuracy_label(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _all_required_label(value: bool) -> str:
    return "yes" if value else "no"


def _aggregate_strategy_metrics(
    reports: list[BenchmarkReport],
    fields: tuple[str, ...],
) -> dict[str, dict[str, list[float]]]:
    strategy_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {field: [] for field in fields}
    )
    for report in reports:
        for result in report.results:
            metrics = strategy_metrics[result.strategy.value]
            for field in fields:
                metrics[field].append(float(getattr(result, field)))
    return strategy_metrics


def _strategy_win_counts(
    reports: list[BenchmarkReport],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    wins: dict[str, dict[str, int]] = {}
    for report in reports:
        for metric in metrics:
            best_result = max(report.results, key=lambda result: float(getattr(result, metric)))
            wins.setdefault(best_result.strategy.value, {}).setdefault(metric, 0)
            wins[best_result.strategy.value][metric] += 1
    return wins


def format_markdown(report: BenchmarkReport) -> str:
    """Render a single benchmark report as a markdown table."""
    lines: list[str] = []
    lines.append(f"## Benchmark: {report.task_id}")
    lines.append(f"**Repo:** {report.repo}")
    lines.append(f"**Question:** {report.question}")
    lines.append(f"**Baseline tokens:** {report.baseline_tokens:,}")
    lines.append("")
    header = (
        "| Strategy | Tokens | Required Recall | Missed File Rate | Missed Task Rate | "
        "All Required | Post Reads | Completion | Receipt Accuracy | Recall | Precision | "
        "F1 | Time (ms) |"
    )
    lines.append(header)
    lines.append(
        "|----------|--------|-----------------|------------------|------------------|"
        "--------------|------------|------------|------------------|--------|-----------|"
        "------|-----------|"
    )
    for r in report.results:
        receipt_accuracy = _receipt_accuracy_label(r.receipt_accuracy)
        all_required = _all_required_label(r.all_required_files_present)
        post_reads = r.post_bundle_read_turns if r.post_bundle_read_turns is not None else "—"
        lines.append(
            f"| {r.strategy.value} | {r.tokens_total:,} | {r.required_file_recall:.2f} "
            f"| {r.missed_required_file_rate:.2f} | {r.missed_required_task_rate:.2f} "
            f"| {all_required} | {post_reads} | {r.task_completion_result.value} "
            f"| {receipt_accuracy} | {r.recall:.2f} | {r.precision:.2f} "
            f"| {r.f1_score:.2f} | {r.wall_time_ms:.0f} |"
        )
    lines.extend(_missing_required_file_appendix(report))
    lines.append("")
    return "\n".join(lines)


def format_json(report: BenchmarkReport) -> str:
    """Render a benchmark report as pretty-printed JSON."""
    return report.model_dump_json(indent=2)


def format_summary(reports: list[BenchmarkReport]) -> str:
    """Render an aggregated cross-task summary."""
    if not reports:
        return "No benchmark results."

    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append(f"**Tasks:** {len(reports)}")
    lines.append("")

    strategy_metrics = _aggregate_strategy_metrics(reports, _SUMMARY_FIELDS)

    lines.append(
        "| Strategy | Avg Tokens | Avg Savings | Avg Efficiency | Avg Recall | "
        "Avg Required Recall | Missed File Rate | Missed Task Rate | Avg F1 | "
        "Avg MRR | Avg nDCG | Avg MAP | Tasks |"
    )
    lines.append(
        "|----------|------------|-------------|----------------|------------|"
        "---------------------|------------------|------------------|--------|---------|----------|---------|-------|"
    )

    for name in sorted(strategy_metrics):
        metrics = strategy_metrics[name]
        count = len(metrics["tokens_total"])
        lines.append(
            f"| {name} | {_mean(metrics['tokens_total']):,.0f} "
            f"| {_mean(metrics['savings_vs_raw']):.1f}% "
            f"| {_mean(metrics['token_efficiency']):.2f} | {_mean(metrics['recall']):.2f} "
            f"| {_mean(metrics['required_file_recall']):.2f} "
            f"| {_mean(metrics['missed_required_file_rate']):.2f} "
            f"| {_mean(metrics['missed_required_task_rate']):.2f} "
            f"| {_mean(metrics['f1_score']):.2f} | {_mean(metrics['mrr']):.2f} "
            f"| {_mean(metrics['ndcg']):.2f} | {_mean(metrics['map_score']):.2f} | {count} |"
        )

    lines.append("")
    return "\n".join(lines)


def format_bucketed_summary(reports: list[BenchmarkReport]) -> str:
    """Render per-category aggregated summaries alongside the global summary.

    Groups tasks by their category (from task YAML) and produces a table
    per category plus a global table, preventing weak categories from hiding
    in overall averages.
    """
    if not reports:
        return "No benchmark results."

    lines: list[str] = []
    lines.append("# Bucketed Benchmark Summary")
    lines.append(f"**Tasks:** {len(reports)}")
    lines.append("")

    # Group reports by category derived from result entries
    buckets: dict[str, list[BenchmarkReport]] = defaultdict(list)
    for report in reports:
        # Derive category from the first archex result that has one,
        # or fall back to "uncategorized"
        cat = "uncategorized"
        for r in report.results:
            if r.category is not None:
                cat = r.category.value
                break
        buckets[cat].append(report)

    def _summary_table(label: str, bucket_reports: list[BenchmarkReport]) -> list[str]:
        tbl: list[str] = []
        tbl.append(f"## {label} ({len(bucket_reports)} tasks)")
        tbl.append("")

        strategy_metrics = _aggregate_strategy_metrics(bucket_reports, _BUCKET_FIELDS)

        tbl.append(
            "| Strategy | Recall | Precision | Required Recall | Missed File Rate "
            "| Missed Task Rate | F1 | MRR | nDCG | MAP | Seed Recall "
            "| Seed Precision | Tasks |"
        )
        tbl.append(
            "|----------|--------|-----------|-----------------|------------------|------------------|------|------|------|------"
            "|-------------|----------------|-------|"
        )
        for name in sorted(strategy_metrics):
            metrics = strategy_metrics[name]
            count = len(metrics["recall"])
            tbl.append(
                f"| {name} "
                f"| {_mean(metrics['recall']):.2f} | {_mean(metrics['precision']):.2f} "
                f"| {_mean(metrics['required_file_recall']):.2f} "
                f"| {_mean(metrics['missed_required_file_rate']):.2f} "
                f"| {_mean(metrics['missed_required_task_rate']):.2f} "
                f"| {_mean(metrics['f1_score']):.2f} | {_mean(metrics['mrr']):.2f} "
                f"| {_mean(metrics['ndcg']):.2f} | {_mean(metrics['map_score']):.2f} "
                f"| {_mean(metrics['seed_recall']):.2f} "
                f"| {_mean(metrics['seed_precision']):.2f} "
                f"| {count} |"
            )
        tbl.append("")
        return tbl

    # Global summary first
    lines.extend(_summary_table("All Tasks", reports))

    # Per-bucket summaries
    for cat in sorted(buckets.keys()):
        lines.extend(_summary_table(cat, buckets[cat]))

    return "\n".join(lines)


def format_strategy_comparison(reports: list[BenchmarkReport]) -> str:
    """Render a per-task strategy head-to-head comparison."""
    if not reports:
        return "No benchmark results."

    lines: list[str] = []
    lines.append("# Strategy Comparison")
    lines.append("")

    # Per-task tables
    for report in reports:
        lines.append(f"## {report.task_id}")
        lines.append("")
        lines.append(
            "| Strategy | Required Recall | Missed File Rate | Missed Task Rate | "
            "All Required | Completion | Receipt Accuracy | Recall | Precision | F1 | "
            "Tokens Total |"
        )
        lines.append(
            "|----------|-----------------|------------------|------------------|"
            "--------------|------------|------------------|--------|-----------|------|"
            "-------------|"
        )
        for r in report.results:
            receipt_accuracy = _receipt_accuracy_label(r.receipt_accuracy)
            all_required = _all_required_label(r.all_required_files_present)
            lines.append(
                f"| {r.strategy.value} | {r.required_file_recall:.2f} "
                f"| {r.missed_required_file_rate:.2f} | {r.missed_required_task_rate:.2f} "
                f"| {all_required} | {r.task_completion_result.value} | {receipt_accuracy} "
                f"| {r.recall:.2f} | {r.precision:.2f} | {r.f1_score:.2f} "
                f"| {r.tokens_total:,} |"
            )
        lines.extend(_missing_required_file_appendix(report))
        lines.append("")

    # Head-to-head wins
    wins = _strategy_win_counts(reports, _COMPARISON_METRICS)

    lines.append("## Head-to-Head Wins")
    lines.append("")
    metric_headers = " | ".join(_COMPARISON_LABELS[m] for m in _COMPARISON_METRICS)
    lines.append(f"| Strategy | {metric_headers} | Total |")
    sep = " | ".join("------" for _ in _COMPARISON_METRICS)
    lines.append(f"|----------|{sep}|-------|")

    all_strategies = sorted({r.strategy.value for report in reports for r in report.results})
    for strategy in all_strategies:
        strat_wins = wins.get(strategy, {})
        counts = [str(strat_wins.get(metric, 0)) for metric in _COMPARISON_METRICS]
        total = sum(strat_wins.get(metric, 0) for metric in _COMPARISON_METRICS)
        lines.append(f"| {strategy} | {' | '.join(counts)} | {total} |")
    lines.append("")

    # Best strategy per metric
    lines.append("## Best Strategy per Metric")
    lines.append("")
    for metric in _COMPARISON_METRICS:
        best_count = 0
        best_strategy = ""
        for strategy in all_strategies:
            count = wins.get(strategy, {}).get(metric, 0)
            if count > best_count:
                best_count = count
                best_strategy = strategy
        label = _COMPARISON_LABELS[metric]
        lines.append(f"- **{label}**: {best_strategy} ({best_count} wins)")
    lines.append("")

    return "\n".join(lines)


def _missing_required_file_appendix(report: BenchmarkReport) -> list[str]:
    lines: list[str] = []
    failures = [result for result in report.results if result.required_files_missing]
    if not failures:
        return lines
    lines.extend(["", "### Missing required files appendix"])
    for result in failures:
        lines.append(
            f"- {result.strategy.value}: missing {', '.join(result.required_files_missing)}"
        )
    return lines


def format_chunker_frontier_table(
    candidate_reports: list[BenchmarkReport],
    baseline_reports: list[BenchmarkReport],
) -> str:
    """Render a default-vs-candidate chunker frontier table."""
    if not candidate_reports or not baseline_reports:
        return "No chunker frontier comparison available."

    rows: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {
            "recall": [],
            "precision": [],
            "f1_score": [],
            "token_efficiency": [],
            "wall_time_ms": [],
            "index_chunk_count": [],
            "mean_chunk_tokens": [],
        }
    )
    for report in [*baseline_reports, *candidate_reports]:
        for result in report.results:
            key = (result.strategy.value, result.chunker)
            rows[key]["recall"].append(result.recall)
            rows[key]["precision"].append(result.precision)
            rows[key]["f1_score"].append(result.f1_score)
            rows[key]["token_efficiency"].append(result.token_efficiency)
            rows[key]["wall_time_ms"].append(result.wall_time_ms)
            rows[key]["index_chunk_count"].append(float(result.index_chunk_count))
            rows[key]["mean_chunk_tokens"].append(result.mean_chunk_tokens)

    lines = [
        "## Chunker Frontier Comparison",
        "",
        "| Strategy | Chunker | Recall | Precision | F1 | Token Efficiency | p95 ms "
        "| Chunk Count | Mean Chunk Tokens |",
        "|----------|---------|--------|-----------|----|------------------|--------|-------------|-------------------|",
    ]
    for strategy, chunker in sorted(rows):
        metrics = rows[(strategy, chunker)]
        lines.append(
            f"| {strategy} | {chunker} | {_mean(metrics['recall']):.3f} "
            f"| {_mean(metrics['precision']):.3f} | {_mean(metrics['f1_score']):.3f} "
            f"| {_mean(metrics['token_efficiency']):.3f} "
            f"| {_percentile(metrics['wall_time_ms'], 0.95):.0f} "
            f"| {_mean(metrics['index_chunk_count']):.0f} "
            f"| {_mean(metrics['mean_chunk_tokens']):.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_delta_summary(results: list[DeltaBenchmarkResult]) -> str:
    """Render a markdown summary table for delta benchmark results."""
    if not results:
        return "No delta benchmark results."

    lines: list[str] = []
    lines.append("# Delta Benchmark Summary")
    lines.append(f"**Tasks:** {len(results)}")
    lines.append("")
    lines.append(
        "| Task | Delta Files | Total Files | Delta % | Delta (ms) "
        "| Full (ms) | Speedup | Correct | Chunks Updated | Chunks Unchanged |"
    )
    lines.append(
        "|------|-------------|-------------|---------|------------"
        "|-----------|---------|---------|----------------|------------------|"
    )

    for r in results:
        correct_str = "yes" if r.correctness else "NO"
        lines.append(
            f"| {r.task_id} | {r.delta_files} | {r.total_files} | {r.delta_pct:.1f}% "
            f"| {r.delta_time_ms:.0f} | {r.full_reindex_time_ms:.0f} "
            f"| {r.speedup_factor:.1f}x | {correct_str} "
            f"| {r.chunks_updated} | {r.chunks_unchanged} |"
        )

    lines.append("")
    return "\n".join(lines)
