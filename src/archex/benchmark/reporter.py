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
    "f1_score",
    "mrr",
    "ndcg",
    "map_score",
)
_BUCKET_FIELDS = (
    "recall",
    "precision",
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
    "f1_score",
    "mrr",
    "ndcg",
    "map_score",
    "token_efficiency",
)
_COMPARISON_LABELS = {
    "recall": "Recall",
    "precision": "Precision",
    "f1_score": "F1",
    "mrr": "MRR",
    "ndcg": "nDCG",
    "map_score": "MAP",
    "token_efficiency": "Efficiency",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


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
        "| Strategy | Tokens | Tokens In | Tokens Out | Efficiency | Savings "
        "| Recall | Precision | F1 | MRR | nDCG | MAP | Files | Time (ms) |"
    )
    lines.append(header)
    lines.append(
        "|----------|--------|-----------|------------|------------|--------"
        "|--------|-----------|------|------|------|------|-------|-----------|"
    )
    for r in report.results:
        lines.append(
            f"| {r.strategy.value} | {r.tokens_total:,} "
            f"| {r.tokens_input:,} | {r.tokens_output:,} | {r.token_efficiency:.2f} "
            f"| {r.savings_vs_raw:.1f}% "
            f"| {r.recall:.2f} | {r.precision:.2f} | {r.f1_score:.2f} | {r.mrr:.2f} "
            f"| {r.ndcg:.2f} | {r.map_score:.2f} "
            f"| {r.files_accessed} | {r.wall_time_ms:.0f} |"
        )
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
        "| Strategy | Avg Tokens | Avg Savings | Avg Efficiency | Avg Recall | Avg F1 "
        "| Avg MRR | Avg nDCG | Avg MAP | Tasks |"
    )
    lines.append(
        "|----------|------------|-------------|----------------|------------|--------"
        "|---------|----------|---------|-------|"
    )

    for name in sorted(strategy_metrics):
        metrics = strategy_metrics[name]
        count = len(metrics["tokens_total"])
        lines.append(
            f"| {name} | {_mean(metrics['tokens_total']):,.0f} "
            f"| {_mean(metrics['savings_vs_raw']):.1f}% "
            f"| {_mean(metrics['token_efficiency']):.2f} "
            f"| {_mean(metrics['recall']):.2f} | {_mean(metrics['f1_score']):.2f} "
            f"| {_mean(metrics['mrr']):.2f} | {_mean(metrics['ndcg']):.2f} "
            f"| {_mean(metrics['map_score']):.2f} | {count} |"
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
            "| Strategy | Recall | Precision | F1 | MRR | nDCG | MAP "
            "| Seed Recall | Seed Precision | Tasks |"
        )
        tbl.append(
            "|----------|--------|-----------|------|------|------|------"
            "|-------------|----------------|-------|"
        )
        for name in sorted(strategy_metrics):
            metrics = strategy_metrics[name]
            count = len(metrics["recall"])
            tbl.append(
                f"| {name} "
                f"| {_mean(metrics['recall']):.2f} | {_mean(metrics['precision']):.2f} "
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
            "| Strategy | Recall | Precision | F1 | MRR | nDCG | MAP | Tokens Total "
            "| Token Efficiency | Savings vs Raw |"
        )
        lines.append(
            "|----------|--------|-----------|------|------|------|------|-------------"
            "|------------------|---------------|"
        )
        for r in report.results:
            lines.append(
                f"| {r.strategy.value} | {r.recall:.2f} | {r.precision:.2f} "
                f"| {r.f1_score:.2f} | {r.mrr:.2f} | {r.ndcg:.2f} "
                f"| {r.map_score:.2f} | {r.tokens_total:,} "
                f"| {r.token_efficiency:.2f} | {r.savings_vs_raw:.1f}% |"
            )
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
