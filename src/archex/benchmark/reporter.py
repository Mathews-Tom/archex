"""Report generation for benchmark results: markdown tables, JSON, summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkReport, DeltaBenchmarkResult


def format_markdown(report: BenchmarkReport) -> str:
    """Render a single benchmark report as a markdown table."""
    lines: list[str] = []
    lines.append(f"## Benchmark: {report.task_id}")
    lines.append(f"**Repo:** {report.repo}")
    lines.append(f"**Question:** {report.question}")
    lines.append(f"**Baseline tokens:** {report.baseline_tokens:,}")
    lines.append("")
    header = "| Strategy | Tokens | Savings | Recall | Precision | F1 | MRR | Files | Time (ms) |"
    lines.append(header)
    lines.append(
        "|----------|--------|---------|--------|-----------|------|------|-------|-----------|"
    )  # noqa: E501
    for r in report.results:
        lines.append(
            f"| {r.strategy.value} | {r.tokens_total:,} | {r.savings_vs_raw:.1f}% "
            f"| {r.recall:.2f} | {r.precision:.2f} | {r.f1_score:.2f} | {r.mrr:.2f} "
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

    # Aggregate per-strategy averages
    strategy_totals: dict[str, list[float]] = {}
    strategy_recalls: dict[str, list[float]] = {}
    strategy_savings: dict[str, list[float]] = {}
    strategy_f1s: dict[str, list[float]] = {}
    strategy_mrrs: dict[str, list[float]] = {}

    for report in reports:
        for r in report.results:
            name = r.strategy.value
            strategy_totals.setdefault(name, []).append(float(r.tokens_total))
            strategy_recalls.setdefault(name, []).append(r.recall)
            strategy_savings.setdefault(name, []).append(r.savings_vs_raw)
            strategy_f1s.setdefault(name, []).append(r.f1_score)
            strategy_mrrs.setdefault(name, []).append(r.mrr)

    lines.append("| Strategy | Avg Tokens | Avg Savings | Avg Recall | Avg F1 | Avg MRR | Tasks |")
    lines.append("|----------|------------|-------------|------------|--------|---------|-------|")

    for name in sorted(strategy_totals.keys()):
        tokens_list = strategy_totals[name]
        recalls_list = strategy_recalls[name]
        savings_list = strategy_savings[name]
        f1s_list = strategy_f1s[name]
        mrrs_list = strategy_mrrs[name]
        count = len(tokens_list)
        avg_tokens = sum(tokens_list) / count
        avg_recall = sum(recalls_list) / count
        avg_savings = sum(savings_list) / count
        avg_f1 = sum(f1s_list) / count
        avg_mrr = sum(mrrs_list) / count
        lines.append(
            f"| {name} | {avg_tokens:,.0f} | {avg_savings:.1f}% "
            f"| {avg_recall:.2f} | {avg_f1:.2f} | {avg_mrr:.2f} | {count} |"
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
