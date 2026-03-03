"""Report generation for benchmark results: markdown tables, JSON, summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkReport


def format_markdown(report: BenchmarkReport) -> str:
    """Render a single benchmark report as a markdown table."""
    lines: list[str] = []
    lines.append(f"## Benchmark: {report.task_id}")
    lines.append(f"**Repo:** {report.repo}")
    lines.append(f"**Question:** {report.question}")
    lines.append(f"**Baseline tokens:** {report.baseline_tokens:,}")
    lines.append("")
    lines.append("| Strategy | Tokens | Savings | Recall | Precision | Files | Time (ms) |")
    lines.append("|----------|--------|---------|--------|-----------|-------|-----------|")
    for r in report.results:
        lines.append(
            f"| {r.strategy.value} | {r.tokens_total:,} | {r.savings_vs_raw:.1f}% "
            f"| {r.recall:.2f} | {r.precision:.2f} | {r.files_accessed} "
            f"| {r.wall_time_ms:.0f} |"
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

    for report in reports:
        for r in report.results:
            name = r.strategy.value
            strategy_totals.setdefault(name, []).append(float(r.tokens_total))
            strategy_recalls.setdefault(name, []).append(r.recall)
            strategy_savings.setdefault(name, []).append(r.savings_vs_raw)

    lines.append("| Strategy | Avg Tokens | Avg Savings | Avg Recall | Tasks |")
    lines.append("|----------|------------|-------------|------------|-------|")

    for name in sorted(strategy_totals.keys()):
        tokens_list = strategy_totals[name]
        recalls_list = strategy_recalls[name]
        savings_list = strategy_savings[name]
        count = len(tokens_list)
        avg_tokens = sum(tokens_list) / count
        avg_recall = sum(recalls_list) / count
        avg_savings = sum(savings_list) / count
        lines.append(
            f"| {name} | {avg_tokens:,.0f} | {avg_savings:.1f}% | {avg_recall:.2f} | {count} |"
        )

    lines.append("")
    return "\n".join(lines)
