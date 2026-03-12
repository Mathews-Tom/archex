"""Report generation for benchmark results: markdown tables, JSON, summaries."""

from __future__ import annotations

from collections import defaultdict
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

    # Aggregate per-strategy averages
    strategy_totals: dict[str, list[float]] = {}
    strategy_recalls: dict[str, list[float]] = {}
    strategy_savings: dict[str, list[float]] = {}
    strategy_f1s: dict[str, list[float]] = {}
    strategy_mrrs: dict[str, list[float]] = {}
    strategy_ndcgs: dict[str, list[float]] = {}
    strategy_maps: dict[str, list[float]] = {}
    strategy_efficiencies: dict[str, list[float]] = {}

    for report in reports:
        for r in report.results:
            name = r.strategy.value
            strategy_totals.setdefault(name, []).append(float(r.tokens_total))
            strategy_recalls.setdefault(name, []).append(r.recall)
            strategy_savings.setdefault(name, []).append(r.savings_vs_raw)
            strategy_f1s.setdefault(name, []).append(r.f1_score)
            strategy_mrrs.setdefault(name, []).append(r.mrr)
            strategy_ndcgs.setdefault(name, []).append(r.ndcg)
            strategy_maps.setdefault(name, []).append(r.map_score)
            strategy_efficiencies.setdefault(name, []).append(r.token_efficiency)

    lines.append(
        "| Strategy | Avg Tokens | Avg Savings | Avg Efficiency | Avg Recall | Avg F1 "
        "| Avg MRR | Avg nDCG | Avg MAP | Tasks |"
    )
    lines.append(
        "|----------|------------|-------------|----------------|------------|--------"
        "|---------|----------|---------|-------|"
    )

    for name in sorted(strategy_totals.keys()):
        tokens_list = strategy_totals[name]
        recalls_list = strategy_recalls[name]
        savings_list = strategy_savings[name]
        f1s_list = strategy_f1s[name]
        mrrs_list = strategy_mrrs[name]
        ndcgs_list = strategy_ndcgs[name]
        maps_list = strategy_maps[name]
        effs_list = strategy_efficiencies[name]
        count = len(tokens_list)
        avg_tokens = sum(tokens_list) / count
        avg_recall = sum(recalls_list) / count
        avg_savings = sum(savings_list) / count
        avg_f1 = sum(f1s_list) / count
        avg_mrr = sum(mrrs_list) / count
        avg_ndcg = sum(ndcgs_list) / count
        avg_map = sum(maps_list) / count
        avg_eff = sum(effs_list) / count
        lines.append(
            f"| {name} | {avg_tokens:,.0f} | {avg_savings:.1f}% | {avg_eff:.2f} "
            f"| {avg_recall:.2f} | {avg_f1:.2f} | {avg_mrr:.2f} "
            f"| {avg_ndcg:.2f} | {avg_map:.2f} | {count} |"
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

        strategy_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        for report in bucket_reports:
            for r in report.results:
                strategy_metrics[r.strategy.value].append(
                    {
                        "recall": r.recall,
                        "precision": r.precision,
                        "f1": r.f1_score,
                        "mrr": r.mrr,
                        "ndcg": r.ndcg,
                        "map": r.map_score,
                        "seed_recall": r.seed_recall,
                        "seed_precision": r.seed_precision,
                    }
                )

        tbl.append(
            "| Strategy | Recall | Precision | F1 | MRR | nDCG | MAP "
            "| Seed Recall | Seed Precision | Tasks |"
        )
        tbl.append(
            "|----------|--------|-----------|------|------|------|------"
            "|-------------|----------------|-------|"
        )
        for name in sorted(strategy_metrics.keys()):
            entries = strategy_metrics[name]
            count = len(entries)

            def _avg(
                key: str,
                _entries: list[dict[str, float]] = entries,
                _count: int = count,
            ) -> float:
                return sum(e[key] for e in _entries) / _count

            tbl.append(
                f"| {name} "
                f"| {_avg('recall'):.2f} | {_avg('precision'):.2f} "
                f"| {_avg('f1'):.2f} | {_avg('mrr'):.2f} "
                f"| {_avg('ndcg'):.2f} | {_avg('map'):.2f} "
                f"| {_avg('seed_recall'):.2f} | {_avg('seed_precision'):.2f} "
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

    metrics = ("recall", "precision", "f1_score", "mrr", "ndcg", "map_score", "token_efficiency")
    metric_labels = {
        "recall": "Recall",
        "precision": "Precision",
        "f1_score": "F1",
        "mrr": "MRR",
        "ndcg": "nDCG",
        "map_score": "MAP",
        "token_efficiency": "Efficiency",
    }

    # Per-task tables
    for report in reports:
        lines.append(f"## {report.task_id}")
        lines.append("")
        lines.append("| Strategy | Recall | Precision | F1 | MRR | nDCG | MAP | Tokens | Savings |")
        lines.append(
            "|----------|--------|-----------|------|------|------|------|--------|---------|"
        )
        for r in report.results:
            lines.append(
                f"| {r.strategy.value} | {r.recall:.2f} | {r.precision:.2f} "
                f"| {r.f1_score:.2f} | {r.mrr:.2f} | {r.ndcg:.2f} "
                f"| {r.map_score:.2f} "
                f"| {r.tokens_total:,} | {r.savings_vs_raw:.1f}% |"
            )
        lines.append("")

    # Head-to-head wins
    wins: dict[str, dict[str, int]] = {}
    for report in reports:
        for metric in metrics:
            best_val = -1.0
            best_strategy = ""
            for r in report.results:
                val = getattr(r, metric)
                if val > best_val:
                    best_val = val
                    best_strategy = r.strategy.value
            if best_strategy:
                wins.setdefault(best_strategy, {}).setdefault(metric, 0)
                wins[best_strategy][metric] += 1

    lines.append("## Head-to-Head Wins")
    lines.append("")
    metric_headers = " | ".join(metric_labels[m] for m in metrics)
    lines.append(f"| Strategy | {metric_headers} | Total |")
    sep = " | ".join("------" for _ in metrics)
    lines.append(f"|----------|{sep}|-------|")

    all_strategies = sorted({r.strategy.value for report in reports for r in report.results})
    for strategy in all_strategies:
        strat_wins = wins.get(strategy, {})
        counts = [str(strat_wins.get(m, 0)) for m in metrics]
        total = sum(strat_wins.get(m, 0) for m in metrics)
        lines.append(f"| {strategy} | {' | '.join(counts)} | {total} |")
    lines.append("")

    # Best strategy per metric
    lines.append("## Best Strategy per Metric")
    lines.append("")
    for metric in metrics:
        best_count = 0
        best_strategy = ""
        for strategy in all_strategies:
            count = wins.get(strategy, {}).get(metric, 0)
            if count > best_count:
                best_count = count
                best_strategy = strategy
        label = metric_labels[metric]
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
