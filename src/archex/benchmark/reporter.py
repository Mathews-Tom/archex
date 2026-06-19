"""Report generation for benchmark results: markdown tables, JSON, summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkReport, BenchmarkResult, DeltaBenchmarkResult


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


def _has_region_metrics(reports: list[BenchmarkReport]) -> bool:
    return any(result.region_recall is not None for report in reports for result in report.results)


def _fmt_opt(value: float | None, *, digits: int = 3) -> str:
    return "unknown" if value is None else f"{value:.{digits}f}"


def _fmt_opt_int(value: int | None) -> str:
    return "unknown" if value is None else f"{value:,}"


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return (sum(present) / len(present)) if present else None


def _region_quality_appendix(report: BenchmarkReport) -> list[str]:
    """Per-task region + context-efficiency table, rendered only when labelled."""
    rows = [result for result in report.results if result.region_recall is not None]
    if not rows:
        return []
    lines = ["", "### Region & Context Efficiency", ""]
    lines.append(
        "| Strategy | Region Recall | Region Prec | Region F1 | Line Recall | Line Prec "
        "| Ranked MRR | Ranked nDCG | Noise Ratio | Useful Tokens | Wasted Tokens | Rel/1k |"
    )
    lines.append(
        "|----------|--------------:|------------:|----------:|------------:|----------:"
        "|-----------:|------------:|------------:|--------------:|--------------:|-------:|"
    )
    for result in rows:
        lines.append(
            f"| {result.strategy.value} | {_fmt_opt(result.region_recall)} "
            f"| {_fmt_opt(result.region_precision)} | {_fmt_opt(result.region_f1)} "
            f"| {_fmt_opt(result.line_recall)} | {_fmt_opt(result.line_precision)} "
            f"| {_fmt_opt(result.ranked_region_mrr)} | {_fmt_opt(result.ranked_region_ndcg)} "
            f"| {_fmt_opt(result.context_noise_ratio)} | {_fmt_opt_int(result.useful_tokens)} "
            f"| {_fmt_opt_int(result.wasted_tokens)} "
            f"| {_fmt_opt(result.relevance_per_1k_tokens, digits=2)} |"
        )
    return lines


def _region_quality_summary(reports: list[BenchmarkReport]) -> list[str]:
    """Aggregated region + context-efficiency table, rendered only when labelled."""
    if not _has_region_metrics(reports):
        return []
    by_strategy: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for report in reports:
        for result in report.results:
            if result.region_recall is not None:
                by_strategy[result.strategy.value].append(result)
    lines = ["## Region & Context Efficiency", ""]
    lines.append(
        "| Strategy | Region Recall | Region F1 | Line Recall | Ranked nDCG "
        "| Noise Ratio | Rel/1k | Tasks |"
    )
    lines.append(
        "|----------|--------------:|----------:|------------:|------------:"
        "|------------:|-------:|------:|"
    )
    for name in sorted(by_strategy):
        results = by_strategy[name]
        lines.append(
            f"| {name} "
            f"| {_fmt_opt(_mean_optional([r.region_recall for r in results]))} "
            f"| {_fmt_opt(_mean_optional([r.region_f1 for r in results]))} "
            f"| {_fmt_opt(_mean_optional([r.line_recall for r in results]))} "
            f"| {_fmt_opt(_mean_optional([r.ranked_region_ndcg for r in results]))} "
            f"| {_fmt_opt(_mean_optional([r.context_noise_ratio for r in results]))} "
            f"| {_fmt_opt(_mean_optional([r.relevance_per_1k_tokens for r in results]), digits=2)} "
            f"| {len(results)} |"
        )
    lines.append("")
    return lines


def _sum_optional_int(values: list[int | None]) -> int:
    return sum(value for value in values if value is not None)


def _compression_appendix(report: BenchmarkReport) -> list[str]:
    """Per-task post-retrieval compression table, rendered only when present."""
    rows = [result for result in report.results if result.bundle_compression_ratio is not None]
    if not rows:
        return []
    lines = ["", "### Post-Retrieval Compression", ""]
    lines.append(
        "| Strategy | Compression Ratio | Uncompressed | Compressed | Passthrough Required "
        "| Hidden Required | Efficiency (compressed+completion) |"
    )
    lines.append(
        "|----------|------------------:|-------------:|-----------:|---------------------:"
        "|----------------:|----------------------------------:|"
    )
    for result in rows:
        lines.append(
            f"| {result.strategy.value} | {_fmt_opt(result.bundle_compression_ratio)} "
            f"| {_fmt_opt_int(result.bundle_tokens_uncompressed)} "
            f"| {_fmt_opt_int(result.bundle_tokens_compressed)} "
            f"| {_fmt_opt_int(result.required_context_passthrough_tokens)} "
            f"| {_fmt_opt_int(result.compression_hidden_required_region_count)} "
            f"| {_fmt_opt(result.token_efficiency_with_compression_and_completion)} |"
        )
    return lines


def _compression_summary(reports: list[BenchmarkReport]) -> list[str]:
    """Aggregated post-retrieval compression table, rendered only when present."""
    by_strategy: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for report in reports:
        for result in report.results:
            if result.bundle_compression_ratio is not None:
                by_strategy[result.strategy.value].append(result)
    if not by_strategy:
        return []
    lines = ["## Post-Retrieval Compression", ""]
    lines.append(
        "Compression ratio is compressed/uncompressed bundle tokens (lower means more "
        "compression). Hidden Required counts required regions hidden by compression and must "
        "stay 0; retrieval misses are reported separately above. Compression cannot make "
        "incomplete context complete."
    )
    lines.append("")
    lines.append(
        "| Strategy | Compression Ratio | Passthrough Tokens | Hidden Required "
        "| Efficiency (compressed+completion) | Tasks |"
    )
    lines.append(
        "|----------|------------------:|-------------------:|----------------:"
        "|----------------------------------:|------:|"
    )
    for name in sorted(by_strategy):
        results = by_strategy[name]
        ratio = _mean_optional([r.bundle_compression_ratio for r in results])
        efficiency = _mean_optional(
            [r.token_efficiency_with_compression_and_completion for r in results]
        )
        passthrough = _sum_optional_int([r.required_context_passthrough_tokens for r in results])
        hidden = _sum_optional_int([r.compression_hidden_required_region_count for r in results])
        lines.append(
            f"| {name} | {_fmt_opt(ratio)} | {passthrough:,} | {hidden} "
            f"| {_fmt_opt(efficiency)} | {len(results)} |"
        )
    lines.append("")
    return lines


def _packing_appendix(report: BenchmarkReport) -> list[str]:
    """Per-task efficiency-aware packing provenance, rendered only when present."""
    rows = [
        result for result in report.results if result.packed_relevance_per_1k_tokens is not None
    ]
    if not rows:
        return []
    lines = ["", "### Efficiency-Aware Packing", ""]
    lines.append(
        "| Strategy | Budget | Include | Compress | Elide | Skip | Rel/1k | Compression Ratio |"
    )
    lines.append(
        "|----------|--------|--------:|---------:|------:|-----:|-------:|------------------:|"
    )
    for result in rows:
        prov = result.provenance
        lines.append(
            f"| {result.strategy.value} | {prov.get('budget_tier', 'unknown')} "
            f"| {_fmt_opt_int(result.packing_included_regions)} "
            f"| {_fmt_opt_int(result.packing_compressed_regions)} "
            f"| {_fmt_opt_int(result.packing_elided_regions)} "
            f"| {_fmt_opt_int(result.packing_skipped_regions)} "
            f"| {_fmt_opt(result.packed_relevance_per_1k_tokens, digits=2)} "
            f"| {_fmt_opt(result.bundle_compression_ratio)} |"
        )
    return lines


_ADVANCED_LANES = (
    "archex_query_dual_transform",
    "archex_query_bounded_rerank",
    "archex_query_summary_sidecar",
    "archex_query_graph_multihop",
)


def _advanced_lane_note(result: BenchmarkResult) -> str:
    """One-line provenance highlight per advanced lane."""
    prov = result.provenance
    strategy = result.strategy.value
    if strategy == "archex_query_dual_transform":
        included = prov.get("fused_included", "unknown")
        return f"fused {included}/{prov.get('fused_candidates', 'unknown')} candidates"
    if strategy == "archex_query_bounded_rerank":
        return (
            f"reranked {prov.get('candidates_reranked', 'unknown')}/"
            f"{prov.get('candidates_total', 'unknown')} "
            f"(ce={prov.get('cross_encoder_status', 'unknown')})"
        )
    if strategy == "archex_query_summary_sidecar":
        return (
            f"summary_first={prov.get('summary_first', 'unknown')}, "
            f"stale={prov.get('entries_stale', 'unknown')}"
        )
    if strategy == "archex_query_graph_multihop":
        return (
            f"expanded {prov.get('files_expanded', 'unknown')} "
            f"(cuts frontier/budget/low-conf="
            f"{prov.get('frontier_cuts', 'unknown')}/{prov.get('budget_cuts', 'unknown')}/"
            f"{prov.get('suppressed_low_confidence', 'unknown')})"
        )
    return ""


def _advanced_lanes_appendix(report: BenchmarkReport) -> list[str]:
    """Per-task advanced-lane comparison: latency, token impact, and quality.

    Rendered only when an advanced lane ran. These lanes are benchmark-only
    experiments (Workstream 5); none changes the product default.
    """
    rows = [result for result in report.results if result.strategy.value in _ADVANCED_LANES]
    if not rows:
        return []
    lines = ["", "### Advanced Quality Lanes", ""]
    lines.append(
        "Benchmark-only experimental lanes (Workstream 5): query transformation, bounded "
        "rerank, summary sidecars, and graph multihop. Each is gated behind the default "
        "switch rule and never changes the product default. Compare added latency and token "
        "impact against `archex_query` in the table above; the summary-sidecar lane also "
        "carries an offline storage/index cost (the prebuilt sidecar)."
    )
    lines.append("")
    lines.append(
        "| Strategy | Tokens | Token Eff (compl.) | Required Recall | Region Recall "
        "| Latency (ms) | Lane notes |"
    )
    lines.append(
        "|----------|-------:|-------------------:|----------------:|--------------:"
        "|-------------:|------------|"
    )
    for result in rows:
        lines.append(
            f"| {result.strategy.value} | {result.tokens_total:,} "
            f"| {result.token_efficiency_with_completion:.2f} "
            f"| {result.required_file_recall:.2f} "
            f"| {_fmt_opt(result.region_recall)} "
            f"| {result.wall_time_ms:.0f} "
            f"| {_advanced_lane_note(result)} |"
        )
    return lines


_PACKING_LANES = (
    "archex_query",
    "archex_query_compressed",
    "archex_query_efficiency_packed",
)
_PACKING_LANE_LABELS = {
    "archex_query": "normal packing",
    "archex_query_compressed": "compressed packing",
    "archex_query_efficiency_packed": "efficiency-aware packing",
}


def _packing_lane_comparison(reports: list[BenchmarkReport]) -> list[str]:
    """Compare normal vs compressed vs efficiency-aware packing on the same retrieval."""
    by_strategy: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for report in reports:
        for result in report.results:
            if result.strategy.value in _PACKING_LANES:
                by_strategy[result.strategy.value].append(result)
    # The section's subject is the efficiency-packed lane; omit it entirely when
    # that lane did not run so unrelated reports are unaffected.
    if "archex_query_efficiency_packed" not in by_strategy:
        return []
    lines = ["## Packing Lane Comparison", ""]
    lines.append(
        "Normal (`archex_query`), compressed (`archex_query_compressed`), and efficiency-aware "
        "(`archex_query_efficiency_packed`) packing on the same retrieval. Token efficiency is "
        "after the completion penalty; region Rel/1k and compression ratio are reported where "
        "available. The efficiency-packed lane is eligible for default consideration only if "
        "required-file/region metrics do not regress and p95 latency does not regress."
    )
    lines.append("")
    lines.append(
        "| Lane | Strategy | Token Efficiency | Region Rel/1k | Compression Ratio | p95 ms "
        "| Tasks |"
    )
    lines.append(
        "|------|----------|-----------------:|--------------:|------------------:|-------:"
        "|------:|"
    )
    for name in _PACKING_LANES:
        results = by_strategy.get(name)
        if not results:
            continue
        # Post-reduction efficiency where the lane shaped tokens (compressed/packed
        # set the compression field), else the plain completion-adjusted efficiency
        # for normal packing — so all three lanes are compared after their shaping.
        efficiency = _mean(
            [
                r.token_efficiency_with_compression_and_completion
                if r.token_efficiency_with_compression_and_completion is not None
                else r.token_efficiency_with_completion
                for r in results
            ]
        )
        region_rel = _mean_optional([r.relevance_per_1k_tokens for r in results])
        ratio = _mean_optional([r.bundle_compression_ratio for r in results])
        p95 = _percentile([r.wall_time_ms for r in results], 0.95)
        lines.append(
            f"| {_PACKING_LANE_LABELS[name]} | {name} | {efficiency:.3f} "
            f"| {_fmt_opt(region_rel, digits=2)} | {_fmt_opt(ratio)} "
            f"| {p95:.0f} | {len(results)} |"
        )
    lines.append("")
    return lines


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
        "All Required | Completion | Receipt Accuracy | Recall | Precision | F1 | Time (ms) |"
    )
    lines.append(header)
    lines.append(
        "|----------|--------|-----------------|------------------|------------------|"
        "--------------|------------|------------------|--------|-----------|------|-----------|"
    )
    for r in report.results:
        receipt_accuracy = _receipt_accuracy_label(r.receipt_accuracy)
        all_required = _all_required_label(r.all_required_files_present)
        lines.append(
            f"| {r.strategy.value} | {r.tokens_total:,} | {r.required_file_recall:.2f} "
            f"| {r.missed_required_file_rate:.2f} | {r.missed_required_task_rate:.2f} "
            f"| {all_required} | {r.task_completion_result.value} | {receipt_accuracy} "
            f"| {r.recall:.2f} | {r.precision:.2f} "
            f"| {r.f1_score:.2f} | {r.wall_time_ms:.0f} |"
        )
    lines.extend(_missing_required_file_appendix(report))
    lines.extend(_bundle_only_eval_appendix(report))
    lines.extend(_region_quality_appendix(report))
    lines.extend(_task_aware_policy_appendix(report))
    lines.extend(_compression_appendix(report))
    lines.extend(_packing_appendix(report))
    lines.extend(_advanced_lanes_appendix(report))
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
    lines.extend(_region_quality_summary(reports))
    lines.extend(_compression_summary(reports))
    lines.extend(_packing_lane_comparison(reports))
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


def _result_for_strategy(report: BenchmarkReport, strategy: str) -> BenchmarkResult | None:
    for result in report.results:
        if result.strategy.value == strategy:
            return result
    return None


def format_baseline_comparison(
    candidate_reports: list[BenchmarkReport],
    baseline_reports: list[BenchmarkReport],
    *,
    candidate_strategy: str,
    baseline_strategy: str,
) -> str:
    """Render per-task and aggregate deltas between two result directories."""
    if not candidate_reports or not baseline_reports:
        return "No baseline comparison available."

    baseline_by_task = {report.task_id: report for report in baseline_reports}
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    compression_ratios: list[float] = []
    for candidate_report in candidate_reports:
        baseline_report = baseline_by_task.get(candidate_report.task_id)
        if baseline_report is None:
            continue
        candidate = _result_for_strategy(candidate_report, candidate_strategy)
        baseline = _result_for_strategy(baseline_report, baseline_strategy)
        if candidate is None or baseline is None:
            continue
        compression_ratio = float(candidate.provenance.get("vector_compression_ratio", "0") or 0)
        if compression_ratio:
            compression_ratios.append(compression_ratio)
        rows.append(
            (
                candidate_report.task_id,
                candidate.recall - baseline.recall,
                candidate.mrr - baseline.mrr,
                candidate.f1_score - baseline.f1_score,
                candidate.required_file_recall - baseline.required_file_recall,
                candidate.wall_time_ms - baseline.wall_time_ms,
                compression_ratio,
            )
        )

    if not rows:
        return "No matching strategy results found for baseline comparison."

    lines = [
        "# Baseline Comparison",
        "",
        f"Candidate: `{candidate_strategy}`",
        f"Baseline: `{baseline_strategy}`",
        "",
        "| Task | Recall Δ | MRR Δ | F1 Δ | Required Recall Δ | Latency Δ ms | Compression |",
        "|------|----------|-------|------|-------------------|--------------|-------------|",
    ]
    for (
        task_id,
        recall_delta,
        mrr_delta,
        f1_delta,
        required_delta,
        latency_delta,
        compression,
    ) in rows:
        compression_cell = f"{compression:.2f}x" if compression else "n/a"
        lines.append(
            f"| {task_id} | {recall_delta:+.3f} | {mrr_delta:+.3f} "
            f"| {f1_delta:+.3f} | {required_delta:+.3f} "
            f"| {latency_delta:+.0f} | {compression_cell} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Recall Δ: {_mean([row[1] for row in rows]):+.3f}",
            f"- MRR Δ: {_mean([row[2] for row in rows]):+.3f}",
            f"- F1 Δ: {_mean([row[3] for row in rows]):+.3f}",
            f"- Required recall Δ: {_mean([row[4] for row in rows]):+.3f}",
            f"- Latency Δ ms: {_mean([row[5] for row in rows]):+.0f}",
            f"- Compression: {_mean(compression_ratios):.2f}x"
            if compression_ratios
            else "- Compression: n/a",
            "",
        ]
    )
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
        lines.extend(_bundle_only_eval_appendix(report))
        lines.extend(_task_aware_policy_appendix(report))
        lines.extend(_packing_appendix(report))
        lines.extend(_advanced_lanes_appendix(report))
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


def _bundle_only_eval_appendix(report: BenchmarkReport) -> list[str]:
    evaluated = [result for result in report.results if result.bundle_only_success is not None]
    if not evaluated:
        return []

    header = (
        "| Strategy | Success | Outside returned | In frontier cut | "
        "In top candidates | Safe-to-act false positive | Post reads |"
    )
    separator = (
        "|----------|---------|------------------|-----------------|"
        "-------------------|----------------------------|------------|"
    )
    lines = ["", "### Bundle-only evaluation", header, separator]
    for result in evaluated:
        outside = ", ".join(result.needed_files_outside_returned or []) or "none"
        frontier = ", ".join(result.needed_files_in_frontier_cut or []) or "none"
        top_candidates = ", ".join(result.needed_files_in_top_candidates or []) or "none"
        false_positive = _receipt_accuracy_label(result.safe_to_act_false_positive)
        post_reads = result.post_bundle_read_turns
        if post_reads is None:
            post_reads = "—"
        if result.bundle_only_success is None:
            continue
        success = result.bundle_only_success.value
        lines.append(
            f"| {result.strategy.value} | {success} | {outside} | {frontier} | "
            f"{top_candidates} | {false_positive} | {post_reads} |"
        )
    return lines


def _task_aware_policy_appendix(report: BenchmarkReport) -> list[str]:
    """Compact policy provenance for ``archex_query_task_aware`` results."""
    from archex.benchmark.models import Strategy

    rows = [
        result
        for result in report.results
        if result.strategy is Strategy.ARCHEX_QUERY_TASK_AWARE and result.provenance
    ]
    if not rows:
        return []
    header = "| Strategy | Modality | Budget | Routing | Cand cap | Dense cap | Fusion | Skipped |"
    separator = (
        "|----------|----------|--------|---------|----------|-----------|--------|---------|"
    )
    lines = ["", "### Task-aware policy", header, separator]
    for result in rows:
        prov = result.provenance
        lines.append(
            f"| {result.strategy.value} | {prov.get('modality', 'unknown')} "
            f"| {prov.get('budget_tier', 'unknown')} "
            f"| {prov.get('routing_decision', 'unknown')} "
            f"| {prov.get('policy_candidate_cap', 'unknown')} "
            f"| {prov.get('policy_dense_candidate_cap', 'unknown')} "
            f"| {prov.get('fusion_used', 'unknown')} "
            f"| {prov.get('policy_skipped_steps', 'unknown')} |"
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
