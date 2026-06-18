"""Failure triage for persisted benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, BenchmarkTask, Strategy

LOW_F1_THRESHOLD = 0.40
LOW_PRECISION_THRESHOLD = 0.35
RAW_READ_GAP_THRESHOLD = 0.25


@dataclass(frozen=True)
class TriageFinding:
    """Ranked failure finding for a single task and strategy."""

    task_id: str
    category: str
    repo: str
    question: str
    expected_files: list[str]
    returned_files: list[str]
    missing_files: list[str]
    extra_files: list[str]
    recall: float
    precision: float
    f1_score: float
    mrr: float
    tokens_total: int
    token_efficiency: float
    savings_vs_raw: float
    seed_files: list[str]
    expanded_files: list[str]
    expansion_ratio: float
    expansion_eligible_seeds: int
    expansion_candidates_found: int
    expansion_import_neighbor_edges: int
    expansion_same_module_candidates: int
    expansion_hub_candidates: int
    expansion_test_candidates_skipped: int
    expansion_zero_candidate_reason: str
    expansion_reason_counts: dict[str, int]
    expanded_file_reasons: dict[str, list[str]]
    raw_read_recall: float | None
    raw_read_precision: float | None
    raw_read_f1_score: float | None
    failure_bucket: str
    failure_reasons: list[str]
    rank_score: float

    def to_json(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "repo": self.repo,
            "question": self.question,
            "expected_files": self.expected_files,
            "returned_files": self.returned_files,
            "missing_files": self.missing_files,
            "extra_files": self.extra_files,
            "metrics": {
                "recall": self.recall,
                "precision": self.precision,
                "f1_score": self.f1_score,
                "mrr": self.mrr,
                "tokens_total": self.tokens_total,
                "token_efficiency": self.token_efficiency,
                "savings_vs_raw": self.savings_vs_raw,
            },
            "seed_files": self.seed_files,
            "expanded_files": self.expanded_files,
            "expansion_ratio": self.expansion_ratio,
            "expansion_diagnostics": {
                "eligible_seeds": self.expansion_eligible_seeds,
                "candidates_found": self.expansion_candidates_found,
                "import_neighbor_edges": self.expansion_import_neighbor_edges,
                "same_module_candidates": self.expansion_same_module_candidates,
                "hub_candidates": self.expansion_hub_candidates,
                "test_candidates_skipped": self.expansion_test_candidates_skipped,
                "zero_candidate_reason": self.expansion_zero_candidate_reason,
                "reason_counts": self.expansion_reason_counts,
                "file_reasons": self.expanded_file_reasons,
            },
            "raw_read_baseline_metrics": {
                "recall": self.raw_read_recall,
                "precision": self.raw_read_precision,
                "f1_score": self.raw_read_f1_score,
            },
            "failure_bucket": self.failure_bucket,
            "failure_reasons": self.failure_reasons,
            "rank_score": self.rank_score,
        }


def load_benchmark_reports(input_dir: Path) -> list[BenchmarkReport]:
    """Load benchmark report JSON files from a directory."""
    reports: list[BenchmarkReport] = []
    for json_file in sorted(input_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))
    return reports


def load_benchmark_tasks(tasks_dir: Path) -> dict[str, BenchmarkTask]:
    """Load benchmark task YAML files keyed by task id."""
    return {task.task_id: task for task in load_tasks(tasks_dir)}


def triage_failures(
    reports: list[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy = Strategy.ARCHEX_QUERY,
) -> list[TriageFinding]:
    """Return ranked benchmark failure findings for the selected strategy."""
    findings: list[TriageFinding] = []
    for report in reports:
        result = _find_result(report, strategy)
        if result is None:
            continue
        raw_read = _find_raw_read_result(report)
        task = tasks_by_id.get(report.task_id)
        expected_files = task.expected_files if task is not None else []
        category = _category(result, task)
        reasons = _failure_reasons(result, raw_read, category)
        if not reasons:
            continue
        returned_files = _returned_files(result)
        missing_files = _missing_files(expected_files, returned_files, result)
        extra_files = [path for path in returned_files if path not in set(expected_files)]
        bucket = _failure_bucket(result, raw_read, category, missing_files)
        findings.append(
            TriageFinding(
                task_id=report.task_id,
                category=category,
                repo=report.repo,
                question=report.question,
                expected_files=expected_files,
                returned_files=returned_files,
                missing_files=missing_files,
                extra_files=extra_files,
                recall=result.recall,
                precision=result.precision,
                f1_score=result.f1_score,
                mrr=result.mrr,
                tokens_total=result.tokens_total,
                token_efficiency=result.token_efficiency,
                savings_vs_raw=result.savings_vs_raw,
                seed_files=result.seed_files,
                expanded_files=result.expanded_files,
                expansion_ratio=result.expansion_ratio,
                expansion_eligible_seeds=result.expansion_eligible_seeds,
                expansion_candidates_found=result.expansion_candidates_found,
                expansion_import_neighbor_edges=result.expansion_import_neighbor_edges,
                expansion_same_module_candidates=result.expansion_same_module_candidates,
                expansion_hub_candidates=result.expansion_hub_candidates,
                expansion_test_candidates_skipped=result.expansion_test_candidates_skipped,
                expansion_zero_candidate_reason=result.expansion_zero_candidate_reason,
                expansion_reason_counts=result.expansion_reason_counts,
                expanded_file_reasons=result.expanded_file_reasons,
                raw_read_recall=raw_read.recall if raw_read is not None else None,
                raw_read_precision=raw_read.precision if raw_read is not None else None,
                raw_read_f1_score=raw_read.f1_score if raw_read is not None else None,
                failure_bucket=bucket,
                failure_reasons=reasons,
                rank_score=_rank_score(result, raw_read, category, bucket),
            )
        )
    return sorted(findings, key=_finding_sort_key)


def format_triage_markdown(findings: list[TriageFinding]) -> str:
    """Render triage findings as Markdown."""
    if not findings:
        return "# Benchmark Failure Triage\n\nNo failures matched the triage thresholds."

    lines = ["# Benchmark Failure Triage", ""]
    lines.append(
        "| Rank | Task | Category | Bucket | Recall | Precision | F1 | Tokens Total "
        "| Token Efficiency | Savings vs Raw | Missing | Extra |"
    )
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for idx, finding in enumerate(findings, start=1):
        lines.append(
            f"| {idx} | `{finding.task_id}` | {finding.category} | {finding.failure_bucket} "
            f"| {finding.recall:.3f} | {finding.precision:.3f} | {finding.f1_score:.3f} "
            f"| {finding.tokens_total:,} | {finding.token_efficiency:.3f} "
            f"| {finding.savings_vs_raw:.1f}% | {_inline_files(finding.missing_files)} "
            f"| {_inline_files(finding.extra_files)} |"
        )
    lines.append("")

    for idx, finding in enumerate(findings, start=1):
        lines.append(f"## {idx}. {finding.task_id}")
        lines.append("")
        lines.append(f"- Bucket: `{finding.failure_bucket}`")
        lines.append(f"- Reasons: {', '.join(f'`{reason}`' for reason in finding.failure_reasons)}")
        lines.append(f"- Repo: `{finding.repo}`")
        lines.append(f"- Category: `{finding.category}`")
        lines.append(f"- Question: {finding.question}")
        lines.append(
            "- Metrics: "
            f"recall `{finding.recall:.3f}`, precision `{finding.precision:.3f}`, "
            f"F1 `{finding.f1_score:.3f}`, MRR `{finding.mrr:.3f}`, "
            f"tokens `{finding.tokens_total:,}`, "
            f"token efficiency `{finding.token_efficiency:.3f}`, "
            f"savings vs raw `{finding.savings_vs_raw:.1f}%`"
        )
        if finding.raw_read_f1_score is not None:
            lines.append(
                "- Raw ripgrep/read baseline: "
                f"recall `{finding.raw_read_recall:.3f}`, "
                f"precision `{finding.raw_read_precision:.3f}`, "
                f"F1 `{finding.raw_read_f1_score:.3f}`"
            )
        lines.append(
            "- Expansion: "
            f"eligible seeds `{finding.expansion_eligible_seeds}`, "
            f"candidates `{finding.expansion_candidates_found}`, "
            f"import edges `{finding.expansion_import_neighbor_edges}`, "
            f"same-module `{finding.expansion_same_module_candidates}`, "
            f"hubs `{finding.expansion_hub_candidates}`, "
            f"test skips `{finding.expansion_test_candidates_skipped}`"
        )
        if finding.expansion_reason_counts:
            reason_summary = ", ".join(
                f"`{reason}` `{count}`"
                for reason, count in sorted(finding.expansion_reason_counts.items())
            )
            lines.append(f"- Expansion reasons: {reason_summary}")
        if finding.expansion_zero_candidate_reason:
            lines.append(f"- Zero expansion reason: `{finding.expansion_zero_candidate_reason}`")
        lines.append("- Expected files:")
        lines.extend(_file_bullets(finding.expected_files))
        lines.append("- Returned files:")
        lines.extend(_file_bullets(finding.returned_files))
        lines.append("- Missing files:")
        lines.extend(_file_bullets(finding.missing_files))
        lines.append("- Extra files:")
        lines.extend(_file_bullets(finding.extra_files))
        lines.append("- Seed files:")
        lines.extend(_file_bullets(finding.seed_files))
        lines.append("- Expanded files:")
        lines.extend(_file_bullets(finding.expanded_files))
        lines.append("")
    return "\n".join(lines)


def format_triage_json(findings: list[TriageFinding]) -> str:
    """Render triage findings as stable JSON."""
    return json.dumps([finding.to_json() for finding in findings], indent=2, sort_keys=True)


def _find_result(report: BenchmarkReport, strategy: Strategy) -> BenchmarkResult | None:
    for result in report.results:
        if result.strategy == strategy:
            return result
    return None


def _category(result: BenchmarkResult, task: BenchmarkTask | None) -> str:
    if result.category is not None:
        return result.category.value
    if task is not None and task.category is not None:
        return task.category.value
    return "uncategorized"


def _returned_files(result: BenchmarkResult) -> list[str]:
    returned: list[str] = []
    seen: set[str] = set()
    for path in [*result.seed_files, *result.expanded_files]:
        if path in seen:
            continue
        seen.add(path)
        returned.append(path)
    return returned


def _missing_files(
    expected_files: list[str],
    returned_files: list[str],
    result: BenchmarkResult,
) -> list[str]:
    if returned_files:
        return sorted(set(expected_files) - set(returned_files))
    if result.recall >= 1.0:
        return []
    return sorted(expected_files)


def _failure_reasons(
    result: BenchmarkResult,
    raw_read: BenchmarkResult | None,
    category: str,
) -> list[str]:
    reasons: list[str] = []
    if result.recall <= 0.0:
        reasons.append("zero_recall")
    if result.f1_score < LOW_F1_THRESHOLD:
        reasons.append("low_f1")
    if result.precision < LOW_PRECISION_THRESHOLD:
        reasons.append("low_precision")
    raw_read_gap_reason = _raw_read_gap_reason(result, raw_read)
    if raw_read_gap_reason is not None:
        reasons.append(raw_read_gap_reason)
    if category == "external-large" and result.f1_score < LOW_F1_THRESHOLD:
        reasons.append("external_large_failure")
    if category == "framework-semantic" and result.f1_score < LOW_F1_THRESHOLD:
        reasons.append("framework_semantic_failure")
    return reasons


def _failure_bucket(
    result: BenchmarkResult,
    raw_read: BenchmarkResult | None,
    category: str,
    missing_files: list[str],
) -> str:
    if result.recall <= 0.0:
        return "zero_recall"
    if category == "external-large":
        return "large_repo_ambiguity"
    if category == "framework-semantic":
        return "semantic_gap"
    raw_read_gap_reason = _raw_read_gap_reason(result, raw_read)
    if raw_read_gap_reason is not None:
        return raw_read_gap_reason
    if result.expanded_files and missing_files and result.precision < LOW_PRECISION_THRESHOLD:
        return "expansion_noise"
    if result.precision < LOW_PRECISION_THRESHOLD:
        return "low_precision"
    return "semantic_gap"


def _find_raw_read_result(report: BenchmarkReport) -> BenchmarkResult | None:
    return _find_result(report, Strategy.RAW_RIPGREP) or _find_result(report, Strategy.RAW_GREPPED)


def _raw_read_gap_reason(
    result: BenchmarkResult,
    raw_read: BenchmarkResult | None,
) -> str | None:
    if raw_read is None:
        return None
    recall_gap = raw_read.recall - result.recall
    f1_gap = raw_read.f1_score - result.f1_score
    if recall_gap < RAW_READ_GAP_THRESHOLD and f1_gap < RAW_READ_GAP_THRESHOLD:
        return None
    return "raw_ripgrep_gap" if raw_read.strategy is Strategy.RAW_RIPGREP else "raw_grepped_gap"


def _rank_score(
    result: BenchmarkResult,
    raw_read: BenchmarkResult | None,
    category: str,
    bucket: str,
) -> float:
    score = 0.0
    if result.recall <= 0.0:
        score += 100.0
    score += max(0.0, LOW_F1_THRESHOLD - result.f1_score) * 25.0
    score += max(0.0, LOW_PRECISION_THRESHOLD - result.precision) * 10.0
    if _raw_read_gap_reason(result, raw_read) is not None:
        score += 20.0
    if category == "external-large":
        score += 12.0
    if category == "framework-semantic":
        score += 10.0
    if bucket == "expansion_noise":
        score += 5.0
    return score


def _finding_sort_key(finding: TriageFinding) -> tuple[float, float, float, str]:
    return (-finding.rank_score, finding.recall, finding.f1_score, finding.task_id)


def _inline_files(files: list[str]) -> str:
    if not files:
        return "none"
    return "<br>".join(f"`{path}`" for path in files)


def _file_bullets(files: list[str]) -> list[str]:
    if not files:
        return ["  - none"]
    return [f"  - `{path}`" for path in files]
