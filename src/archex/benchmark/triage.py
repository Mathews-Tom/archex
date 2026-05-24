"""Failure triage for persisted benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, BenchmarkTask, Strategy

LOW_F1_THRESHOLD = 0.40
LOW_PRECISION_THRESHOLD = 0.35
RAW_GREPPED_GAP_THRESHOLD = 0.25


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
    seed_files: list[str]
    expanded_files: list[str]
    expansion_ratio: float
    raw_grepped_recall: float | None
    raw_grepped_precision: float | None
    raw_grepped_f1_score: float | None
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
            },
            "seed_files": self.seed_files,
            "expanded_files": self.expanded_files,
            "expansion_ratio": self.expansion_ratio,
            "raw_grepped_metrics": {
                "recall": self.raw_grepped_recall,
                "precision": self.raw_grepped_precision,
                "f1_score": self.raw_grepped_f1_score,
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
        raw_grepped = _find_result(report, Strategy.RAW_GREPPED)
        task = tasks_by_id.get(report.task_id)
        expected_files = task.expected_files if task is not None else []
        category = _category(result, task)
        reasons = _failure_reasons(result, raw_grepped, category)
        if not reasons:
            continue
        returned_files = _returned_files(result)
        missing_files = _missing_files(expected_files, returned_files, result)
        extra_files = [path for path in returned_files if path not in set(expected_files)]
        bucket = _failure_bucket(result, raw_grepped, category, missing_files)
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
                seed_files=result.seed_files,
                expanded_files=result.expanded_files,
                expansion_ratio=result.expansion_ratio,
                raw_grepped_recall=raw_grepped.recall if raw_grepped is not None else None,
                raw_grepped_precision=raw_grepped.precision if raw_grepped is not None else None,
                raw_grepped_f1_score=raw_grepped.f1_score if raw_grepped is not None else None,
                failure_bucket=bucket,
                failure_reasons=reasons,
                rank_score=_rank_score(result, raw_grepped, category, bucket),
            )
        )
    return sorted(findings, key=_finding_sort_key)


def format_triage_markdown(findings: list[TriageFinding]) -> str:
    """Render triage findings as Markdown."""
    if not findings:
        return "# Benchmark Failure Triage\n\nNo failures matched the triage thresholds."

    lines = ["# Benchmark Failure Triage", ""]
    lines.append("| Rank | Task | Category | Bucket | Recall | Precision | F1 | Missing | Extra |")
    lines.append("|---:|---|---|---|---:|---:|---:|---|---|")
    for idx, finding in enumerate(findings, start=1):
        lines.append(
            f"| {idx} | `{finding.task_id}` | {finding.category} | {finding.failure_bucket} "
            f"| {finding.recall:.3f} | {finding.precision:.3f} | {finding.f1_score:.3f} "
            f"| {_inline_files(finding.missing_files)} | {_inline_files(finding.extra_files)} |"
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
            f"F1 `{finding.f1_score:.3f}`, MRR `{finding.mrr:.3f}`"
        )
        if finding.raw_grepped_f1_score is not None:
            lines.append(
                "- Raw grepped: "
                f"recall `{finding.raw_grepped_recall:.3f}`, "
                f"precision `{finding.raw_grepped_precision:.3f}`, "
                f"F1 `{finding.raw_grepped_f1_score:.3f}`"
            )
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
    raw_grepped: BenchmarkResult | None,
    category: str,
) -> list[str]:
    reasons: list[str] = []
    if result.recall <= 0.0:
        reasons.append("zero_recall")
    if result.f1_score < LOW_F1_THRESHOLD:
        reasons.append("low_f1")
    if result.precision < LOW_PRECISION_THRESHOLD:
        reasons.append("low_precision")
    if _raw_grepped_gap(result, raw_grepped):
        reasons.append("raw_grepped_gap")
    if category == "external-large" and result.f1_score < LOW_F1_THRESHOLD:
        reasons.append("external_large_failure")
    if category == "framework-semantic" and result.f1_score < LOW_F1_THRESHOLD:
        reasons.append("framework_semantic_failure")
    return reasons


def _failure_bucket(
    result: BenchmarkResult,
    raw_grepped: BenchmarkResult | None,
    category: str,
    missing_files: list[str],
) -> str:
    if result.recall <= 0.0:
        return "zero_recall"
    if category == "external-large":
        return "large_repo_ambiguity"
    if category == "framework-semantic":
        return "semantic_gap"
    if _raw_grepped_gap(result, raw_grepped):
        return "raw_grepped_gap"
    if result.expanded_files and missing_files and result.precision < LOW_PRECISION_THRESHOLD:
        return "expansion_noise"
    if result.precision < LOW_PRECISION_THRESHOLD:
        return "low_precision"
    return "semantic_gap"


def _raw_grepped_gap(
    result: BenchmarkResult,
    raw_grepped: BenchmarkResult | None,
) -> bool:
    if raw_grepped is None:
        return False
    recall_gap = raw_grepped.recall - result.recall
    f1_gap = raw_grepped.f1_score - result.f1_score
    return recall_gap >= RAW_GREPPED_GAP_THRESHOLD or f1_gap >= RAW_GREPPED_GAP_THRESHOLD


def _rank_score(
    result: BenchmarkResult,
    raw_grepped: BenchmarkResult | None,
    category: str,
    bucket: str,
) -> float:
    score = 0.0
    if result.recall <= 0.0:
        score += 100.0
    score += max(0.0, LOW_F1_THRESHOLD - result.f1_score) * 25.0
    score += max(0.0, LOW_PRECISION_THRESHOLD - result.precision) * 10.0
    if _raw_grepped_gap(result, raw_grepped):
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
