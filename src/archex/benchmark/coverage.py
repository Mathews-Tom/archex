"""Read required-file coverage evidence without consulting benchmark task oracles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkReport, Strategy


@dataclass(frozen=True)
class RequiredFileCoverage:
    """Returned-file evidence for one task and one benchmark strategy."""

    task_id: str
    strategy: Strategy
    returned_files: tuple[str, ...]
    missing_required_files: tuple[str, ...]
    required_file_recall: float
    completion_adjusted_token_efficiency: float
    warm_latency_ms: float | None
    seed_files: tuple[str, ...]
    expanded_files: tuple[str, ...]


def read_required_file_coverage(
    reports: Iterable[BenchmarkReport], strategy: Strategy
) -> list[RequiredFileCoverage]:
    """Return one required-file evidence row per report for *strategy*.

    The reader uses only fields emitted in benchmark reports. It deliberately
    does not load benchmark tasks or expected-file definitions.
    """
    rows: list[RequiredFileCoverage] = []
    seen_task_ids: set[str] = set()
    for report in reports:
        if report.task_id in seen_task_ids:
            msg = f"Duplicate benchmark report task ID: {report.task_id}"
            raise ValueError(msg)
        seen_task_ids.add(report.task_id)
        matches = [result for result in report.results if result.strategy is strategy]
        if len(matches) != 1:
            msg = (
                f"Expected exactly one {strategy.value} result for {report.task_id}, "
                f"found {len(matches)}"
            )
            raise ValueError(msg)
        result = matches[0]
        rows.append(
            RequiredFileCoverage(
                task_id=report.task_id,
                strategy=strategy,
                returned_files=tuple(result.result_files),
                missing_required_files=tuple(result.required_files_missing),
                required_file_recall=result.required_file_recall,
                completion_adjusted_token_efficiency=result.token_efficiency_with_completion,
                warm_latency_ms=result.warm_latency_ms,
                seed_files=tuple(result.seed_files),
                expanded_files=tuple(result.expanded_files),
            )
        )
    return rows
