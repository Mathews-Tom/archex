"""Read file-ranking and result-set-noise evidence without consulting benchmark
task oracles.

Mirrors `archex.benchmark.coverage`: a report-only reader over fields that
`archex benchmark run` already emits. It never loads benchmark tasks or
expected-file definitions; it only classifies already-recorded gate metrics
(recall, precision, F1, MRR) and file counts into a rank/noise failure
taxonomy for M0.3 characterization.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkReport, Strategy, TaskFamily

# "rank_below_first": every required file the task asks for is present
# somewhere in the returned set (required_file_recall == 1.0), but the first
# required file is not ranked first (mrr < 1.0). The result set is correct;
# only its order is wrong.
#
# "broad_result_set": required-file recall clears the gate floor, but
# precision does not -- the returned file set is dominated by files the task
# never asked for. The order may or may not be correct; the set is too big.
#
# A task can carry both classes (recall is fine, first hit is not first, and
# the set is also noisy) or neither (a pure coverage miss, out of M0.3 scope
# and left to M0.2/M0.4).
RANK_BELOW_FIRST = "rank_below_first"
BROAD_RESULT_SET = "broad_result_set"
BELOW_RECALL_FLOOR = "below_recall_floor"
BELOW_PRECISION_FLOOR = "below_precision_floor"
BELOW_F1_FLOOR = "below_f1_floor"
BELOW_MRR_FLOOR = "below_mrr_floor"


@dataclass(frozen=True)
class RankNoiseObservation:
    """Returned-file rank/noise evidence for one task and one benchmark strategy."""

    task_id: str
    strategy: Strategy
    family: TaskFamily
    required_file_recall: float
    recall: float
    precision: float
    f1_score: float
    mrr: float
    result_file_count: int
    required_file_count: int
    failure_classes: tuple[str, ...]


def read_rank_noise_observations(
    reports: Iterable[BenchmarkReport],
    strategy: Strategy,
    *,
    min_recall: float,
    min_precision: float,
    min_f1: float,
    min_mrr: float,
) -> list[RankNoiseObservation]:
    """Return one rank/noise observation per report for *strategy*.

    The reader uses only fields emitted in benchmark reports (recall,
    precision, F1, MRR, required-file recall, and returned/required file
    counts). It deliberately does not load benchmark tasks or expected-file
    definitions.
    """
    rows: list[RankNoiseObservation] = []
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
        required_file_count = len(result.required_files_present) + len(
            result.required_files_missing
        )

        classes: list[str] = []
        if result.recall < min_recall:
            classes.append(BELOW_RECALL_FLOOR)
        if result.precision < min_precision:
            classes.append(BELOW_PRECISION_FLOOR)
        if result.f1_score < min_f1:
            classes.append(BELOW_F1_FLOOR)
        if result.mrr < min_mrr:
            classes.append(BELOW_MRR_FLOOR)
        if result.required_file_recall >= 1.0 and result.mrr < 1.0:
            classes.append(RANK_BELOW_FIRST)
        if result.required_file_recall >= min_recall and result.precision < min_precision:
            classes.append(BROAD_RESULT_SET)

        rows.append(
            RankNoiseObservation(
                task_id=report.task_id,
                strategy=strategy,
                family=result.family,
                required_file_recall=result.required_file_recall,
                recall=result.recall,
                precision=result.precision,
                f1_score=result.f1_score,
                mrr=result.mrr,
                result_file_count=len(result.result_files),
                required_file_count=required_file_count,
                failure_classes=tuple(classes),
            )
        )
    return rows
