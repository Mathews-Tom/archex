from __future__ import annotations

from pathlib import Path

from archex.exceptions import AcquireError
from archex.metrics.health import note_metrics_recording_failure, read_metrics_health


def test_expected_source_unavailability_does_not_latch_warning(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"

    note_metrics_recording_failure(AcquireError("Path does not exist: /repo"), db_path=db_path)

    assert read_metrics_health(db_path=db_path).status == "ok"


def test_missing_file_baseline_failure_does_not_latch_warning(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"

    note_metrics_recording_failure(FileNotFoundError("/repo"), db_path=db_path)

    assert read_metrics_health(db_path=db_path).status == "ok"


def test_unexpected_failure_latches_warning(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"

    note_metrics_recording_failure(RuntimeError("boom"), db_path=db_path)

    health = read_metrics_health(db_path=db_path)
    assert health.status == "warning"
    assert health.last_failure_operation == "record"
    assert health.last_failure_message == "boom"
