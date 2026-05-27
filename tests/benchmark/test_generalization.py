"""Generalization guards for retrieval-quality changes."""

from __future__ import annotations

from pathlib import Path

from archex.benchmark.loader import load_tasks


ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "benchmarks" / "tasks"
HELD_OUT_PATH = ROOT / "benchmarks" / "held_out.txt"


def _held_out_ids() -> list[str]:
    return [
        line.strip()
        for line in HELD_OUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_held_out_set_has_five_existing_tasks() -> None:
    held_out = _held_out_ids()
    tasks_by_id = {task.task_id: task for task in load_tasks(TASKS_DIR)}

    assert len(held_out) == 5
    assert len(set(held_out)) == len(held_out)
    assert set(held_out) <= set(tasks_by_id)


def test_held_out_set_mixes_self_and_external_tasks() -> None:
    tasks_by_id = {task.task_id: task for task in load_tasks(TASKS_DIR)}
    held_out_tasks = [tasks_by_id[task_id] for task_id in _held_out_ids()]

    assert any(task.repo == "." for task in held_out_tasks)
    assert any(task.repo != "." for task in held_out_tasks)
