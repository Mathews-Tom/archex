"""Tests for benchmark progress rendering."""

from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.progress import WARMING_ACTIVITY, BenchmarkProgress


def _task(task_id: str, repo: str = ".") -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        repo=repo,
        commit="HEAD",
        question="How does this work?",
        expected_files=["main.py"],
    )


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def test_progress_renders_overall_and_active_task_text() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, color_system=None, width=140)
    tasks = [_task("task_one", "owner/repo"), _task("task_two")]

    with BenchmarkProgress(tasks, console=console, refresh_per_second=30) as progress:
        progress.start_task(tasks[0])
        progress.start_warmup()
        active = progress.active_task
        assert active is not None
        assert active.fields["activity"] == WARMING_ACTIVITY
        progress.refresh()
        progress.finish_warmup([Strategy.RAW_FILES, Strategy.ARCHEX_QUERY])
        progress.start_strategy(Strategy.RAW_FILES)
        active = progress.active_task
        assert active is not None
        assert active.fields["activity"] == "raw_files"
        progress.refresh()
        progress.finish_strategy()
        progress.start_strategy(Strategy.ARCHEX_QUERY)
        progress.refresh()

    output = _strip_ansi(buffer.getvalue())
    assert "task_one (owner/repo)" in output
    assert "0/2" in output
    assert "archex_query" in output


def test_warmup_is_indeterminate_then_flips_to_strategy_total() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, color_system=None, width=120)
    task = _task("vector_task")
    strategies = [Strategy.RAW_FILES, Strategy.ARCHEX_QUERY, Strategy.ARCHEX_QUERY_FUSION]

    with BenchmarkProgress([task], console=console) as progress:
        progress.start_task(task)
        progress.start_warmup()
        active = progress.active_task
        assert active is not None
        assert active.total is None
        assert active.fields["activity"] == WARMING_ACTIVITY

        progress.finish_warmup(strategies)
        active = progress.active_task
        assert active is not None
        assert active.total == len(strategies)
        assert active.completed == 0


def test_non_tty_disables_live_display_and_progress_rows() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False)
    task = _task("plain_task")

    with BenchmarkProgress([task], console=console) as progress:
        assert progress.live_display_enabled is False
        assert progress.overall_progress.disable is True
        assert progress.active_progress.disable is True
        progress.start_task(task)
        progress.console.log("plain log line")

    assert "plain log line" in buffer.getvalue()
    assert "plain_task" not in buffer.getvalue()
