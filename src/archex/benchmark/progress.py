"""Terminal progress rendering for benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from types import TracebackType
from typing import Protocol

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

WARMING_ACTIVITY = "warming vector index…"


class _BenchmarkTaskLike(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def repo(self) -> str: ...


class BenchmarkProgress:
    """Own the benchmark live display and its two progress rows."""

    def __init__(
        self,
        tasks: Sequence[_BenchmarkTaskLike],
        *,
        console: Console | None = None,
        force_disable: bool = False,
        refresh_per_second: float = 8.0,
    ) -> None:
        self.console = console or Console(stderr=True)
        self._tasks = tasks
        self._refresh_per_second = refresh_per_second
        self._disable = force_disable or not self.console.is_terminal
        self._overall_task_id: TaskID | None = None
        self._active_task_id: TaskID | None = None
        self._group: Group | None = None
        self._live: Live | None = None
        self.overall_progress = self._make_overall_progress()
        self.active_progress = self._make_active_progress()

    @property
    def live_display_enabled(self) -> bool:
        return not self._disable

    @property
    def active_task(self) -> Task | None:
        if self._active_task_id is None:
            return None
        return self.active_progress.tasks[self._active_task_id]

    def __enter__(self) -> BenchmarkProgress:
        self._overall_task_id = self.overall_progress.add_task(
            "",
            total=len(self._tasks),
            visible=bool(self._tasks),
        )
        self._active_task_id = self.active_progress.add_task(
            "",
            total=None,
            activity="",
            visible=False,
        )
        self._group = Group(self.overall_progress, self.active_progress)
        if self.live_display_enabled:
            self._live = Live(
                self._group,
                console=self.console,
                refresh_per_second=self._refresh_per_second,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start(refresh=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._live is not None:
            self._live.stop()
            self._live = None

    def start_task(self, task: _BenchmarkTaskLike) -> None:
        self._require_started()
        self.overall_progress.update(
            self._overall_id(),
            description=f"{task.task_id} ({task.repo})",
        )
        self.active_progress.update(
            self._active_id(),
            description=f"  {task.task_id}",
            total=None,
            completed=0,
            activity="",
            visible=True,
        )

    def start_warmup(self) -> None:
        self._require_started()
        self.active_progress.update(
            self._active_id(),
            total=None,
            completed=0,
            activity=WARMING_ACTIVITY,
            visible=True,
        )

    def finish_warmup(self, strategies: Sequence[object]) -> None:
        self._require_started()
        self.active_progress.update(
            self._active_id(),
            total=len(strategies),
            completed=0,
            activity="",
            visible=True,
        )

    def start_strategy(self, strategy: StrEnum) -> None:
        self._require_started()
        self.active_progress.update(
            self._active_id(),
            activity=strategy.value,
            visible=True,
        )

    def finish_strategy(self) -> None:
        self._require_started()
        self.active_progress.advance(self._active_id())

    def finish_task(self) -> None:
        self._require_started()
        self.overall_progress.advance(self._overall_id())

    def refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _make_overall_progress(self) -> Progress:
        return Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            disable=self._disable,
            refresh_per_second=self._refresh_per_second,
        )

    def _make_active_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[activity]}"),
            console=self.console,
            disable=self._disable,
            refresh_per_second=self._refresh_per_second,
        )

    def _require_started(self) -> None:
        if self._overall_task_id is None or self._active_task_id is None:
            raise RuntimeError("BenchmarkProgress must be used as a context manager")

    def _overall_id(self) -> TaskID:
        self._require_started()
        assert self._overall_task_id is not None
        return self._overall_task_id

    def _active_id(self) -> TaskID:
        self._require_started()
        assert self._active_task_id is not None
        return self._active_task_id
