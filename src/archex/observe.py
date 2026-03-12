"""Lightweight structured observability for the archex retrieval pipeline.

Provides timing instrumentation and structured logging using only stdlib.
No external dependencies.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger("archex.observe")


@dataclass
class StepTiming:
    """Timing record for a single pipeline step."""

    name: str
    start_ns: int = 0
    end_ns: int = 0
    metadata: dict[str, str | int | float | bool] = field(default_factory=lambda: {})

    @property
    def duration_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds."""
        return (self.end_ns - self.start_ns) / 1_000_000

    def to_dict(self) -> dict[str, str | int | float | bool]:
        result: dict[str, str | int | float | bool] = {
            "step": self.name,
            "duration_ms": round(self.duration_ms, 2),
        }
        result.update(self.metadata)
        return result


@dataclass
class PipelineTrace:
    """Accumulated trace for a single pipeline execution (one request).

    Collects ordered StepTiming records and exposes a structured summary.
    """

    operation: str
    steps: list[StepTiming] = field(default_factory=lambda: [])
    start_ns: int = 0
    end_ns: int = 0
    metadata: dict[str, str | int | float | bool] = field(default_factory=lambda: {})

    @property
    def total_ms(self) -> float:
        if self.end_ns > 0:
            return (self.end_ns - self.start_ns) / 1_000_000
        return sum(s.duration_ms for s in self.steps)

    def add_step(self, step: StepTiming) -> None:
        self.steps.append(step)

    def step_durations(self) -> dict[str, float]:
        """Map of step name to duration in ms."""
        return {s.name: round(s.duration_ms, 2) for s in self.steps}

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "total_ms": round(self.total_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def log_summary(self, level: int = logging.DEBUG) -> None:
        """Emit a structured log line with the trace summary."""
        if not logger.isEnabledFor(level):
            return
        parts = [f"{self.operation} completed in {self.total_ms:.1f}ms"]
        for step in self.steps:
            parts.append(f"  {step.name}: {step.duration_ms:.1f}ms")
        logger.log(level, "\n".join(parts))


class TraceCollector:
    """Accumulates PipelineTrace objects across multiple operations.

    Thread-safe append; intended for request-scoped or session-scoped collection.
    """

    def __init__(self) -> None:
        self._traces: list[PipelineTrace] = []

    @property
    def traces(self) -> list[PipelineTrace]:
        return list(self._traces)

    def add(self, trace: PipelineTrace) -> None:
        self._traces.append(trace)

    def clear(self) -> None:
        self._traces.clear()

    def summary(self) -> list[dict[str, object]]:
        return [t.to_dict() for t in self._traces]


@contextmanager
def traced_step(trace: PipelineTrace, name: str) -> Generator[StepTiming, None, None]:
    """Context manager that times a named step and appends it to the trace.

    Usage::

        with traced_step(trace, "bm25_search") as step:
            results = bm25.search(query)
            step.metadata["candidates"] = len(results)
    """
    step = StepTiming(name=name, start_ns=time.perf_counter_ns())
    yield step
    step.end_ns = time.perf_counter_ns()
    trace.add_step(step)


@contextmanager
def traced_operation(
    operation: str,
    collector: TraceCollector | None = None,
) -> Generator[PipelineTrace, None, None]:
    """Context manager that creates a PipelineTrace for a top-level operation.

    Automatically records start/end timestamps and optionally appends to a collector.

    Usage::

        with traced_operation("query") as trace:
            with traced_step(trace, "acquire"):
                ...
            with traced_step(trace, "search"):
                ...
        # trace.total_ms is now populated
    """
    trace = PipelineTrace(operation=operation, start_ns=time.perf_counter_ns())
    yield trace
    trace.end_ns = time.perf_counter_ns()
    trace.log_summary()
    if collector is not None:
        collector.add(trace)
