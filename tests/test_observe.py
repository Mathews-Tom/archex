"""Tests for the archex.observe observability module."""

# pyright: reportIndexIssue=false, reportArgumentType=false
from __future__ import annotations

import json
import logging
import time

from archex.observe import (
    PipelineTrace,
    StepTiming,
    TraceCollector,
    traced_operation,
    traced_step,
)


class TestStepTiming:
    def test_duration_ms(self) -> None:
        step = StepTiming(name="test", start_ns=0, end_ns=5_000_000)
        assert step.duration_ms == 5.0

    def test_duration_ms_zero(self) -> None:
        step = StepTiming(name="test", start_ns=100, end_ns=100)
        assert step.duration_ms == 0.0

    def test_to_dict_includes_metadata(self) -> None:
        step = StepTiming(
            name="search",
            start_ns=0,
            end_ns=10_000_000,
            metadata={"candidates": 42},
        )
        d = step.to_dict()
        assert d["step"] == "search"
        assert d["duration_ms"] == 10.0
        assert d["candidates"] == 42

    def test_to_dict_without_metadata(self) -> None:
        step = StepTiming(name="parse", start_ns=0, end_ns=1_000_000)
        d = step.to_dict()
        assert set(d.keys()) == {"step", "duration_ms"}


class TestPipelineTrace:
    def test_add_step(self) -> None:
        trace = PipelineTrace(operation="query")
        trace.add_step(StepTiming(name="a", start_ns=0, end_ns=1_000_000))
        trace.add_step(StepTiming(name="b", start_ns=1_000_000, end_ns=3_000_000))
        assert len(trace.steps) == 2

    def test_total_ms_from_timestamps(self) -> None:
        trace = PipelineTrace(
            operation="query",
            start_ns=0,
            end_ns=10_000_000,
        )
        assert trace.total_ms == 10.0

    def test_total_ms_from_steps_when_no_timestamps(self) -> None:
        trace = PipelineTrace(operation="query")
        trace.add_step(StepTiming(name="a", start_ns=0, end_ns=3_000_000))
        trace.add_step(StepTiming(name="b", start_ns=3_000_000, end_ns=8_000_000))
        assert trace.total_ms == 8.0

    def test_step_durations(self) -> None:
        trace = PipelineTrace(operation="test")
        trace.add_step(StepTiming(name="parse", start_ns=0, end_ns=2_000_000))
        trace.add_step(StepTiming(name="index", start_ns=2_000_000, end_ns=5_000_000))
        durations = trace.step_durations()
        assert durations == {"parse": 2.0, "index": 3.0}

    def test_to_dict(self) -> None:
        trace = PipelineTrace(
            operation="query",
            start_ns=0,
            end_ns=10_000_000,
            metadata={"cache_hit": True},
        )
        trace.add_step(StepTiming(name="search", start_ns=0, end_ns=5_000_000))
        d = trace.to_dict()
        assert d["operation"] == "query"
        assert d["total_ms"] == 10.0
        assert d["metadata"] == {"cache_hit": True}
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step"] == "search"

    def test_to_json(self) -> None:
        trace = PipelineTrace(operation="analyze", start_ns=0, end_ns=1_000_000)
        parsed = json.loads(trace.to_json())
        assert parsed["operation"] == "analyze"
        assert parsed["total_ms"] == 1.0

    def test_log_summary(self) -> None:
        trace = PipelineTrace(operation="query", start_ns=0, end_ns=5_000_000)
        trace.add_step(StepTiming(name="search", start_ns=0, end_ns=3_000_000))
        logger = logging.getLogger("archex.observe")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        try:
            trace.log_summary(level=logging.DEBUG)
        finally:
            logger.removeHandler(handler)

    def test_metadata_default_empty(self) -> None:
        trace = PipelineTrace(operation="x")
        assert trace.metadata == {}


class TestTraceCollector:
    def test_add_and_list(self) -> None:
        collector = TraceCollector()
        t1 = PipelineTrace(operation="a", start_ns=0, end_ns=1_000_000)
        t2 = PipelineTrace(operation="b", start_ns=0, end_ns=2_000_000)
        collector.add(t1)
        collector.add(t2)
        assert len(collector.traces) == 2
        assert collector.traces[0].operation == "a"

    def test_traces_returns_copy(self) -> None:
        collector = TraceCollector()
        collector.add(PipelineTrace(operation="x"))
        traces = collector.traces
        traces.clear()
        assert len(collector.traces) == 1

    def test_clear(self) -> None:
        collector = TraceCollector()
        collector.add(PipelineTrace(operation="x"))
        collector.clear()
        assert len(collector.traces) == 0

    def test_summary(self) -> None:
        collector = TraceCollector()
        collector.add(PipelineTrace(operation="query", start_ns=0, end_ns=5_000_000))
        summary = collector.summary()
        assert len(summary) == 1
        assert summary[0]["operation"] == "query"


class TestTracedStep:
    def test_records_timing(self) -> None:
        trace = PipelineTrace(operation="test")
        with traced_step(trace, "work") as step:
            time.sleep(0.001)
            step.metadata["items"] = 10
        assert len(trace.steps) == 1
        assert trace.steps[0].name == "work"
        assert trace.steps[0].duration_ms > 0
        assert trace.steps[0].metadata["items"] == 10

    def test_step_appended_after_block(self) -> None:
        trace = PipelineTrace(operation="test")
        with traced_step(trace, "a"):
            pass
        with traced_step(trace, "b"):
            pass
        assert [s.name for s in trace.steps] == ["a", "b"]


class TestTracedOperation:
    def test_creates_trace_with_timestamps(self) -> None:
        with traced_operation("query") as trace:
            time.sleep(0.001)
        assert trace.operation == "query"
        assert trace.start_ns > 0
        assert trace.end_ns > trace.start_ns
        assert trace.total_ms > 0

    def test_appends_to_collector(self) -> None:
        collector = TraceCollector()
        with traced_operation("query", collector=collector):
            pass
        assert len(collector.traces) == 1
        assert collector.traces[0].operation == "query"

    def test_no_collector(self) -> None:
        with traced_operation("query") as trace:
            pass
        assert trace.total_ms >= 0

    def test_nested_steps(self) -> None:
        with traced_operation("query") as trace:
            with traced_step(trace, "search"):
                time.sleep(0.001)
            with traced_step(trace, "assemble"):
                pass
        assert len(trace.steps) == 2
        assert trace.steps[0].name == "search"
        assert trace.steps[1].name == "assemble"
        assert trace.total_ms >= trace.steps[0].duration_ms
