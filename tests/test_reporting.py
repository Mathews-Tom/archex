"""Tests for reporting module: token counting and efficiency metrics."""

from __future__ import annotations

from archex.models import PipelineTiming, TokenMeta
from archex.reporting import compute_meta, count_tokens, print_savings, print_timing


def test_count_tokens_known_string() -> None:
    text = "def hello(): pass"
    result = count_tokens(text)
    assert result > 0
    assert isinstance(result, int)


def test_count_tokens_empty_string() -> None:
    assert count_tokens("") == 0


def test_count_tokens_treats_special_token_literals_as_text() -> None:
    result = count_tokens('return "<|endoftext|>"')

    assert result > 0


def test_compute_meta_basic() -> None:
    meta = compute_meta(
        tool_name="test_tool",
        response_text="short response",
        raw_file_tokens=1000,
        strategy="bm25",
    )
    assert isinstance(meta, TokenMeta)
    assert meta.tool_name == "test_tool"
    assert meta.strategy == "bm25"
    assert meta.tokens_returned > 0
    assert meta.tokens_raw_equivalent == 1000
    assert meta.savings_pct > 0
    assert meta.cached is False
    assert meta.index_time_ms == 0.0
    assert meta.query_time_ms == 0.0


def test_compute_meta_subtracts_envelope_overhead_from_savings() -> None:
    response_text = "wrapper tokens around payload"
    returned = count_tokens(response_text)
    meta = compute_meta(
        tool_name="query_repo",
        response_text=response_text,
        raw_file_tokens=10,
        envelope_overhead_tokens=returned - 1,
        strategy="bm25+graph",
    )

    assert meta.tokens_returned == 1
    assert meta.savings_pct == 90.0


def test_compute_meta_zero_raw() -> None:
    meta = compute_meta(
        tool_name="t",
        response_text="something",
        raw_file_tokens=0,
        strategy="s",
    )
    assert meta.savings_pct == 0.0


def test_compute_meta_cached_flag() -> None:
    meta = compute_meta(
        tool_name="t",
        response_text="x",
        raw_file_tokens=100,
        strategy="s",
        cached=True,
    )
    assert meta.cached is True


def test_compute_meta_timing_fields() -> None:
    meta = compute_meta(
        tool_name="t",
        response_text="x",
        raw_file_tokens=100,
        strategy="s",
        index_time_ms=12.345,
        query_time_ms=6.789,
    )
    assert meta.index_time_ms == 12.3
    assert meta.query_time_ms == 6.8


def test_print_savings_format() -> None:
    import io
    import sys

    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        print_savings(returned=100, raw=1000, elapsed_ms=42.5)
    finally:
        sys.stderr = old_stderr

    output = captured.getvalue()
    assert "[savings] 100 tokens returned" in output
    assert "[savings] Raw equivalent: 1,000 tokens" in output
    assert "[savings] Saved 90.0% vs raw file access" in output
    assert "[timing] 42ms total" in output


def test_print_savings_zero_raw() -> None:
    import io
    import sys

    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        print_savings(returned=50, raw=0, elapsed_ms=10.0)
    finally:
        sys.stderr = old_stderr

    output = captured.getvalue()
    assert "Saved 0.0%" in output


def test_print_savings_with_budget_and_file_count() -> None:
    import io
    import sys

    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        print_savings(returned=7891, raw=48320, elapsed_ms=12.0, budget=8192, file_count=6)
    finally:
        sys.stderr = old_stderr

    output = captured.getvalue()
    assert "(budget: 8,192)" in output
    assert "across 6 files" in output
    assert "7,891 tokens returned" in output
    assert "48,320 tokens" in output


def test_print_timing_cached() -> None:
    import io
    import sys

    pt = PipelineTiming(cached=True, search_ms=5.0, assemble_ms=3.0, total_ms=8.0)
    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        print_timing(pt)
    finally:
        sys.stderr = old_stderr

    output = captured.getvalue()
    assert "Cache hit -- skipped parse" in output
    assert "Search + assemble in 8ms" in output
    assert "Acquired repo" not in output
    assert "Parsed" not in output


def test_print_timing_cache_miss() -> None:
    import io
    import sys

    pt = PipelineTiming(
        acquire_ms=10.0,
        parse_ms=50.0,
        index_ms=20.0,
        search_ms=3.0,
        assemble_ms=2.0,
        total_ms=85.0,
        cached=False,
    )
    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        print_timing(pt)
    finally:
        sys.stderr = old_stderr

    output = captured.getvalue()
    assert "Cache hit" not in output
    assert "Acquired repo in 10ms" in output
    assert "Parsed + indexed in 50ms" in output
    assert "Search + assemble in 5ms" in output
