"""Benchmark data models: tasks, results, reports, and strategy enum."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from archex.models import PipelineTiming  # noqa: TCH001 — Pydantic needs at runtime


class Strategy(StrEnum):
    RAW_FILES = "raw_files"
    RAW_GREPPED = "raw_grepped"
    ARCHEX_QUERY = "archex_query"
    ARCHEX_QUERY_HYBRID = "archex_query_hybrid"
    ARCHEX_SYMBOL_LOOKUP = "archex_symbol_lookup"


class BenchmarkTask(BaseModel):
    task_id: str
    repo: str
    commit: str
    question: str
    expected_files: list[str]
    expected_symbols: list[str] = []
    token_budget: int = 8192
    keywords: list[str] = []


class BenchmarkResult(BaseModel):
    task_id: str
    strategy: Strategy
    tokens_total: int
    tool_calls: int
    files_accessed: int
    recall: float
    precision: float
    symbol_recall: float = 0.0
    savings_vs_raw: float
    wall_time_ms: float
    cached: bool
    timing: PipelineTiming | None = None
    timestamp: str


class BenchmarkReport(BaseModel):
    task_id: str
    repo: str
    question: str
    results: list[BenchmarkResult]
    baseline_tokens: int
