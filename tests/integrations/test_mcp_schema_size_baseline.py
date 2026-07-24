"""M11 PR-2: guard the checked-in MCP tool-schema size baseline.

Asserts the current, live `measure_tool_schema_size()` output for each
scope profile recorded in `benchmarks/results/m11_mcp_schema_overhead/
BASELINE.json` never exceeds the checked-in `after` figure -- a future
PR that re-bloats a tool description (or adds a new tool without
trimming elsewhere) fails loudly here instead of silently eroding the
M11 reduction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

from archex.integrations.mcp import measure_tool_schema_size, resolve_tool_scope

_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "results"
    / "m11_mcp_schema_overhead"
    / "BASELINE.json"
)


def _load_baseline() -> dict[str, object]:
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


class TestSchemaSizeBaseline:
    def test_baseline_file_exists_and_parses(self) -> None:
        baseline = _load_baseline()
        assert baseline["milestone"] == "M11"
        assert set(baseline["profiles"]) == {"all", "core", "graph"}  # type: ignore[arg-type]

    @pytest.mark.parametrize("scope_name", ["all", "core", "graph"])
    def test_current_size_does_not_exceed_recorded_after(self, scope_name: str) -> None:
        baseline = _load_baseline()
        profile = baseline["profiles"][scope_name]  # type: ignore[index]
        recorded_after = profile["after"]  # type: ignore[index]

        tool_names = resolve_tool_scope(None if scope_name == "all" else scope_name)
        current = measure_tool_schema_size(tool_names)

        assert current["tool_count"] == recorded_after["tool_count"]  # type: ignore[index]
        assert current["total_chars"] <= recorded_after["total_chars"]  # type: ignore[index]

    def test_unscoped_size_decreased_from_recorded_before(self) -> None:
        """The M11 acceptance row: the full unscoped listing's total schema
        size decreases from the pre-change (PR-1) baseline."""
        baseline = _load_baseline()
        before_total = baseline["profiles"]["all"]["before"]["total_chars"]  # type: ignore[index]

        current = measure_tool_schema_size(None)
        assert current["total_chars"] < before_total

    def test_per_tool_chars_after_matches_current_unscoped_measurement(self) -> None:
        baseline = _load_baseline()
        recorded_per_tool = baseline["per_tool_chars_after"]
        current = measure_tool_schema_size(None)
        assert current["per_tool_chars"] == recorded_per_tool
