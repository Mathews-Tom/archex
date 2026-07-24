"""M11: guard the checked-in MCP tool-schema size baseline.

Asserts the current, live `measure_tool_schema_size()` output for each
scope profile matches the final recorded stage
(`benchmarks/results/m11_mcp_schema_overhead/BASELINE.json`'s
`stages.pr3_graph_query`) -- a future PR that re-bloats a tool
description, or silently changes the tool set, fails loudly here
instead of eroding the M11 result unnoticed.

`all` (every registered tool, including the five deprecated `graph_*`
tools kept for M11's no-removal constraint) is *larger* than the
pre-M11 baseline -- adding a real new tool (`graph_query`) without
removing anything can only grow the unscoped total. The `core` scope
(everything except the five raw `graph_*` tools) is the one that must
decrease: it picks up `graph_query` automatically and lands below the
original pre-M11 unscoped baseline, which is M11's actual objective --
a properly-scoped client pays less than the original full surface.
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
        assert set(baseline["stages"]) == {  # type: ignore[arg-type]
            "pr1_tool_scoping",
            "pr2_trimmed_descriptions",
            "pr3_graph_query",
        }

    @pytest.mark.parametrize("scope_name", ["all", "core", "graph"])
    def test_current_size_matches_final_recorded_stage(self, scope_name: str) -> None:
        baseline = _load_baseline()
        final_stage = baseline["stages"]["pr3_graph_query"]  # type: ignore[index]
        recorded = final_stage[scope_name]  # type: ignore[index]

        tool_names = resolve_tool_scope(None if scope_name == "all" else scope_name)
        current = measure_tool_schema_size(tool_names)

        assert current["tool_count"] == recorded["tool_count"]  # type: ignore[index]
        assert current["total_chars"] == recorded["total_chars"]  # type: ignore[index]

    def test_core_scope_size_decreased_below_pre_m11_unscoped_baseline(self) -> None:
        """The M11 objective's actual target: a client that scopes to 'core'
        (drops the five raw graph_* tools, keeps graph_query) is not charged
        more than the original pre-M11 unscoped surface -- it pays less."""
        baseline = _load_baseline()
        pre_m11_total = baseline["pre_m11_unscoped_baseline"]["total_chars"]  # type: ignore[index]

        current_core = measure_tool_schema_size(resolve_tool_scope("core"))
        assert current_core["total_chars"] < pre_m11_total

    def test_graph_only_scope_size_decreased_from_pr1_baseline(self) -> None:
        """PR-2's description trims hold for the graph scope regardless of
        graph_query's addition -- the five graph_* tools are untouched by PR-3."""
        baseline = _load_baseline()
        pr1_graph_total = baseline["stages"]["pr1_tool_scoping"]["graph"]["total_chars"]  # type: ignore[index]

        current_graph = measure_tool_schema_size(resolve_tool_scope("graph"))
        assert current_graph["total_chars"] < pr1_graph_total

    def test_unscoped_all_growth_is_exactly_graph_query_no_removal_no_surprise(self) -> None:
        """'all' necessarily grows once graph_query is added without removing
        the five originals (M11's own no-removal constraint) -- but growth
        must be exactly graph_query's own schema size, never more (a
        regression here means some *other* tool grew too, not just the
        expected new one)."""
        baseline = _load_baseline()
        pr2_all_total = baseline["stages"]["pr2_trimmed_descriptions"]["all"]["total_chars"]  # type: ignore[index]
        graph_query_total = baseline["stages"]["pr3_graph_query"]["graph_query"]["total_chars"]  # type: ignore[index]

        current_all = measure_tool_schema_size(None)
        assert current_all["total_chars"] == pr2_all_total + graph_query_total

    def test_per_tool_chars_final_matches_current_unscoped_measurement(self) -> None:
        baseline = _load_baseline()
        recorded_per_tool = baseline["per_tool_chars_final"]
        current = measure_tool_schema_size(None)
        assert current["per_tool_chars"] == recorded_per_tool
