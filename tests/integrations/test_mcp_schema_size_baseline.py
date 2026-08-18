"""Guard recorded MCP tool-schema measurements.

The M11 stages preserve the original tool-scoping experiment. Intentional
post-M11 additions are recorded separately, while ``current_surface`` records
the live surface those additions produce. Exact measurements keep the test
detecting silent schema growth without rewriting historical results.

The ``all`` scope includes every registered tool. The ``core`` scope excludes
only the five raw ``graph_*`` tools. A new MCP tool therefore changes both
``all`` and ``core`` unless it belongs to that graph-only set.
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
        assert set(baseline["post_m11_additions"]) == {"session"}  # type: ignore[arg-type]
        assert set(baseline["current_surface"]) == {  # type: ignore[arg-type]
            "measured_at",
            "source_revision",
            "all",
            "core",
            "graph",
            "per_tool_chars",
        }

    @pytest.mark.parametrize("scope_name", ["all", "core", "graph"])
    def test_current_size_matches_recorded_current_stage(self, scope_name: str) -> None:
        baseline = _load_baseline()
        current_surface = baseline["current_surface"]  # type: ignore[index]
        recorded = current_surface[scope_name]  # type: ignore[index]

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

    def test_unscoped_all_growth_matches_known_additions(self) -> None:
        """Every post-M11 addition is separately accounted for."""
        baseline = _load_baseline()
        pr2_all_total = baseline["stages"]["pr2_trimmed_descriptions"]["all"]["total_chars"]  # type: ignore[index]
        graph_query_total = baseline["stages"]["pr3_graph_query"]["graph_query"]["total_chars"]  # type: ignore[index]
        session_total = baseline["post_m11_additions"]["session"]["total_chars"]  # type: ignore[index]

        current_all = measure_tool_schema_size(None)
        assert current_all["total_chars"] == pr2_all_total + graph_query_total + session_total

    def test_per_tool_chars_current_matches_unscoped_measurement(self) -> None:
        baseline = _load_baseline()
        recorded_per_tool = baseline["current_surface"]["per_tool_chars"]  # type: ignore[index]
        current = measure_tool_schema_size(None)
        assert current["per_tool_chars"] == recorded_per_tool
