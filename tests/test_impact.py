"""Backend tests for diff-scoped impact analysis.

Covers diff-to-symbol resolution (git_diff_hunks, resolve_diff_symbols) and
risk classification (_classify_file_risk, analyze_diff_impact).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from archex.api import index_repository
from archex.impact import (
    ImpactReport,
    SymbolImpact,
    SymbolRiskLevel,
    SymbolRiskSignal,
    _classify_file_risk,  # pyright: ignore[reportPrivateUsage]
    analyze_diff_impact,
    git_changed_files,
    git_diff_hunks,
    resolve_diff_symbols,
)
from archex.index.graph import DependencyGraph
from archex.models import Config, IndexConfig, RepoSource
from archex.parse.adapters import default_adapter_registry

if TYPE_CHECKING:
    from archex.index.store import IndexStore


def _index(repo_path: Path) -> IndexStore:
    source = RepoSource(local_path=str(repo_path), stable_identity="impact-diff-test@1")
    return index_repository(
        source, config=Config(cache=False), index_config=IndexConfig(vector=False)
    )


# ---------------------------------------------------------------------------
# git_diff_hunks
# ---------------------------------------------------------------------------


def test_git_diff_hunks_maps_single_line_edit(impact_diff_repo: Path) -> None:
    hub = impact_diff_repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))

    hunks = git_diff_hunks(impact_diff_repo, "HEAD")

    assert hunks == {"hub.py": [(5, 5)]}


def test_git_diff_hunks_covers_whole_added_file(impact_diff_repo: Path) -> None:
    new_file = impact_diff_repo / "extra.py"
    new_file.write_text("def extra() -> int:\n    return 1\n")
    subprocess.run(["git", "add", "extra.py"], cwd=impact_diff_repo, check=True)

    hunks = git_diff_hunks(impact_diff_repo, "HEAD")

    assert hunks == {"extra.py": [(1, 2)]}


def test_git_diff_hunks_skips_deleted_files(impact_diff_repo: Path) -> None:
    (impact_diff_repo / "leaf.py").unlink()

    hunks = git_diff_hunks(impact_diff_repo, "HEAD")

    assert "leaf.py" not in hunks


def test_git_diff_hunks_records_pure_deletion_as_point(impact_diff_repo: Path) -> None:
    hub = impact_diff_repo / "hub.py"
    lines = hub.read_text().splitlines(keepends=True)
    # Drop the blank separator line between the two functions (a pure deletion,
    # zero new lines added at that position).
    del lines[6]
    hub.write_text("".join(lines))

    hunks = git_diff_hunks(impact_diff_repo, "HEAD")

    assert hunks == {"hub.py": [(6, 6)]}


# ---------------------------------------------------------------------------
# resolve_diff_symbols
# ---------------------------------------------------------------------------


def test_resolve_diff_symbols_hub_file_touches_only_changed_function(
    impact_diff_repo: Path,
) -> None:
    hub = impact_diff_repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        touched = resolve_diff_symbols(store, changes, hunks)
    finally:
        store.close()

    assert [(chunk.file_path, chunk.symbol_name) for chunk in touched] == [
        ("hub.py", "shared_helper")
    ]


def test_resolve_diff_symbols_leaf_file_touches_only_changed_function(
    impact_diff_repo: Path,
) -> None:
    leaf = impact_diff_repo / "leaf.py"
    old_text = leaf.read_text()
    leaf.write_text(old_text.replace("shared_helper(value) - 1", "shared_helper(value) - 2"))

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        touched = resolve_diff_symbols(store, changes, hunks)
    finally:
        store.close()

    assert [(chunk.file_path, chunk.symbol_name) for chunk in touched] == [("leaf.py", "isolated")]


def test_resolve_diff_symbols_added_file_reports_all_symbols(impact_diff_repo: Path) -> None:
    new_file = impact_diff_repo / "extra.py"
    new_file.write_text(
        "def extra_a() -> int:\n    return 1\n\n\ndef extra_b() -> int:\n    return 2\n"
    )
    subprocess.run(["git", "add", "extra.py"], cwd=impact_diff_repo, check=True)

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        touched = resolve_diff_symbols(store, changes, hunks)
    finally:
        store.close()

    assert {chunk.symbol_name for chunk in touched} == {"extra_a", "extra_b"}


def test_resolve_diff_symbols_skips_deleted_files(impact_diff_repo: Path) -> None:
    (impact_diff_repo / "leaf.py").unlink()

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        touched = resolve_diff_symbols(store, changes, hunks)
    finally:
        store.close()

    assert touched == []


def test_resolve_diff_symbols_ignores_untouched_symbols_in_changed_file(
    impact_diff_repo: Path,
) -> None:
    """Editing one function in hub.py must not report the sibling function as touched."""
    hub = impact_diff_repo / "hub.py"
    hub.write_text(hub.read_text().replace("value + 1", "value + 2"))

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        touched = resolve_diff_symbols(store, changes, hunks)
    finally:
        store.close()

    symbol_names = {chunk.symbol_name for chunk in touched}
    assert symbol_names == {"other_helper"}
    assert "shared_helper" not in symbol_names


# ---------------------------------------------------------------------------
# _classify_file_risk -- tier boundaries
# ---------------------------------------------------------------------------


def test_classify_file_risk_zero_signals_is_low() -> None:
    graph = DependencyGraph()
    graph.add_file_node("solo.py")

    level, signals = _classify_file_risk(
        "solo.py",
        graph,
        centrality={"solo.py": 0.1},
        mean_centrality=0.1,
        repo_root=Path("/nonexistent"),
        path_to_language={},
        adapters={},
    )

    assert level == SymbolRiskLevel.LOW
    assert {signal.name for signal in signals} == {
        "structural_centrality",
        "fan_in",
        "entry_point_proximity",
    }


def test_classify_file_risk_one_signal_fan_in_is_medium() -> None:
    graph = DependencyGraph()
    for i in range(3):
        graph.add_file_edge(f"importer_{i}.py", "target.py")

    level, _signals = _classify_file_risk(
        "target.py",
        graph,
        centrality={"target.py": 0.1},
        mean_centrality=0.1,  # ratio 1.0x, below the 2.0x hub threshold
        repo_root=Path("/nonexistent"),
        path_to_language={},
        adapters={},
    )

    assert level == SymbolRiskLevel.MEDIUM


def test_classify_file_risk_two_signals_is_high() -> None:
    graph = DependencyGraph()
    for i in range(3):
        graph.add_file_edge(f"importer_{i}.py", "target.py")

    level, _signals = _classify_file_risk(
        "target.py",
        graph,
        centrality={"target.py": 0.5},
        mean_centrality=0.1,  # ratio 5.0x, above the 2.0x hub threshold
        repo_root=Path("/nonexistent"),
        path_to_language={},
        adapters={},
    )

    assert level == SymbolRiskLevel.HIGH


def test_classify_file_risk_entry_point_itself_forces_high(tmp_path: Path) -> None:
    (tmp_path / "main_entry.py").write_text('if __name__ == "__main__":\n    pass\n')
    graph = DependencyGraph()
    graph.add_file_node("main_entry.py")
    adapters = default_adapter_registry.build_all()

    level, signals = _classify_file_risk(
        "main_entry.py",
        graph,
        centrality={"main_entry.py": 0.01},
        mean_centrality=0.5,  # deliberately high so centrality does not fire
        repo_root=tmp_path,
        path_to_language={"main_entry.py": "python"},
        adapters=adapters,
    )

    assert level == SymbolRiskLevel.HIGH
    entry_signal = next(s for s in signals if s.name == "entry_point_proximity")
    assert "entry_point_distance=0" in entry_signal.detail


def test_classify_file_risk_entry_proximity_alone_is_medium(tmp_path: Path) -> None:
    (tmp_path / "entry.py").write_text('if __name__ == "__main__":\n    pass\n')
    (tmp_path / "target.py").write_text("value = 1\n")
    graph = DependencyGraph()
    graph.add_file_edge("entry.py", "target.py")
    adapters = default_adapter_registry.build_all()

    level, signals = _classify_file_risk(
        "target.py",
        graph,
        centrality={"target.py": 0.01, "entry.py": 0.01},
        mean_centrality=0.5,
        repo_root=tmp_path,
        path_to_language={"target.py": "python", "entry.py": "python"},
        adapters=adapters,
    )

    assert level == SymbolRiskLevel.MEDIUM
    entry_signal = next(s for s in signals if s.name == "entry_point_proximity")
    assert "entry_point_distance=1" in entry_signal.detail


def test_classify_file_risk_entry_point_beyond_max_depth_does_not_fire() -> None:
    graph = DependencyGraph()
    graph.add_file_edge("entry.py", "mid.py")
    graph.add_file_edge("mid.py", "far.py")
    graph.add_file_edge("far.py", "target.py")

    level, signals = _classify_file_risk(
        "target.py",
        graph,
        centrality={"target.py": 0.1},
        mean_centrality=0.1,
        repo_root=Path("/nonexistent"),
        path_to_language={},  # no language info -> entry detection never fires anyway
        adapters={},
    )

    assert level == SymbolRiskLevel.LOW
    entry_signal = next(s for s in signals if s.name == "entry_point_proximity")
    assert "no entry point within max_depth" in entry_signal.detail


# ---------------------------------------------------------------------------
# analyze_diff_impact -- end-to-end acceptance (hub file -> HIGH, leaf -> LOW)
# ---------------------------------------------------------------------------


def test_analyze_diff_impact_hub_file_edit_is_high_risk(impact_diff_repo: Path) -> None:
    hub = impact_diff_repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        report = analyze_diff_impact(store, impact_diff_repo, changes, hunks, "HEAD")
    finally:
        store.close()

    assert report.diff_ref == "HEAD"
    assert [(s.file_path, s.symbol_name, s.level) for s in report.affected_symbols] == [
        ("hub.py", "shared_helper", SymbolRiskLevel.HIGH)
    ]
    fired = report.affected_symbols[0].signals
    assert {signal.name for signal in fired} == {
        "structural_centrality",
        "fan_in",
        "entry_point_proximity",
    }


def test_analyze_diff_impact_leaf_file_edit_is_low_risk(impact_diff_repo: Path) -> None:
    leaf = impact_diff_repo / "leaf.py"
    old_text = leaf.read_text()
    leaf.write_text(old_text.replace("shared_helper(value) - 1", "shared_helper(value) - 2"))

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        report = analyze_diff_impact(store, impact_diff_repo, changes, hunks, "HEAD")
    finally:
        store.close()

    assert [(s.file_path, s.symbol_name, s.level) for s in report.affected_symbols] == [
        ("leaf.py", "isolated", SymbolRiskLevel.LOW)
    ]


def test_analyze_diff_impact_no_touched_symbols_still_sets_diff_ref(
    impact_diff_repo: Path,
) -> None:
    (impact_diff_repo / "README.md").write_text("docs only\n")
    subprocess.run(["git", "add", "README.md"], cwd=impact_diff_repo, check=True)

    changes = git_changed_files(impact_diff_repo, "HEAD")
    hunks = git_diff_hunks(impact_diff_repo, "HEAD")
    store = _index(impact_diff_repo)
    try:
        report = analyze_diff_impact(store, impact_diff_repo, changes, hunks, "HEAD")
    finally:
        store.close()

    assert report.diff_ref == "HEAD"
    assert report.affected_symbols == []


# ---------------------------------------------------------------------------
# ImpactReport rendering -- additive-only when diff mode is unused
# ---------------------------------------------------------------------------


def test_impact_report_to_json_excludes_diff_fields_when_not_used() -> None:
    payload = json.loads(ImpactReport().to_json())

    assert "diff_ref" not in payload
    assert "affected_symbols" not in payload


def test_impact_report_to_json_includes_diff_fields_when_used() -> None:
    payload = json.loads(ImpactReport(diff_ref="HEAD").to_json())

    assert payload["diff_ref"] == "HEAD"
    assert payload["affected_symbols"] == []


def test_impact_report_to_markdown_omits_symbol_risk_section_when_not_used() -> None:
    assert "Symbol Risk" not in ImpactReport().to_markdown()


def test_impact_report_to_markdown_includes_symbol_risk_section_when_used() -> None:
    report = ImpactReport(
        diff_ref="HEAD",
        affected_symbols=[
            SymbolImpact(
                file_path="hub.py",
                symbol_name="shared_helper",
                start_line=4,
                end_line=5,
                level=SymbolRiskLevel.HIGH,
                signals=[SymbolRiskSignal(name="fan_in", detail="fan_in=5 >= threshold=3")],
            )
        ],
    )

    markdown = report.to_markdown()

    assert "Symbol Risk" in markdown
    assert "shared_helper" in markdown
    assert "high" in markdown
