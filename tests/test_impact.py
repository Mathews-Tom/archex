"""Backend tests for diff-scoped impact analysis (git_diff_hunks, resolve_diff_symbols)."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from archex.api import index_repository
from archex.impact import git_changed_files, git_diff_hunks, resolve_diff_symbols
from archex.models import Config, IndexConfig, RepoSource

if TYPE_CHECKING:
    from pathlib import Path

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
