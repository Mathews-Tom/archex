"""Contract tests for explicit project-session records and priming."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from archex.cache import CacheManager
from archex.cli.main import cli
from archex.config import load_config
from archex.index.delta import compute_working_tree_signature
from archex.index.store import IndexStore
from archex.integrations.mcp import handle_session
from archex.project import ProjectState, init_project
from archex.session import (
    SessionRecordKind,
    capture_session_record,
    list_session_records,
    render_session_primer,
)


def _make_fresh_index(repo: Path) -> None:
    init_project(repo)
    project = ProjectState.resolve(repo)
    config = load_config(project.repo_root)
    with IndexStore(project.index_path) as store:
        store.set_metadata("commit_hash", CacheManager.git_head(str(repo)) or "")
        store.set_metadata(
            "working_tree_signature",
            compute_working_tree_signature(project.repo_root, config),
        )
        store.clear_reindex_flag()


def test_primer_renders_explicit_records_and_rejects_stale_index(
    python_simple_repo: Path,
) -> None:
    _make_fresh_index(python_simple_repo)

    active = capture_session_record(
        python_simple_repo,
        kind=SessionRecordKind.ACTIVE_TASK,
        content="Repair the parser boundary.",
        creator="test",
    )
    decision = capture_session_record(
        python_simple_repo,
        kind=SessionRecordKind.DECISION,
        content="Keep the parser interface stable.",
        creator="test",
    )

    primer = render_session_primer(python_simple_repo, token_budget=256)

    constrained_primer = render_session_primer(python_simple_repo, token_budget=1)

    assert constrained_primer.ready is True
    assert constrained_primer.content == ""
    assert constrained_primer.receipt.consumed_budget == 0
    assert constrained_primer.receipt.recommended_next_action == "increase_budget"
    assert primer.ready is True
    assert primer.receipt.index_state == "fresh"
    assert primer.receipt.consumed_budget <= primer.receipt.requested_budget
    assert primer.receipt.included_record_ids == [active.id, decision.id]
    assert "Repair the parser boundary." in primer.content
    assert "Keep the parser interface stable." in primer.content

    (python_simple_repo / "main.py").write_text("changed = True\n", encoding="utf-8")

    stale_primer = render_session_primer(python_simple_repo, token_budget=256)

    assert stale_primer.ready is False
    assert stale_primer.content == ""
    assert stale_primer.receipt.index_state == "dirty"
    assert stale_primer.receipt.recommended_next_action == "refresh_index"


def test_active_task_replaces_prior_task_and_mcp_is_explicit(
    python_simple_repo: Path,
) -> None:
    _make_fresh_index(python_simple_repo)

    first = capture_session_record(
        python_simple_repo,
        kind=SessionRecordKind.ACTIVE_TASK,
        content="First task.",
        creator="test",
    )
    response = handle_session(
        str(python_simple_repo),
        "record",
        kind="active_task",
        content="Second task.",
    )
    second = json.loads(response)["record"]

    active_records = list_session_records(python_simple_repo)
    all_records = list_session_records(python_simple_repo, include_inactive=True)
    assert [record.content for record in active_records] == ["Second task."]
    assert second["content"] == "Second task."
    assert {record.id for record in all_records} == {first.id, second["id"]}
    first_record = next(record for record in all_records if record.id == first.id)
    assert first_record.state.value == "superseded"


def test_anchored_records_are_excluded_after_index_revision_changes(
    python_simple_repo: Path,
) -> None:
    _make_fresh_index(python_simple_repo)
    record = capture_session_record(
        python_simple_repo,
        kind=SessionRecordKind.DECISION,
        content="Anchor the decision to main.py.",
        creator="test",
        file_path="main.py",
    )
    project = ProjectState.resolve(python_simple_repo)
    with IndexStore(project.index_path) as store:
        store.set_metadata("chunk_count", "1")

    primer = render_session_primer(python_simple_repo, token_budget=256)

    assert primer.ready is True
    assert primer.records == []
    assert primer.receipt.included_record_ids == []
    assert primer.receipt.skipped_records[0].record_id == record.id
    assert primer.receipt.skipped_records[0].reason == "stale_index_revision"


def test_records_are_scoped_to_the_current_branch(python_simple_repo: Path) -> None:
    capture_session_record(
        python_simple_repo,
        kind=SessionRecordKind.BLOCKER,
        content="Only the original branch sees this.",
        creator="test",
    )
    subprocess.run(
        ["git", "checkout", "-b", "other-branch"],
        cwd=python_simple_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    assert list_session_records(python_simple_repo) == []


def test_session_cli_records_lists_and_requires_delete_confirmation(
    python_simple_repo: Path,
) -> None:
    runner = CliRunner()

    recorded = runner.invoke(
        cli,
        [
            "session",
            "record",
            "decision",
            "Use explicit user-approved state only.",
            "--source",
            str(python_simple_repo),
        ],
    )
    assert recorded.exit_code == 0, recorded.output
    record_id = json.loads(recorded.output)["id"]
    listed = runner.invoke(cli, ["session", "list", "--source", str(python_simple_repo)])
    assert listed.exit_code == 0, listed.output

    unconfirmed_delete = runner.invoke(
        cli,
        ["session", "delete", record_id, "--source", str(python_simple_repo)],
    )
    deleted = runner.invoke(
        cli,
        ["session", "delete", record_id, "--source", str(python_simple_repo), "--force"],
    )

    assert json.loads(listed.output)[0]["id"] == record_id
    assert unconfirmed_delete.exit_code != 0
    assert "requires --force" in unconfirmed_delete.output
    assert deleted.exit_code == 0, deleted.output
