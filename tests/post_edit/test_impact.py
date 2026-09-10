"""Post-edit impact emission: freshness, bounds, fallbacks, and wording."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.api import index_repository
from archex.cli.main import cli
from archex.config import load_config, load_index_config
from archex.impact import ImpactFileChange, ImpactReport, ImpactRisk, ImpactRiskLevel
from archex.index.delta import compute_working_tree_signature
from archex.models import PipelineTiming, RepoSource
from archex.post_edit import (
    PostEditOutcome,
    build_event,
    read_state,
    record_edit,
    render_post_edit_block,
    synchronize_and_report,
    synchronize_and_report_with_timeout,
)
from archex.post_edit.impact import (
    DEFAULT_POST_EDIT_TIMEOUT_SECONDS,
    MAX_RENDERED_AFFECTED_FILES,
    _validate_generation,  # pyright: ignore[reportPrivateUsage]
    post_edit_timeout_seconds,
)
from archex.post_edit.models import PostEditStatus
from archex.project import init_project

if TYPE_CHECKING:
    import pytest


def _indexed(repo: Path) -> Path:
    init_project(repo)
    result = CliRunner().invoke(cli, ["index", str(repo)])
    assert result.exit_code == 0, result.output
    return repo


def _record(repo: Path, *paths: str) -> None:
    event, _rejected = build_event(
        client="claude-code", tool_name="Edit", repo_root=repo, raw_paths=list(paths)
    )
    record_edit(repo, event)


def _first_source_file(repo: Path) -> str:
    candidates = sorted(path for path in repo.rglob("*.py") if ".archex" not in path.parts)
    assert candidates, "fixture has no python sources"
    return candidates[0].relative_to(repo).as_posix()


# ---------------------------------------------------------------------------
# Fresh path
# ---------------------------------------------------------------------------


def test_a_recorded_edit_synchronizes_and_emits_a_fresh_receipt(
    python_simple_repo: Path,
) -> None:
    repo = _indexed(python_simple_repo)
    target = _first_source_file(repo)
    (repo / target).write_text((repo / target).read_text() + "\n# post-edit marker\n")
    _record(repo, target)

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.EMITTED
    assert feedback.generation_id
    assert feedback.index_revision
    assert feedback.synchronized_paths == [target]
    assert feedback.text is not None
    assert "[archex post-edit receipt]" in feedback.text
    assert f"generation={feedback.generation_id[:12]}" in feedback.text
    assert target in feedback.text


def test_emission_retires_the_pending_state(python_simple_repo: Path) -> None:
    repo = _indexed(python_simple_repo)
    target = _first_source_file(repo)
    (repo / target).write_text((repo / target).read_text() + "\n# marker\n")
    _record(repo, target)

    feedback = synchronize_and_report(repo, client="omp")

    state = read_state(repo)
    assert state.status is PostEditStatus.CLEAN
    assert state.pending_paths == []
    assert state.synchronized_generation == feedback.generation_id


def test_no_recorded_edit_produces_no_output_and_no_refresh(
    python_simple_repo: Path,
) -> None:
    repo = _indexed(python_simple_repo)

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.NO_PENDING_EDITS
    assert feedback.text is None


def test_a_full_reindex_fallback_still_emits_fresh_impact(
    python_simple_repo: Path,
) -> None:
    """A change ratio past `delta_threshold` must fall back to a full rebuild.

    Asserting only that impact was emitted would pass even if the delta path
    silently kept being taken, so the strategy the pipeline actually chose is
    observed directly through `PipelineTiming` before the post-edit assertion.
    """
    repo = _indexed(python_simple_repo)
    settings = repo / ".archex" / "settings.toml"
    settings.write_text(
        settings.read_text().replace("delta_threshold = 0.5", "delta_threshold = 0.01")
    )
    sources = sorted(path for path in repo.rglob("*.py") if ".archex" not in path.parts)
    for path in sources:
        path.write_text(path.read_text() + "\n# widened\n")

    source = RepoSource(local_path=str(repo))
    timing = PipelineTiming()
    index_repository(
        source,
        config=load_config(source),
        timing=timing,
        index_config=load_index_config(source),
    ).close()
    assert timing.strategy == "full"

    for path in sources:
        path.write_text(path.read_text() + "\n# widened again\n")
    _record(repo, *[path.relative_to(repo).as_posix() for path in sources])

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.EMITTED
    assert read_state(repo).status is PostEditStatus.CLEAN


def test_a_deleted_file_is_reported_without_raising(python_simple_repo: Path) -> None:
    repo = _indexed(python_simple_repo)
    sources = sorted(path for path in repo.rglob("*.py") if ".archex" not in path.parts)
    assert len(sources) > 1
    victim = sources[-1]
    relative = victim.relative_to(repo).as_posix()
    _record(repo, relative)
    victim.unlink()

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.EMITTED
    assert feedback.text is not None
    assert relative in feedback.text


# ---------------------------------------------------------------------------
# Fail-closed paths
# ---------------------------------------------------------------------------


def test_a_generation_that_no_longer_matches_the_tree_emits_nothing(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _indexed(python_simple_repo)
    target = _first_source_file(repo)
    (repo / target).write_text((repo / target).read_text() + "\n# marker\n")
    _record(repo, target)

    def _moved_on(*_args: object, **_kwargs: object) -> str:
        return "signature-of-a-tree-that-moved-on"

    monkeypatch.setattr("archex.post_edit.impact.compute_working_tree_signature", _moved_on)

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.STALE
    assert feedback.text is None
    assert read_state(repo).status is PostEditStatus.DIRTY


def test_a_store_without_a_generation_identity_emits_nothing(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _indexed(python_simple_repo)
    _record(repo, _first_source_file(repo))

    def _no_identity(_store: object) -> str | None:
        return None

    monkeypatch.setattr("archex.post_edit.impact.read_generation_id", _no_identity)

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.STALE
    assert feedback.detail == "no persisted generation id"
    assert read_state(repo).status is PostEditStatus.DIRTY


def test_a_refresh_failure_degrades_to_no_output(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _indexed(python_simple_repo)
    _record(repo, _first_source_file(repo))

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr("archex.post_edit.impact.index_repository", _boom)

    feedback = synchronize_and_report(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.UNAVAILABLE
    assert feedback.text is None
    assert read_state(repo).status is PostEditStatus.DIRTY


def test_an_uninitialized_repository_degrades_to_no_output(tmp_path: Path) -> None:
    _record(tmp_path, "a.py")

    feedback = synchronize_and_report(tmp_path, client="claude-code")

    assert feedback.outcome in {PostEditOutcome.UNAVAILABLE, PostEditOutcome.STALE}
    assert feedback.text is None


def test_a_cycle_exceeding_its_budget_times_out_without_output(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    monkeypatch.setenv("ARCHEX_POST_EDIT_TIMEOUT_SECONDS", "0.05")
    repo = _indexed(python_simple_repo)
    _record(repo, _first_source_file(repo))

    def _slow(*_args: object, **_kwargs: object) -> object:
        time.sleep(5)
        raise AssertionError("should have been abandoned")

    monkeypatch.setattr("archex.post_edit.impact.index_repository", _slow)

    started = time.monotonic()
    feedback = synchronize_and_report_with_timeout(repo, client="claude-code")
    elapsed = time.monotonic() - started

    assert feedback.outcome is PostEditOutcome.TIMED_OUT
    assert feedback.text is None
    assert elapsed < 2.0
    assert "post_edit_timeout" in log.read_text()
    assert read_state(repo).status is PostEditStatus.DIRTY


def test_timeout_budget_reads_the_environment_and_rejects_nonsense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARCHEX_POST_EDIT_TIMEOUT_SECONDS", raising=False)
    assert post_edit_timeout_seconds() == DEFAULT_POST_EDIT_TIMEOUT_SECONDS

    monkeypatch.setenv("ARCHEX_POST_EDIT_TIMEOUT_SECONDS", "1.5")
    assert post_edit_timeout_seconds() == 1.5

    for bogus in ("nope", "0", "-3"):
        monkeypatch.setenv("ARCHEX_POST_EDIT_TIMEOUT_SECONDS", bogus)
        assert post_edit_timeout_seconds() == DEFAULT_POST_EDIT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Projection wording and bounds
# ---------------------------------------------------------------------------


def _report(**overrides: object) -> ImpactReport:
    base: dict[str, object] = {
        "changed_files": [ImpactFileChange(path="src/a.py", status="M")],
        "affected_files": ["src/a.py", "src/b.py"],
        "affected_tests": ["tests/test_a.py"],
        "risk": ImpactRisk(level=ImpactRiskLevel.MODERATE, reasons=["fan-in"]),
    }
    base.update(overrides)
    return ImpactReport.model_validate(base)


def test_rendered_block_labels_risk_as_file_scoped_and_never_call_graph() -> None:
    text = render_post_edit_block(
        _report(), client="omp", generation_id="g" * 64, index_revision="r" * 64
    )

    assert "file-scoped" in text
    assert "not from call-graph analysis" in text
    lowered = text.lower()
    assert "call graph" not in lowered.replace("call-graph analysis", "")
    assert "blast radius" not in lowered


def test_rendered_block_omits_the_edited_file_from_its_own_dependents() -> None:
    text = render_post_edit_block(
        _report(), client="omp", generation_id="g" * 64, index_revision="r" * 64
    )

    dependents = text.split("Files that depend on them")[1]
    assert "src/b.py" in dependents
    assert "src/a.py" not in dependents.split("Tests in the affected set")[0]


def test_a_complete_view_is_labelled_complete() -> None:
    text = render_post_edit_block(
        _report(), client="omp", generation_id="g" * 64, index_revision="r" * 64
    )

    assert "confidence=complete" in text


def test_unindexed_paths_are_stated_and_downgrade_confidence() -> None:
    text = render_post_edit_block(
        _report(unmapped_files=["docs/readme.md"]),
        client="omp",
        generation_id="g" * 64,
        index_revision="r" * 64,
    )

    assert "confidence=partial" in text
    assert "not indexed (1)" in text
    assert "docs/readme.md" in text


def test_dropped_edits_are_stated_and_downgrade_confidence() -> None:
    text = render_post_edit_block(
        _report(),
        client="omp",
        generation_id="g" * 64,
        index_revision="r" * 64,
        dropped_path_count=4,
    )

    assert "confidence=partial" in text
    assert "4 further edited path(s) exceeded the recorded-edit cap" in text


def test_long_lists_are_truncated_with_the_full_count_shown() -> None:
    affected = [f"src/dep{index:03d}.py" for index in range(MAX_RENDERED_AFFECTED_FILES + 7)]

    text = render_post_edit_block(
        _report(affected_files=affected),
        client="omp",
        generation_id="g" * 64,
        index_revision="r" * 64,
    )

    assert f"Files that depend on them ({len(affected)}, showing " in text
    assert "confidence=partial" in text
    assert text.count("src/dep") == MAX_RENDERED_AFFECTED_FILES


def test_empty_sections_say_none_rather_than_being_omitted() -> None:
    text = render_post_edit_block(
        _report(affected_files=["src/a.py"], affected_tests=[]),
        client="omp",
        generation_id="g" * 64,
        index_revision="r" * 64,
    )

    assert "Tests in the affected set: none." in text


def test_receipt_truncates_identities_to_a_stable_prefix() -> None:
    text = render_post_edit_block(
        _report(), client="cursor", generation_id="a" * 64, index_revision="b" * 64
    )

    header = text.splitlines()[0]
    assert "generation=" + "a" * 12 + " " in header
    assert "index_revision=" + "b" * 12 + " " in header
    assert "a" * 13 not in header
    assert "client=cursor" in header


def test_rendered_block_is_plain_text_not_json() -> None:
    text = render_post_edit_block(
        _report(), client="omp", generation_id="g" * 64, index_revision="r" * 64
    )

    try:
        json.loads(text)
    except json.JSONDecodeError:
        return
    raise AssertionError("post-edit block must be human-readable text, not a JSON document")


def test_a_commit_change_under_a_clean_tree_is_caught_as_stale(
    python_simple_repo: Path,
) -> None:
    """A clean tree's working-tree signature is the constant "clean".

    It is therefore identical at every commit, so a checkout landing between
    the refresh and validation is invisible to a signature-only comparison.
    Re-deriving the whole generation identity, which folds in HEAD, is what
    catches it. This test moves HEAD for real rather than mocking, so it
    fails if the identity comparison is removed.
    """
    repo = _indexed(python_simple_repo)
    source = RepoSource(local_path=str(repo))
    config = load_config(source)
    index_config = load_index_config(source)
    store = index_repository(source, config=config, index_config=index_config)
    try:
        assert compute_working_tree_signature(repo, config) == "clean"
        assert _validate_generation(store, repo, config, index_config)[0] is None

        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "moves HEAD, leaves the tree clean"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        assert compute_working_tree_signature(repo, config) == "clean"
        stale_reason, generation_id = _validate_generation(store, repo, config, index_config)
    finally:
        store.close()

    assert stale_reason == "index generation does not describe the current checkout"
    assert generation_id is None


def test_a_cycle_finishing_past_its_deadline_does_not_retire_the_edit(
    python_simple_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late worker must not silently consume an edit the caller gave up on."""
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    repo = _indexed(python_simple_repo)
    target = _first_source_file(repo)
    (repo / target).write_text((repo / target).read_text() + "\n# marker\n")
    _record(repo, target)

    feedback = synchronize_and_report(repo, client="claude-code", deadline=time.monotonic() - 1)

    assert feedback.outcome is PostEditOutcome.TIMED_OUT
    assert feedback.text is None
    assert read_state(repo).status is PostEditStatus.DIRTY
    assert read_state(repo).pending_paths == [target]
    assert "post_edit_abandoned_after_deadline" in log.read_text()


def test_the_timeout_worker_thread_is_a_daemon(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-daemon worker would block interpreter shutdown for any caller
    that returns normally instead of calling os._exit."""
    repo = _indexed(python_simple_repo)
    _record(repo, _first_source_file(repo))
    monkeypatch.setenv("ARCHEX_POST_EDIT_TIMEOUT_SECONDS", "0.05")
    observed: list[bool] = []

    def _slow(*_args: object, **_kwargs: object) -> object:
        observed.append(threading.current_thread().daemon)
        time.sleep(3)
        raise AssertionError("should have been abandoned")

    monkeypatch.setattr("archex.post_edit.impact.index_repository", _slow)

    feedback = synchronize_and_report_with_timeout(repo, client="claude-code")

    assert feedback.outcome is PostEditOutcome.TIMED_OUT
    assert observed == [True]
