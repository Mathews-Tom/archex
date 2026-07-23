"""Tests for the M3 pinned-external-corpus and sealed-holdout policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.benchmark.external_corpus import (
    SEALED_TASKS_DIR,
    SealedCorpusAccessError,
    SealedCorpusPolicyError,
    enforce_sealed_corpus_access,
    find_ci_sealed_references,
    find_unpinned_external_tasks,
    find_vocabulary_leaks,
    is_external_task,
    is_pinned_commit,
    is_sealed_tasks_dir,
    load_sealed_tasks,
    pinned_external_tasks,
    sealed_vocabulary_terms,
)
from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkTask
from archex.cli.benchmark_cmd import benchmark_cmd

_REPO_ROOT = Path(__file__).parents[2]
_PUBLIC_TASKS_DIR = _REPO_ROOT / "benchmarks" / "tasks"
_SRC_ROOT = _REPO_ROOT / "src" / "archex"
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"


def _task(**overrides: object) -> BenchmarkTask:
    defaults: dict[str, object] = {
        "task_id": "t",
        "repo": "owner/repo",
        "commit": "v1.0.0",
        "question": "q",
        "expected_files": ["a.py"],
    }
    defaults.update(overrides)
    return BenchmarkTask.model_validate(defaults)


class TestIsPinnedCommit:
    @pytest.mark.parametrize(
        "commit", ["v2.32.3", "3.1.0", "e186482ca00f8d884ddcbe20417f3654d03315a4"]
    )
    def test_accepts_tags_and_shas(self, commit: str) -> None:
        assert is_pinned_commit(commit)

    @pytest.mark.parametrize(
        "commit", ["", "HEAD", "main", "MAIN", "master", "trunk", "develop", "latest", "  "]
    )
    def test_rejects_floating_refs(self, commit: str) -> None:
        assert not is_pinned_commit(commit)


class TestPublicCorpusPinningCompliance:
    def test_every_public_external_task_is_pinned(self) -> None:
        tasks = load_tasks(_PUBLIC_TASKS_DIR)
        violations = find_unpinned_external_tasks(tasks)
        assert violations == []

    def test_pinned_external_tasks_excludes_self_repo(self) -> None:
        tasks = [
            _task(task_id="external", repo="owner/repo", commit="v1.0.0"),
            _task(task_id="self", repo=".", commit="HEAD"),
        ]
        pinned = pinned_external_tasks(tasks)
        assert [t.task_id for t in pinned] == ["external"]

    def test_is_external_task(self) -> None:
        assert is_external_task(_task(repo="owner/repo"))
        assert not is_external_task(_task(repo="."))

    def test_find_unpinned_external_tasks_flags_floating_ref(self) -> None:
        tasks = [_task(task_id="floating", repo="owner/repo", commit="main")]
        violations = find_unpinned_external_tasks(tasks)
        assert [(v.task_id, v.commit) for v in violations] == [("floating", "main")]


class TestSealedTasksCorpus:
    def test_sealed_tasks_load_and_are_pinned_external(self) -> None:
        tasks = load_sealed_tasks(SEALED_TASKS_DIR)
        assert len(tasks) >= 2
        for task in tasks:
            assert is_external_task(task)
            assert is_pinned_commit(task.commit)

    def test_sealed_tasks_cover_both_families(self) -> None:
        tasks = load_sealed_tasks(SEALED_TASKS_DIR)
        families = {task.family.value for task in tasks}
        assert families == {"comprehension", "localization"}

    def test_sealed_corpus_rejects_self_repo_task(self, tmp_path: Path) -> None:
        sealed_dir = tmp_path / "sealed_tasks"
        sealed_dir.mkdir()
        (sealed_dir / "bad.yaml").write_text(
            "task_id: bad\nrepo: .\ncommit: HEAD\nquestion: q\nexpected_files: [a.py]\n"
        )
        with pytest.raises(SealedCorpusPolicyError, match="working tree"):
            load_sealed_tasks(sealed_dir)

    def test_sealed_corpus_rejects_unpinned_external_task(self, tmp_path: Path) -> None:
        sealed_dir = tmp_path / "sealed_tasks"
        sealed_dir.mkdir()
        (sealed_dir / "bad.yaml").write_text(
            "task_id: bad\nrepo: owner/repo\ncommit: main\nquestion: q\nexpected_files: [a.py]\n"
        )
        with pytest.raises(SealedCorpusPolicyError, match="immutable ref"):
            load_sealed_tasks(sealed_dir)


class TestVocabularyIsolation:
    def test_sealed_vocabulary_terms_is_task_id_only(self) -> None:
        task = _task(task_id="distinctive_task_id", keywords=["proxy"], expected_symbols=["fn"])
        terms = sealed_vocabulary_terms(task)
        assert terms == {"distinctive_task_id"}

    def test_sealed_vocabulary_terms_excludes_short_task_id(self) -> None:
        task = _task(task_id="tid")
        assert sealed_vocabulary_terms(task) == set()

    def test_real_sealed_tasks_are_not_leaked_into_production_code(self) -> None:
        tasks = load_sealed_tasks(SEALED_TASKS_DIR)
        leaks = find_vocabulary_leaks(tasks, _SRC_ROOT)
        assert leaks == []

    def test_find_vocabulary_leaks_detects_a_real_leak(self, tmp_path: Path) -> None:
        task = _task(task_id="distinctive_leak_marker")
        src_root = tmp_path / "src"
        src_root.mkdir()
        (src_root / "module.py").write_text("distinctive_leak_marker = True\n")
        leaks = find_vocabulary_leaks([task], src_root)
        assert len(leaks) == 1
        assert leaks[0].task_id == "distinctive_leak_marker"


class TestCiBoundedness:
    def test_sealed_corpus_is_never_referenced_by_ci_workflows(self) -> None:
        assert find_ci_sealed_references(_WORKFLOWS_DIR) == []

    def test_find_ci_sealed_references_detects_a_reference(self, tmp_path: Path) -> None:
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        (workflows / "bad.yml").write_text("run: archex benchmark run --tasks-dir sealed_tasks\n")
        found = find_ci_sealed_references(workflows)
        assert found == [workflows / "bad.yml"]

    def test_find_ci_sealed_references_empty_dir(self, tmp_path: Path) -> None:
        assert find_ci_sealed_references(tmp_path / "missing") == []


class TestSealedCorpusAccessControl:
    def test_is_sealed_tasks_dir_matches_by_name(self) -> None:
        assert is_sealed_tasks_dir(Path("benchmarks/sealed_tasks"))
        assert is_sealed_tasks_dir(Path("/abs/path/sealed_tasks"))
        assert not is_sealed_tasks_dir(Path("benchmarks/tasks"))

    def test_enforce_blocks_sealed_dir_without_opt_in(self) -> None:
        with pytest.raises(SealedCorpusAccessError, match="--allow-sealed-corpus"):
            enforce_sealed_corpus_access(Path("benchmarks/sealed_tasks"), allow_sealed=False)

    def test_enforce_allows_sealed_dir_with_opt_in(self) -> None:
        enforce_sealed_corpus_access(Path("benchmarks/sealed_tasks"), allow_sealed=True)

    def test_enforce_allows_public_dir_without_opt_in(self) -> None:
        enforce_sealed_corpus_access(Path("benchmarks/tasks"), allow_sealed=False)


class TestCliSealedCorpusAccess:
    def test_run_rejects_sealed_tasks_dir_without_flag(self, tmp_path: Path) -> None:
        sealed_dir = tmp_path / "sealed_tasks"
        sealed_dir.mkdir()
        result = CliRunner().invoke(benchmark_cmd, ["run", "--tasks-dir", str(sealed_dir)])
        assert result.exit_code != 0
        assert "--allow-sealed-corpus" in str(result.output)

    def test_gate_rejects_sealed_tasks_dir_without_flag(self, tmp_path: Path) -> None:
        sealed_dir = tmp_path / "sealed_tasks"
        sealed_dir.mkdir()
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        result = CliRunner().invoke(
            benchmark_cmd,
            ["gate", "--input", str(result_dir), "--tasks-dir", str(sealed_dir)],
        )
        assert result.exit_code != 0
        assert "--allow-sealed-corpus" in str(result.output)
