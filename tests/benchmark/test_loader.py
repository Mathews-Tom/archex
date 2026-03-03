"""Tests for benchmark YAML task loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.benchmark.loader import load_task, load_tasks, validate_task
from archex.benchmark.models import BenchmarkTask

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    content = """\
task_id: test_task
repo: owner/repo
commit: abc123
question: "How does X work?"
expected_files:
  - src/main.py
  - src/utils.py
keywords:
  - main
  - utils
"""
    p = tmp_path / "test_task.yaml"
    p.write_text(content)
    return p


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    for i in range(3):
        content = f"""\
task_id: task_{i}
repo: owner/repo
commit: abc{i}
question: "Question {i}?"
expected_files:
  - file_{i}.py
"""
        (tmp_path / f"task_{i}.yaml").write_text(content)
    return tmp_path


class TestLoadTask:
    def test_load_valid_yaml(self, sample_yaml: Path) -> None:
        task = load_task(sample_yaml)
        assert task.task_id == "test_task"
        assert task.repo == "owner/repo"
        assert task.commit == "abc123"
        assert len(task.expected_files) == 2
        assert task.keywords == ["main", "utils"]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_task(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_task(p)

    def test_load_missing_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "incomplete.yaml"
        p.write_text("task_id: test\nrepo: owner/repo\n")
        with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
            load_task(p)


class TestLoadTasks:
    def test_load_directory(self, tasks_dir: Path) -> None:
        tasks = load_tasks(tasks_dir)
        assert len(tasks) == 3
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        tasks = load_tasks(tmp_path)
        assert tasks == []

    def test_load_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_tasks(tmp_path / "nonexistent")

    def test_sorted_by_filename(self, tasks_dir: Path) -> None:
        tasks = load_tasks(tasks_dir)
        ids = [t.task_id for t in tasks]
        assert ids == ["task_0", "task_1", "task_2"]


class TestValidateTask:
    def test_valid_task(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does main work?",
            expected_files=["main.py", "utils.py"],
        )
        errors = validate_task(task, python_simple_repo)
        assert errors == []

    def test_missing_expected_file(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How?",
            expected_files=["main.py", "nonexistent.py"],
        )
        errors = validate_task(task, python_simple_repo)
        assert any("nonexistent.py" in e for e in errors)

    def test_empty_expected_files(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How?",
            expected_files=[],
        )
        errors = validate_task(task, python_simple_repo)
        assert any("No expected_files" in e for e in errors)

    def test_nonexistent_repo_path(self, tmp_path: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How?",
            expected_files=["main.py"],
        )
        errors = validate_task(task, tmp_path / "nonexistent")
        assert any("does not exist" in e for e in errors)

    def test_empty_question(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="   ",
            expected_files=["main.py"],
        )
        errors = validate_task(task, python_simple_repo)
        assert any("Empty question" in e for e in errors)
