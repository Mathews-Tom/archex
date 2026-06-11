"""Tests for benchmark YAML task loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.benchmark.loader import (
    load_arch_task,
    load_arch_tasks,
    load_delta_task,
    load_delta_tasks,
    load_task,
    load_tasks,
    validate_task,
)
from archex.benchmark.models import ArchitectureBenchmarkTask, BenchmarkTask, DeltaBenchmarkTask


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
        assert task.include_paths == []

    def test_load_include_paths(self, tmp_path: Path) -> None:
        p = tmp_path / "scoped.yaml"
        p.write_text("""\
task_id: scoped
repo: owner/repo
commit: abc
question: "How?"
include_paths:
  - src/pkg
expected_files:
  - src/pkg/main.py
""")
        task = load_task(p)
        assert task.include_paths == ["src/pkg"]

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

    def test_load_rejects_unknown_field_with_path(self, tmp_path: Path) -> None:
        p = tmp_path / "unknown.yaml"
        p.write_text("""\
task_id: unknown
repo: owner/repo
commit: abc
question: "How?"
expected_files:
  - src/main.py
unexpected: true
""")

        with pytest.raises(ValueError, match=r"unknown\.yaml.*unknown field 'unexpected'"):
            load_task(p)


class TestLoadArchTask:
    def test_load_valid_arch_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "arch.yaml"
        p.write_text("""\
task_id: arch_fixture
repo: "."
commit: HEAD
question: "Which architectural patterns are present?"
include_paths:
  - tests/fixtures/python_patterns
languages: [python]
arch_oracle:
  patterns:
    - name: strategy
      evidence_symbols:
        - SortStrategy
  interfaces:
    - name: SortStrategy
      file_path: tests/fixtures/python_patterns/strategies.py
""")
        task = load_arch_task(p)

        assert isinstance(task, ArchitectureBenchmarkTask)
        assert task.task_id.startswith("arch_")
        assert task.arch_oracle.patterns[0].name == "strategy"

    def test_load_arch_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list")

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_arch_task(p)

    def test_load_arch_rejects_unknown_nested_oracle_field(self, tmp_path: Path) -> None:
        p = tmp_path / "arch_unknown.yaml"
        p.write_text("""\
task_id: arch_unknown
repo: "."
commit: HEAD
question: "Which architectural patterns are present?"
include_paths:
  - tests/fixtures/python_patterns
arch_oracle:
  patterns:
    - name: strategy
      extra_pattern_field: nope
""")

        message = (
            r"arch_unknown\.yaml.*unknown field "
            r"'arch_oracle\.patterns\.0\.extra_pattern_field'"
        )
        with pytest.raises(ValueError, match=message):
            load_arch_task(p)

    def test_load_arch_tasks_directory(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            (tmp_path / f"{name}.yaml").write_text(f"""\
task_id: arch_{name}
repo: "."
commit: HEAD
question: "Which architecture is present?"
include_paths:
  - tests/fixtures/python_patterns
arch_oracle:
  patterns:
    - name: strategy
""")

        tasks = load_arch_tasks(tmp_path)

        assert [task.task_id for task in tasks] == ["arch_a", "arch_b"]

    def test_load_arch_tasks_rejects_duplicate_ids_with_paths(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            (tmp_path / f"{name}.yaml").write_text("""\
task_id: arch_duplicate
repo: "."
commit: HEAD
question: "Which architecture is present?"
include_paths:
  - tests/fixtures/python_patterns
arch_oracle:
  patterns:
    - name: strategy
""")

        message = r"Duplicate task_id 'arch_duplicate'.*a\.yaml.*b\.yaml"
        with pytest.raises(ValueError, match=message):
            load_arch_tasks(tmp_path)

    def test_load_arch_tasks_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_arch_tasks(tmp_path / "missing")

    def test_repository_arch_tasks_load(self) -> None:
        tasks_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "arch_tasks"
        tasks = load_arch_tasks(tasks_dir)

        assert {task.task_id for task in tasks} == {
            "go_middleware_interfaces_architecture",
            "mixed_language_false_positives_architecture",
            "python_false_positives_architecture",
            "python_patterns_architecture",
            "python_strategy_sorting_architecture",
            "rust_traits_async_architecture",
            "typescript_react_hooks_architecture",
        }
        assert all(
            task.arch_oracle.patterns or task.arch_oracle.interfaces or task.arch_oracle.modules
            for task in tasks
        )


class TestLoadTasks:
    def test_load_directory(self, tasks_dir: Path) -> None:
        tasks = load_tasks(tasks_dir)
        assert len(tasks) == 3
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_benchmark_scopes_cover_expected_files(self) -> None:
        tasks_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "tasks"
        tasks = load_tasks(tasks_dir)
        for task in tasks:
            if not task.include_paths:
                continue
            for expected_file in task.expected_files:
                assert any(
                    expected_file == include_path
                    or expected_file.startswith(f"{include_path.rstrip('/')}/")
                    for include_path in task.include_paths
                ), f"{task.task_id}: {expected_file} is outside include_paths"

    def test_external_benchmark_scopes_include_distractors(self) -> None:
        tasks_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "tasks"
        tasks = load_tasks(tasks_dir)
        external_tasks = [task for task in tasks if task.repo != "."]
        assert external_tasks
        for task in external_tasks:
            assert task.include_paths, f"{task.task_id}: external task must declare include_paths"
            assert set(task.include_paths) != set(task.expected_files), (
                f"{task.task_id}: include_paths must be broader than exact expected files"
            )

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

    def test_load_directory_rejects_duplicate_ids_with_paths(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            (tmp_path / f"{name}.yaml").write_text("""\
task_id: duplicate
repo: owner/repo
commit: abc
question: "How?"
expected_files:
  - src/main.py
""")

        message = r"Duplicate task_id 'duplicate'.*a\.yaml.*b\.yaml"
        with pytest.raises(ValueError, match=message):
            load_tasks(tmp_path)


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


class TestLoadDeltaTask:
    def test_load_valid_delta_yaml(self, tmp_path: Path) -> None:
        content = """\
task_id: delta_test
repo: "."
base_commit: abc123
delta_commit: def456
expected_delta:
  - src/main.py
language: python
"""
        p = tmp_path / "delta_test.yaml"
        p.write_text(content)
        task = load_delta_task(p)
        assert isinstance(task, DeltaBenchmarkTask)
        assert task.task_id == "delta_test"
        assert task.base_commit == "abc123"
        assert task.delta_commit == "def456"
        assert task.expected_delta == ["src/main.py"]
        assert task.language == "python"

    def test_load_delta_tasks_directory(self, tmp_path: Path) -> None:
        for i in range(2):
            content = f"""\
task_id: delta_{i}
repo: "."
base_commit: base{i}
delta_commit: delta{i}
"""
            (tmp_path / f"delta_{i}.yaml").write_text(content)
        tasks = load_delta_tasks(tmp_path)
        assert len(tasks) == 2
        assert all(isinstance(t, DeltaBenchmarkTask) for t in tasks)
        assert [t.task_id for t in tasks] == ["delta_0", "delta_1"]

    def test_load_delta_tasks_rejects_duplicate_ids_with_paths(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            (tmp_path / f"{name}.yaml").write_text("""\
task_id: delta_duplicate
repo: "."
base_commit: base
delta_commit: delta
""")

        with pytest.raises(
            ValueError,
            match=r"Duplicate task_id 'delta_duplicate'.*a\.yaml.*b\.yaml",
        ):
            load_delta_tasks(tmp_path)

    def test_load_delta_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_delta_tasks(tmp_path / "nonexistent")

    def test_load_delta_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_delta_task(p)

    def test_load_delta_rejects_unknown_field_with_path(self, tmp_path: Path) -> None:
        p = tmp_path / "delta_unknown.yaml"
        p.write_text("""\
task_id: delta_unknown
repo: "."
base_commit: abc123
delta_commit: def456
expected_delta:
  - src/main.py
extra_delta_field: nope
""")

        with pytest.raises(
            ValueError,
            match=r"delta_unknown\.yaml.*unknown field 'extra_delta_field'",
        ):
            load_delta_task(p)
