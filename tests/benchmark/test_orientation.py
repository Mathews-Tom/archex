"""R24: the orientation profile comparison harness.

Two of this harness's numbers are models rather than measurements, so the tests
pin the model definitions as behaviour: a caller reading `completeness` or
`mean_locator_breadth` is relying on exactly how a handle is matched to an
expected file, and the omission columns only mean something if a silent
truncation in the `full` profile is still counted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.api import index_repository
from archex.benchmark.orientation import (
    OrientationManifest,
    OrientationManifestError,
    OrientationReport,
    format_orientation_markdown,
    load_orientation_manifest,
    load_orientation_tasks,
    run_orientation_benchmark,
)
from archex.cli.main import cli
from archex.config import load_config, load_index_config
from archex.graph_artifact import ArchGraph, build_arch_graph_from_store
from archex.models import RepoSource

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def simple_graph(python_simple_repo: Path) -> ArchGraph:
    source = RepoSource(local_path=str(python_simple_repo))
    store = index_repository(
        source,
        config=load_config(source),
        index_config=load_index_config(source),
    )
    try:
        return build_arch_graph_from_store(store, repo_root=python_simple_repo)
    finally:
        store.close()


def _manifest(**overrides: object) -> OrientationManifest:
    fields: dict[str, object] = {
        "milestone": "R24",
        "repository": "fixture",
        "tasks_dir": "benchmarks/tasks",
        "task_ids": ["archex_adapter_registry"],
        "full_max_files": 40,
        "compact_token_budgets": [400],
    }
    fields.update(overrides)
    return OrientationManifest.model_validate(fields)


class TestFrozenManifest:
    def test_the_checked_in_manifest_resolves_every_declared_task(self) -> None:
        manifest = load_orientation_manifest(_REPO_ROOT / "benchmarks/orientation/manifest.yaml")
        tasks = load_orientation_tasks(manifest, _REPO_ROOT)

        assert [task_id for task_id, _ in tasks] == manifest.task_ids
        assert all(expected for _, expected in tasks), (
            "a task with no expected files proves nothing"
        )

    def test_a_task_id_with_no_file_is_refused(self, tmp_path: Path) -> None:
        manifest = _manifest(task_ids=["does_not_exist"])

        with pytest.raises(OrientationManifestError, match="has no file at"):
            load_orientation_tasks(manifest, _REPO_ROOT)

        del tmp_path

    def test_duplicate_task_ids_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(
            "milestone: R24\nrepository: fixture\ntasks_dir: benchmarks/tasks\n"
            "full_max_files: 40\ncompact_token_budgets: [400]\n"
            "task_ids: [archex_scoring, archex_scoring]\n",
            encoding="utf-8",
        )

        with pytest.raises(OrientationManifestError, match="Duplicate task ids"):
            load_orientation_manifest(path)

    def test_an_unknown_manifest_field_is_refused(self, tmp_path: Path) -> None:
        """Silently ignoring a field would let a protocol change go unrecorded."""
        path = tmp_path / "manifest.yaml"
        path.write_text(
            "milestone: R24\nrepository: fixture\ntasks_dir: benchmarks/tasks\n"
            "full_max_files: 40\ncompact_token_budgets: [400]\n"
            "task_ids: [archex_scoring]\nsample_size: 3\n",
            encoding="utf-8",
        )

        with pytest.raises(OrientationManifestError, match="sample_size"):
            load_orientation_manifest(path)


class TestReachabilityMetrics:
    def test_an_exact_path_scores_one_call_and_breadth_one(self, simple_graph: ArchGraph) -> None:
        report = run_orientation_benchmark(
            simple_graph,
            [("exact", ["main.py"])],
            _manifest(full_max_files=40, compact_token_budgets=[900]),
        )

        full = report.profiles[0]
        assert full.exact_path_hits == 1
        assert full.locatable_files == 0
        assert full.mean_locator_breadth == 1.0
        assert full.modeled_exploration_calls == 1
        assert full.completeness == 1.0

    def test_an_unreachable_path_is_unlocated_and_charged_the_whole_corpus(
        self, simple_graph: ArchGraph
    ) -> None:
        report = run_orientation_benchmark(
            simple_graph,
            [("missing", ["vendor/third_party/absent.py"])],
            _manifest(compact_token_budgets=[900]),
        )

        for profile in report.profiles:
            assert profile.unlocated_files == 1
            assert profile.exact_path_hits == 0
            assert profile.locatable_files == 0
            assert profile.completeness == 0.0
            assert profile.mean_locator_breadth == float(report.indexed_files)
            assert profile.modeled_exploration_calls == 2

    def test_files_behind_one_shared_directory_cost_one_extra_call(
        self, simple_graph: ArchGraph
    ) -> None:
        """The call model charges per distinct locator directory, not per file."""
        report = run_orientation_benchmark(
            simple_graph,
            [("shared", ["main.py", "utils.py"])],
            _manifest(compact_token_budgets=[900]),
        )

        compact = report.profiles[1]
        assert compact.expected_files == 2
        assert compact.exact_path_hits + compact.locatable_files == 2
        assert compact.modeled_exploration_calls <= 1 + 1 + compact.unlocated_files


class TestOmissionHonesty:
    def test_the_full_profile_silent_truncation_is_counted_and_marked_unreported(
        self, simple_graph: ArchGraph
    ) -> None:
        report = run_orientation_benchmark(
            simple_graph,
            [("t", ["main.py"])],
            _manifest(full_max_files=1, compact_token_budgets=[900]),
        )

        full = report.profiles[0]
        assert full.self_reported_items == 0, "the full profile declares nothing"
        assert full.omissions, "the harness must reconstruct what it truncated"
        assert all(not omission.reported_by_tool for omission in full.omissions)
        assert {omission.unit for omission in full.omissions} == {"files"}

    def test_the_compact_profile_reports_every_item_it_drops(self, simple_graph: ArchGraph) -> None:
        report = run_orientation_benchmark(
            simple_graph,
            [("t", ["main.py"])],
            _manifest(compact_token_budgets=[150]),
        )

        compact = report.profiles[1]
        assert compact.omissions
        assert all(omission.reported_by_tool for omission in compact.omissions)
        assert compact.self_reported_items == sum(
            omission.omitted_items for omission in compact.omissions
        )

    def test_files_named_is_measured_identically_for_both_profiles(
        self, simple_graph: ArchGraph
    ) -> None:
        """The only omission figure comparable across profiles must share a denominator."""
        report = run_orientation_benchmark(
            simple_graph,
            [("t", ["main.py"])],
            _manifest(compact_token_budgets=[900]),
        )

        full, compact = report.profiles[0], report.profiles[1]
        assert full.corpus_files == compact.corpus_files == report.indexed_files
        for profile in (full, compact):
            assert profile.files_named + profile.files_not_named == profile.corpus_files
        assert full.files_named > compact.files_named, "enumeration names more files than clusters"

    def test_units_are_never_summed_across_kinds(self, simple_graph: ArchGraph) -> None:
        """Folded directories keep a locator; truncated files do not. Both must stay labelled."""
        report = run_orientation_benchmark(
            simple_graph,
            [("t", ["main.py"])],
            _manifest(compact_token_budgets=[150, 900]),
        )

        units = {omission.unit for profile in report.profiles for omission in profile.omissions}
        assert units
        assert units <= {"files", "rows", "directories", "items"}


class TestReportIsRegenerable:
    def test_the_same_inputs_render_byte_identical_markdown(self, simple_graph: ArchGraph) -> None:
        manifest = _manifest(compact_token_budgets=[400, 900])
        tasks = [("t", ["main.py", "utils.py"])]

        first = format_orientation_markdown(
            run_orientation_benchmark(simple_graph, tasks, manifest)
        )
        second = format_orientation_markdown(
            run_orientation_benchmark(simple_graph, tasks, manifest)
        )

        assert first == second
        assert "| `full` |" in first
        assert "| `compact` |" in first


class TestOrientationCli:
    def test_json_and_markdown_report_the_same_measurement(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        graph_path = tmp_path / "graph.json"
        export = runner.invoke(
            cli, ["graph", "export", str(python_simple_repo), "--output", str(graph_path)]
        )
        assert export.exit_code == 0, export.output
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            "milestone: R24\nrepository: fixture\ntasks_dir: benchmarks/tasks\n"
            "full_max_files: 40\ncompact_token_budgets: [900]\n"
            "task_ids: [archex_scoring]\n",
            encoding="utf-8",
        )

        markdown = runner.invoke(
            cli,
            [
                "benchmark",
                "orientation",
                "--graph",
                str(graph_path),
                "--manifest",
                str(manifest_path),
                "--format",
                "markdown",
            ],
        )
        rendered_json = runner.invoke(
            cli,
            [
                "benchmark",
                "orientation",
                "--graph",
                str(graph_path),
                "--manifest",
                str(manifest_path),
                "--format",
                "json",
            ],
        )

        assert markdown.exit_code == 0, markdown.output
        assert rendered_json.exit_code == 0, rendered_json.output
        assert "Orientation profile comparison (R24)" in markdown.output
        # The JSON must be the same measurement, not a parallel one: re-rendering it
        # has to reproduce the markdown byte-for-byte, which is what makes the
        # checked-in evidence pair auditable.
        parsed = OrientationReport.model_validate_json(rendered_json.output)
        assert parsed.repository == "fixture"
        assert format_orientation_markdown(parsed) == markdown.output
