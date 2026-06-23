"""Tests for the offline cross-tool tokens-at-fixed-recall comparison."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from archex.benchmark.cross_tool import (
    CrossToolReport,
    CrossToolTaskComparison,
    NaiveBaselineModel,
    NaiveBaselineResult,
    PathTokensAtRecall,
    RetrievalUnit,
    aggregate_cross_tool,
    archex_units,
    compare_task,
    corpus_of,
    naive_units,
    run_cross_tool,
    tokens_at_recall,
)
from archex.benchmark.models import BenchmarkTask, TaskFamily
from archex.benchmark.region_metrics import ReturnedRegion
from archex.benchmark.reporter import format_cross_tool_comparison

_QUESTION = "Where is zorptoken handled?"


def _build_repo(root: Path) -> Path:
    """A small repo: a required file with sparse hits, a noisier false positive."""
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    # 2 keyword hits in the file that must be localized.
    (pkg / "target.py").write_text("def handle(zorptoken):\n    return zorptoken\n")
    # 3 keyword hits: a naive agent triages this false positive first.
    (pkg / "noise.py").write_text("zorptoken\nzorptoken\nzorptoken\n")
    # One hit near the top, then many non-matching lines: window << full file.
    body = "\n".join(["zorptoken = 0", *[f"padding_{i} = {i}" for i in range(40)]])
    (pkg / "big.py").write_text(body + "\n")
    # No keyword hits at all.
    (pkg / "unrelated.py").write_text("answer = 41\n")
    return root


def _git_init(repo: Path) -> None:
    """Initialise a git repo and stage files so discovery takes the git ls-files path."""
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)


def _loc_task(expected: list[str], *, repo: str = "owner/repo") -> BenchmarkTask:
    return BenchmarkTask(
        task_id="loc_zorp",
        repo=repo,
        commit="HEAD",
        question=_QUESTION,
        expected_files=expected,
        family=TaskFamily.LOCALIZATION,
    )


class TestTokensAtRecall:
    def test_walks_until_target_reached(self) -> None:
        units = [
            RetrievalUnit("a.py", 10),
            RetrievalUnit("b.py", 20),
            RetrievalUnit("c.py", 30),
        ]
        at = tokens_at_recall(units, ["b.py", "c.py"], 1.0)
        assert at.target_reached
        assert at.tokens == 60  # pays for the false positive a.py too
        assert at.recall_reached == 1.0
        assert at.units_consumed == 3

    def test_stops_at_partial_target(self) -> None:
        units = [RetrievalUnit("a.py", 10), RetrievalUnit("b.py", 20), RetrievalUnit("c.py", 30)]
        at = tokens_at_recall(units, ["b.py", "c.py"], 0.5)
        # Half the required set is covered by b.py; stop before paying for c.py.
        assert at.target_reached
        assert at.tokens == 30
        assert at.recall_reached == 0.5
        assert at.units_consumed == 2

    def test_unreached_target(self) -> None:
        units = [RetrievalUnit("a.py", 10)]
        at = tokens_at_recall(units, ["b.py"], 1.0)
        assert not at.target_reached
        assert at.tokens == 10
        assert at.recall_reached == 0.0

    def test_empty_required_set_is_trivially_reached(self) -> None:
        at = tokens_at_recall([RetrievalUnit("a.py", 10)], [], 1.0)
        assert at.target_reached
        assert at.tokens == 0
        assert at.units_consumed == 0


class TestCorpusOf:
    def test_self_repo(self) -> None:
        assert corpus_of(_loc_task(["pkg/target.py"], repo=".")) == "self"

    def test_external_localization(self) -> None:
        assert corpus_of(_loc_task(["pkg/target.py"])) == "external-localization"

    def test_external_comprehension(self) -> None:
        task = BenchmarkTask(
            task_id="comp",
            repo="owner/repo",
            commit="HEAD",
            question=_QUESTION,
            expected_files=["pkg/target.py"],
            family=TaskFamily.COMPREHENSION,
        )
        assert corpus_of(task) == "external-comprehension"


class TestNaiveTokenModel:
    def test_full_file_model_is_deterministic(self, tmp_path: Path) -> None:
        repo = _build_repo(tmp_path)
        task = _loc_task(["pkg/target.py"])
        first = naive_units(repo, task, model=NaiveBaselineModel.FULL_FILE, context_window=5)
        second = naive_units(repo, task, model=NaiveBaselineModel.FULL_FILE, context_window=5)
        assert first == second

    def test_orders_files_by_hit_count(self, tmp_path: Path) -> None:
        repo = _build_repo(tmp_path)
        task = _loc_task(["pkg/target.py"])
        units = naive_units(repo, task, model=NaiveBaselineModel.FULL_FILE, context_window=5)
        paths = [unit.path for unit in units]
        # noise.py (3 hits) outranks big.py and target.py (lower hit counts).
        assert paths[0] == "pkg/noise.py"
        # unrelated.py has no keyword hit, so a naive grep never opens it.
        assert "pkg/unrelated.py" not in paths

    def test_grep_window_never_exceeds_full_file(self, tmp_path: Path) -> None:
        repo = _build_repo(tmp_path)
        task = _loc_task(["pkg/target.py"])
        full = {
            unit.path: unit.tokens
            for unit in naive_units(
                repo, task, model=NaiveBaselineModel.FULL_FILE, context_window=5
            )
        }
        window = {
            unit.path: unit.tokens
            for unit in naive_units(
                repo, task, model=NaiveBaselineModel.GREP_WINDOW, context_window=5
            )
        }
        for path, full_tokens in full.items():
            assert window[path] <= full_tokens
        # big.py has sparse hits, so the window is strictly cheaper than the file.
        assert window["pkg/big.py"] < full["pkg/big.py"]

    def test_full_file_model_is_deterministic_over_git_corpus(self, tmp_path: Path) -> None:
        # A real benchmark repo has a .git dir, so discovery uses git ls-files
        # rather than the rglob fallback; the model must be deterministic there too.
        repo = _build_repo(tmp_path)
        _git_init(repo)
        task = _loc_task(["pkg/target.py"])
        first = naive_units(repo, task, model=NaiveBaselineModel.FULL_FILE, context_window=5)
        second = naive_units(repo, task, model=NaiveBaselineModel.FULL_FILE, context_window=5)
        assert first == second
        # The git ls-files path discovers the same source files as the fallback.
        assert [unit.path for unit in first][0] == "pkg/noise.py"
        assert "pkg/unrelated.py" not in [unit.path for unit in first]


class TestCompareTask:
    def test_recall_held_equal_when_both_reach_target(self, tmp_path: Path) -> None:
        repo = _build_repo(tmp_path)
        task = _loc_task(["pkg/target.py"])
        regions = [ReturnedRegion("pkg/target.py", 1, 2, "handle", 8)]
        comparison = compare_task(
            task,
            repo,
            regions,
            target_recall=1.0,
            context_window=5,
            models=[NaiveBaselineModel.FULL_FILE],
        )
        naive = comparison.naive[0].at_recall
        assert comparison.archex.target_reached and naive.target_reached
        # Both measured at the same achieved recall.
        assert comparison.archex.recall_reached == naive.recall_reached == 1.0
        # archex's targeted region is cheaper than reading full grep-hit files.
        assert comparison.archex.tokens < naive.tokens

    def test_naive_miss_is_not_marked_reached(self, tmp_path: Path) -> None:
        repo = _build_repo(tmp_path)
        # unrelated.py has no keyword hit: the naive grep path can never reach it.
        task = _loc_task(["pkg/unrelated.py"])
        regions = [ReturnedRegion("pkg/unrelated.py", 1, 1, None, 5)]
        comparison = compare_task(
            task,
            repo,
            regions,
            target_recall=1.0,
            context_window=5,
            models=[NaiveBaselineModel.FULL_FILE],
        )
        assert comparison.archex.target_reached
        assert not comparison.naive[0].at_recall.target_reached


def _comparison(
    task_id: str,
    corpus: str,
    family: TaskFamily,
    *,
    archex_tokens: int,
    naive_tokens: int,
    naive_reached: bool = True,
) -> CrossToolTaskComparison:
    return CrossToolTaskComparison(
        task_id=task_id,
        repo="." if corpus == "self" else "owner/repo",
        corpus=corpus,
        family=family,
        category=None,
        required_file_count=1,
        target_recall=1.0,
        context_window=5,
        archex=PathTokensAtRecall(
            tokens=archex_tokens, recall_reached=1.0, target_reached=True, units_consumed=1
        ),
        naive=[
            NaiveBaselineResult(
                model=NaiveBaselineModel.FULL_FILE,
                at_recall=PathTokensAtRecall(
                    tokens=naive_tokens,
                    recall_reached=1.0 if naive_reached else 0.0,
                    target_reached=naive_reached,
                    units_consumed=1,
                ),
            )
        ],
    )


class TestAggregate:
    def test_grades_localization_separately(self) -> None:
        comparisons = [
            _comparison(
                "loc",
                "external-localization",
                TaskFamily.LOCALIZATION,
                archex_tokens=100,
                naive_tokens=1000,
            ),
            _comparison(
                "comp",
                "external-comprehension",
                TaskFamily.COMPREHENSION,
                archex_tokens=200,
                naive_tokens=400,
            ),
        ]
        aggregates = aggregate_cross_tool(comparisons, [NaiveBaselineModel.FULL_FILE])
        by_corpus = {agg.corpus: agg for agg in aggregates}
        assert set(by_corpus) == {"external-localization", "external-comprehension"}
        # The localization corpus aggregates only its own task.
        loc = by_corpus["external-localization"]
        assert loc.archex_tokens == 100 and loc.naive_tokens == 1000
        assert loc.token_reduction_pct == pytest.approx(90.0)  # pyright: ignore[reportUnknownMemberType]
        assert loc.mean_token_ratio == pytest.approx(10.0)  # pyright: ignore[reportUnknownMemberType]

    def test_excludes_tasks_at_unequal_recall(self) -> None:
        comparisons = [
            _comparison(
                "reached", "self", TaskFamily.COMPREHENSION, archex_tokens=100, naive_tokens=500
            ),
            _comparison(
                "missed",
                "self",
                TaskFamily.COMPREHENSION,
                archex_tokens=100,
                naive_tokens=999,
                naive_reached=False,
            ),
        ]
        aggregates = aggregate_cross_tool(comparisons, [NaiveBaselineModel.FULL_FILE])
        agg = aggregates[0]
        assert agg.task_count == 2
        # Only the both-reached task contributes to the token delta.
        assert agg.comparable_count == 1
        assert agg.archex_tokens == 100 and agg.naive_tokens == 500


class TestRunCrossTool:
    def test_orchestrates_with_injected_regions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _build_repo(tmp_path)
        task = _loc_task(["pkg/target.py"])

        import archex.benchmark.runner as runner_mod

        def _fixed_repo(
            _task: BenchmarkTask,
            _cache: dict[tuple[str, str, tuple[str, ...]], Path],
            _cleanup: list[Path],
        ) -> Path:
            return repo

        monkeypatch.setattr(runner_mod, "repo_path_for_task", _fixed_repo)
        regions = [ReturnedRegion("pkg/target.py", 1, 2, "handle", 8)]

        report = run_cross_tool(
            [task],
            models=[NaiveBaselineModel.FULL_FILE],
            regions_provider=lambda _task, _path: regions,
        )

        assert report.target_recall == 1.0
        assert len(report.comparisons) == 1
        assert report.comparisons[0].archex.tokens == 8
        agg = report.aggregates[0]
        assert agg.corpus == "external-localization"
        assert agg.comparable_count == 1
        assert agg.naive_tokens > agg.archex_tokens


class TestArchexUnits:
    def test_preserves_region_order(self) -> None:
        regions = [
            ReturnedRegion("a.py", 1, 5, "f", 12),
            ReturnedRegion("b.py", 1, 5, "g", 34),
        ]
        units = archex_units(regions)
        assert [(u.path, u.tokens) for u in units] == [("a.py", 12), ("b.py", 34)]


class TestCheckedInArtifact:
    """Validate the committed per-corpus reference artifact and its rendering."""

    ARTIFACT = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "cross-tool-efficiency"
        / "cross-tool-comparison.json"
    )

    def _load(self) -> CrossToolReport:
        return CrossToolReport.model_validate_json(self.ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_exists_and_parses(self) -> None:
        report = self._load()
        assert report.strategy == "archex_query"
        assert report.comparisons
        assert report.aggregates

    def test_localization_graded_as_its_own_corpus(self) -> None:
        report = self._load()
        corpora = {aggregate.corpus for aggregate in report.aggregates}
        # The localization family is graded separately, never merged with comprehension.
        assert "external-localization" in corpora
        assert "external-comprehension" in corpora
        loc_tasks = {
            comparison.task_id
            for comparison in report.comparisons
            if comparison.corpus == "external-localization"
        }
        comp_tasks = {
            comparison.task_id
            for comparison in report.comparisons
            if comparison.corpus == "external-comprehension"
        }
        assert loc_tasks and comp_tasks
        # Disjoint task sets: no task is counted under both corpora.
        assert loc_tasks.isdisjoint(comp_tasks)
        # Every localization-corpus task actually carries the localization family.
        for comparison in report.comparisons:
            if comparison.corpus == "external-localization":
                assert comparison.family is TaskFamily.LOCALIZATION

    def test_recall_held_equal_in_every_scored_comparison(self) -> None:
        report = self._load()
        for comparison in report.comparisons:
            for naive in comparison.naive:
                # A delta is only meaningful when both paths reached the target; when
                # they do, the artifact measured them at the same achieved recall.
                if comparison.archex.target_reached and naive.at_recall.target_reached:
                    assert comparison.archex.recall_reached == naive.at_recall.recall_reached
        # Aggregates never count more comparable tasks than exist in the corpus.
        for aggregate in report.aggregates:
            assert aggregate.comparable_count <= aggregate.task_count

    def test_artifact_renders(self) -> None:
        md = format_cross_tool_comparison(self._load())
        assert "# Cross-Tool Token Efficiency (offline benchmark)" in md
        assert "Recall is held equal" in md
        assert "External localization" in md
