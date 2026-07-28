"""Tests for the R4 corpus validity audit."""

from __future__ import annotations

from typing import Any

import pytest

from archex.benchmark.corpus_audit import (
    CorpusAuditError,
    audit_held_out,
    describe_clusters,
    minimum_detectable_effect,
    score_corpus_leakage,
    score_task_leakage,
    simulate_power,
)
from archex.benchmark.models import BenchmarkTask


def _task(
    task_id: str = "t1",
    *,
    repo: str = "owner/project",
    question: str = "How is caching wired end to end?",
    symbols: list[str] | None = None,
    files: list[str] | None = None,
    keywords: list[str] | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        repo=repo,
        commit="HEAD",
        question=question,
        expected_files=files if files is not None else ["src/thing/widget.py"],
        expected_symbols=symbols if symbols is not None else [],
        keywords=keywords if keywords is not None else [],
    )


class TestLeakage:
    def test_identifier_shaped_symbol_in_the_question_is_a_strong_leak(self) -> None:
        task = _task(question="How does PythonAdapter register itself?", symbols=["PythonAdapter"])
        signals = score_task_leakage(task)
        assert [(s.kind, s.value, s.surface) for s in signals] == [
            ("symbol", "PythonAdapter", "question")
        ]

    def test_a_symbol_in_the_keywords_leaks_too(self) -> None:
        """Keywords are as visible to the retriever as the question is."""
        task = _task(
            question="How is embedding wired?", symbols=["Embedder"], keywords=["Embedder"]
        )
        assert [s.surface for s in score_task_leakage(task)] == ["keywords"]

    def test_a_gold_symbol_that_is_an_ordinary_word_is_not_the_strong_tier(self) -> None:
        """`retry` is a gold symbol and the plain English word for the question."""
        task = _task(question="How does task retry work?", symbols=["retry"])
        assert [s.kind for s in score_task_leakage(task)] == ["symbol_word"]

    def test_the_repository_name_is_never_a_leak(self) -> None:
        """Every self-repo gold path starts with the project name; that is not evidence."""
        task = _task(
            repo=".",
            question="How does archex build its index?",
            files=["src/archex/index/store.py"],
        )
        assert score_task_leakage(task) == ()

    def test_generic_stems_are_not_leaks(self) -> None:
        task = _task(question="Where is the config for the client?", files=["src/a/config.py"])
        assert score_task_leakage(task) == ()

    def test_substrings_do_not_count(self) -> None:
        """Token boundaries matter: `parse` must not match `parsed`."""
        task = _task(question="How is the parsed tree cached?", symbols=["parse_module"])
        assert score_task_leakage(task) == ()

    def test_a_clean_task_reports_nothing(self) -> None:
        task = _task(question="Where does the request lifecycle terminate?", symbols=["Dispatcher"])
        assert score_task_leakage(task) == ()

    def test_corpus_rates_separate_the_tiers(self) -> None:
        tasks = [
            _task("strong", question="How does PythonAdapter work?", symbols=["PythonAdapter"]),
            _task("weak", question="How does retry work?", symbols=["retry"]),
            _task("clean", question="Where is the boundary?", symbols=["Dispatcher"]),
        ]
        report = score_corpus_leakage(tasks)
        assert report.symbol_leaked_task_ids == ("strong",)
        assert report.symbol_leak_rate == pytest.approx(1 / 3)  # pyright: ignore[reportUnknownMemberType]
        assert set(report.leaked_task_ids) == {"strong", "weak"}
        assert report.leak_rate == pytest.approx(2 / 3)  # pyright: ignore[reportUnknownMemberType]

    def test_an_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(CorpusAuditError, match="empty corpus"):
            score_corpus_leakage([])


class TestClustering:
    def test_clusters_are_repositories_not_tasks(self) -> None:
        tasks = [_task(f"t{i}", repo="a/a") for i in range(3)] + [_task("t9", repo="b/b")]
        report = describe_clusters(tasks)
        assert report.cluster_count == 2
        assert report.cluster_sizes == {"a/a": 3, "b/b": 1}
        assert report.largest_cluster == "a/a"
        assert report.largest_cluster_share == pytest.approx(0.75)  # pyright: ignore[reportUnknownMemberType]

    def test_self_repo_share_is_tracked_separately(self) -> None:
        tasks = [_task("a", repo="."), _task("b", repo="."), _task("c", repo="x/y")]
        assert describe_clusters(tasks).self_repo_share == pytest.approx(2 / 3)  # pyright: ignore[reportUnknownMemberType]

    def test_effective_sample_size_shrinks_as_icc_rises(self) -> None:
        """The design effect is the whole reason task count overstates the corpus."""
        tasks = [_task(f"t{i}", repo=f"r{i // 4}") for i in range(64)]
        report = describe_clusters(tasks)
        assert report.effective_sample_size(0.0) == pytest.approx(64.0)  # pyright: ignore[reportUnknownMemberType]
        assert report.effective_sample_size(1.0) == pytest.approx(16.0)  # pyright: ignore[reportUnknownMemberType]
        assert report.effective_sample_size(0.3) < report.effective_sample_size(0.1)

    @pytest.mark.parametrize("icc", [-0.1, 1.1])
    def test_an_out_of_range_icc_is_rejected(self, icc: float) -> None:
        tasks = [_task("a", repo="x"), _task("b", repo="y")]
        with pytest.raises(CorpusAuditError, match=r"icc must lie"):
            describe_clusters(tasks).effective_sample_size(icc)


class TestHeldOut:
    def test_an_overlapping_declaration_is_reported_as_a_full_leak(self) -> None:
        tasks = [_task("a"), _task("b")]
        report = audit_held_out(["a", "b"], tasks, enforced_by_code=False)
        assert report.also_in_task_corpus == ("a", "b")
        assert report.leak_rate == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert report.enforced_by_code is False

    def test_a_genuinely_separate_set_reports_no_leak(self) -> None:
        report = audit_held_out(["z"], [_task("a")], enforced_by_code=True)
        assert report.also_in_task_corpus == ()
        assert report.leak_rate == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]

    def test_an_empty_declaration_is_rejected(self) -> None:
        with pytest.raises(CorpusAuditError, match="empty"):
            audit_held_out([], [_task("a")], enforced_by_code=False)


class TestPower:
    def test_a_large_effect_on_many_clusters_is_detectable(self) -> None:
        result = simulate_power(
            [200] * 8,
            effect_points=30.0,
            base_rate=0.4,
            cluster_sd=0.05,
            simulations=40,
            resamples=100,
            seed=1,
        )
        assert result.power > 0.9

    def test_a_small_effect_on_few_small_clusters_is_not(self) -> None:
        """This is the finding: archex-shaped corpora cannot see small effects."""
        result = simulate_power(
            [4] * 16,
            effect_points=2.0,
            base_rate=0.5,
            cluster_sd=0.08,
            simulations=40,
            resamples=100,
            seed=1,
        )
        assert result.power < 0.3

    def test_power_is_deterministic_under_a_fixed_seed(self) -> None:
        kwargs: dict[str, Any] = {
            "effect_points": 10.0,
            "base_rate": 0.5,
            "cluster_sd": 0.05,
            "simulations": 30,
            "resamples": 80,
            "seed": 7,
        }
        first = simulate_power([10] * 6, **kwargs)
        second = simulate_power([10] * 6, **kwargs)
        assert (first.power, first.mean_ci_width) == (second.power, second.mean_ci_width)

    def test_more_clusters_narrow_the_interval(self) -> None:
        narrow = simulate_power(
            [4] * 64,
            effect_points=5.0,
            base_rate=0.5,
            cluster_sd=0.08,
            simulations=30,
            resamples=100,
            seed=3,
        )
        wide = simulate_power(
            [4] * 8,
            effect_points=5.0,
            base_rate=0.5,
            cluster_sd=0.08,
            simulations=30,
            resamples=100,
            seed=3,
        )
        assert narrow.mean_ci_width < wide.mean_ci_width

    def test_a_single_cluster_cannot_be_simulated(self) -> None:
        with pytest.raises(CorpusAuditError, match="at least two clusters"):
            simulate_power(
                [10],
                effect_points=5.0,
                base_rate=0.5,
                cluster_sd=0.05,
                simulations=5,
                resamples=10,
                seed=1,
            )

    def test_no_reachable_effect_returns_none_rather_than_raising(self) -> None:
        """ "Undetectable at any searched effect" is an answer, not an error."""
        detectable, curve = minimum_detectable_effect(
            [2] * 3,
            base_rate=0.5,
            cluster_sd=0.08,
            target_power=0.99,
            candidates=[0.5, 1.0],
            simulations=20,
            resamples=50,
            seed=1,
        )
        assert detectable is None
        assert len(curve) == 2

    def test_minimum_detectable_effect_picks_the_smallest_that_clears(self) -> None:
        detectable, _ = minimum_detectable_effect(
            [200] * 8,
            base_rate=0.4,
            cluster_sd=0.05,
            target_power=0.8,
            candidates=[1.0, 30.0, 60.0],
            simulations=30,
            resamples=100,
            seed=2,
        )
        assert detectable == 30.0
