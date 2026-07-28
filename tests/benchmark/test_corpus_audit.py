"""Tests for the R4 corpus validity audit."""

from __future__ import annotations

import math
from typing import Any

import pytest

from archex.benchmark.corpus_audit import (
    CorpusAuditError,
    audit_held_out,
    describe_clusters,
    detection_bracket,
    estimate_effect_heterogeneity,
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

    def test_the_design_effect_uses_the_size_weighted_mean_not_the_arithmetic_one(self) -> None:
        """One large cluster dominates the design effect in proportion to its size."""
        tasks = [_task(f"big{i}", repo="big") for i in range(24)] + [
            _task(f"small{i}", repo=f"r{i}") for i in range(4)
        ]
        report = describe_clusters(tasks)
        # 28 tasks: one cluster of 24 and four of 1. Arithmetic mean is 5.6;
        # size-weighted is (576 + 4) / 28 = 20.714.
        assert report.weighted_mean_cluster_size == pytest.approx(580 / 28)  # pyright: ignore[reportUnknownMemberType]
        arithmetic_would_give = 28 / (1 + (5.6 - 1) * 0.3)
        assert report.effective_sample_size(0.3) < arithmetic_would_give / 2

    def test_effective_sample_size_shrinks_as_icc_rises(self) -> None:
        """The design effect is the whole reason task count overstates the corpus."""
        tasks = [_task(f"t{i}", repo=f"r{i // 4}") for i in range(64)]
        report = describe_clusters(tasks)
        assert report.effective_sample_size(0.0) == pytest.approx(64.0)  # pyright: ignore[reportUnknownMemberType]
        # Equal clusters make the weighted and arithmetic means coincide, so at
        # ICC 1 each of the 16 clusters contributes one usable observation.
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
        first = simulate_power([10] * 8, **kwargs)
        second = simulate_power([10] * 8, **kwargs)
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
        with pytest.raises(CorpusAuditError, match="at least 8 are required"):
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
            [2] * 16,
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


class TestDetectorRegressions:
    """Each test here kills a mutant that survived the first review."""

    def test_a_snake_case_symbol_quoted_verbatim_is_a_strong_leak(self) -> None:
        """The matcher once normalised the surface but not the needle, so no
        snake_case or dotted gold symbol could ever match -- silently clearing
        the exact identifier class the strong tier exists to catch."""
        task = _task(
            question="How does default_adapter_registry resolve adapters?",
            symbols=["default_adapter_registry"],
        )
        assert [(s.kind, s.value) for s in score_task_leakage(task)] == [
            ("symbol", "default_adapter_registry")
        ]

    def test_a_dotted_symbol_quoted_verbatim_is_a_strong_leak(self) -> None:
        task = _task(question="Where does Client.send dispatch?", symbols=["Client.send"])
        assert [s.kind for s in score_task_leakage(task)] == ["symbol"]

    def test_an_underscored_symbol_does_not_match_the_bare_word(self) -> None:
        """The fix for the above must not readmit `_merge` matching "merge"."""
        assert (
            score_task_leakage(_task(question="How does session merge?", symbols=["_merge"])) == ()
        )

    def test_an_underscored_symbol_does_not_match_the_words_split_apart(self) -> None:
        task = _task(question="How does the runtime block on a future?", symbols=["block_on"])
        assert score_task_leakage(task) == ()

    def test_a_generic_symbol_lands_in_the_weak_tier_rather_than_vanishing(self) -> None:
        """`Config` was dropped outright, so it appeared in no tier at all."""
        task = _task(question="How is config resolved?", symbols=["Config"], keywords=["config"])
        assert {s.kind for s in score_task_leakage(task)} == {"symbol_word"}

    def test_a_symbol_equal_to_the_repository_name_is_never_a_leak(self) -> None:
        task = _task(repo=".", question="How does archex index?", symbols=["archex"])
        assert score_task_leakage(task) == ()

    def test_a_saturating_effect_is_refused_rather_than_silently_clamped(self) -> None:
        """Clamping made every effect above the ceiling produce identical data,
        so the power curve plateaued below 1.0 and an MDE was read off it."""
        with pytest.raises(CorpusAuditError, match="saturates the treatment arm"):
            simulate_power(
                [10] * 8,
                effect_points=30.0,
                base_rate=0.85,
                cluster_sd=0.05,
                simulations=5,
                resamples=20,
                seed=1,
            )

    def test_power_is_invariant_to_cluster_order(self) -> None:
        """Cluster sizes arrived in alphabetical-repo-name order, which made the
        published figure depend on how repositories happened to be named."""
        sizes = [24, 4, 3, 2, 2, 2, 2, 2]
        kwargs: dict[str, Any] = {
            "effect_points": 20.0,
            "base_rate": 0.5,
            "cluster_sd": 0.08,
            "simulations": 60,
            "resamples": 120,
            "seed": 4,
        }
        assert (
            simulate_power(sizes, **kwargs).power
            == simulate_power(list(reversed(sizes)), **kwargs).power
        )

    def test_resamples_and_simulations_are_independent_knobs(self) -> None:
        """A single interleaved RNG stream made a +/-1 change to resamples
        reshuffle every later simulation, so neighbouring values disagreed."""
        base: dict[str, Any] = {
            "effect_points": 20.0,
            "base_rate": 0.5,
            "cluster_sd": 0.08,
            "simulations": 200,
            "seed": 4,
        }
        powers = [simulate_power([4] * 16, resamples=n, **base).power for n in (399, 400, 401)]
        # The interval is itself a Monte Carlo estimate, so neighbouring resample
        # counts may differ slightly. What must not happen -- and did, when one
        # interleaved stream fed every simulation -- is a wholesale reshuffle.
        assert max(powers) - min(powers) < 0.05

    def test_a_two_sided_interval_detects_a_negative_effect(self) -> None:
        """A one-sided detection rule would silently miss regressions."""
        result = simulate_power(
            [200] * 8,
            effect_points=-30.0,
            base_rate=0.5,
            cluster_sd=0.05,
            simulations=40,
            resamples=120,
            seed=2,
        )
        assert result.power > 0.9

    def test_the_interval_width_matches_the_analytic_two_proportion_width(self) -> None:
        """Pins the interval at 95%: a 90% interval would be about 16% narrower."""
        n = 1600
        result = simulate_power(
            [200] * 8,
            effect_points=0.0,
            base_rate=0.5,
            cluster_sd=0.0,
            simulations=60,
            resamples=400,
            seed=5,
        )
        analytic = 2 * 1.96 * 100 * math.sqrt(2 * 0.25 / n)
        assert result.mean_ci_width == pytest.approx(analytic, rel=0.15)  # pyright: ignore[reportUnknownMemberType]

    def test_power_reports_its_own_monte_carlo_error(self) -> None:
        result = simulate_power(
            [4] * 16,
            effect_points=25.0,
            base_rate=0.5,
            cluster_sd=0.08,
            simulations=100,
            resamples=100,
            seed=6,
        )
        assert 0.0 < result.monte_carlo_se < 0.06
        assert result.clears(0.0) is True
        assert result.clears(0.99) is False

    def test_a_bracket_reports_a_range_when_the_grid_straddles_the_target(self) -> None:
        _, curve = minimum_detectable_effect(
            [4] * 16,
            base_rate=0.5,
            cluster_sd=0.08,
            target_power=0.8,
            candidates=[10.0, 25.0, 30.0],
            simulations=200,
            resamples=200,
            seed=20260728,
        )
        described = detection_bracket(curve, target_power=0.8).describe()
        assert "points" in described

    def test_heterogeneity_estimate_removes_within_cluster_noise(self) -> None:
        """Raw per-cluster spread is mostly sampling noise, not heterogeneity."""
        deltas = [2.5, 2.0, -3.5, 12.0, 3.5, -1.0, 4.5, 2.5]
        assert (
            estimate_effect_heterogeneity(deltas, tasks_per_cluster=200, base_rate=0.41625) == 0.0
        )
        # The same spread over far more tasks per cluster cannot be noise.
        assert estimate_effect_heterogeneity(deltas, tasks_per_cluster=100000, base_rate=0.5) > 4.0

    def test_effect_heterogeneity_makes_clusters_matter(self) -> None:
        """Without it, repository count cannot be the binding constraint."""
        kwargs: dict[str, Any] = {
            "effect_points": 10.0,
            "base_rate": 0.5,
            "cluster_sd": 0.05,
            "simulations": 60,
            "resamples": 150,
            "seed": 8,
        }
        homogeneous = simulate_power([50] * 8, effect_sd=0.0, **kwargs)
        heterogeneous = simulate_power([50] * 8, effect_sd=8.0, **kwargs)
        assert heterogeneous.mean_ci_width > homogeneous.mean_ci_width

    def test_too_few_clusters_is_refused(self) -> None:
        """Below eight clusters the percentile bootstrap narrows artificially and
        reports higher power for a worse design."""
        with pytest.raises(CorpusAuditError, match="at least 8 are required"):
            simulate_power(
                [32] * 2,
                effect_points=5.0,
                base_rate=0.5,
                cluster_sd=0.05,
                simulations=10,
                resamples=50,
                seed=1,
            )

    def test_by_family_counts_tasks_not_signals(self) -> None:
        task = _task(
            question="How do PythonAdapter and AdapterRegistry interact?",
            symbols=["PythonAdapter", "AdapterRegistry"],
        )
        report = score_corpus_leakage([task])
        assert sum(report.by_family.values()) == 1
        assert report.by_kind["symbol"] == 2
