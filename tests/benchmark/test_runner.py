"""Tests for benchmark runner logic."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from archex.benchmark.models import BenchmarkRetrievalOptions, BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES, run_benchmark
from archex.index.embeddings import (
    JINA_BERT_CODE_REVISION,
    JINA_V2_MAX_SEQ_LENGTH,
    JINA_V2_MODEL_REVISION,
)

if TYPE_CHECKING:
    from pathlib import Path

JINA_V2_CACHE_IDENTITY = (
    f"jina-v2@{JINA_V2_MODEL_REVISION}"
    f"+code={JINA_BERT_CODE_REVISION}"
    f"+max_seq={JINA_V2_MAX_SEQ_LENGTH}"
)


@pytest.fixture
def fixture_task(python_simple_repo: Path) -> tuple[BenchmarkTask, Path]:
    task = BenchmarkTask(
        task_id="fixture_test",
        repo="test/python_simple",
        commit="HEAD",
        question="How does the main module work?",
        expected_files=["main.py", "utils.py"],
    )
    return task, python_simple_repo


class TestAvailableStrategies:
    def test_default_strategies(self) -> None:
        assert DEFAULT_STRATEGIES == [
            Strategy.RAW_FILES,
            Strategy.RAW_GREPPED,
            Strategy.ARCHEX_QUERY,
        ]
        assert Strategy.ARCHEX_QUERY_FUSION not in DEFAULT_STRATEGIES
        assert Strategy.CROSS_LAYER_FUSION not in DEFAULT_STRATEGIES
        assert Strategy.ARCHEX_SCOUT_FETCH in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY_FUSION in AVAILABLE_STRATEGIES
        assert Strategy.CROSS_LAYER_FUSION in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_SYMBOL_LOOKUP not in AVAILABLE_STRATEGIES


class TestBenchmarkPreflight:
    def test_warms_configured_rerank_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from archex.benchmark.preflight import warm_benchmark_models

        warmed_models: list[str] = []

        class RecordingEmbedder:
            dimension = 768

        class RecordingReranker:
            def __init__(self, model_name: str) -> None:
                self._model_name = model_name

            def warm(self) -> None:
                warmed_models.append(self._model_name)

        def create_embedder(_index_config: object) -> RecordingEmbedder:
            return RecordingEmbedder()

        from archex.index.embeddings import default_embedder_registry

        monkeypatch.setattr(default_embedder_registry, "load_entry_points", lambda: None)
        monkeypatch.setattr(default_embedder_registry, "create", create_embedder)
        monkeypatch.setattr("archex.index.rerank.CrossEncoderReranker", RecordingReranker)
        warmed = warm_benchmark_models(
            [Strategy.ARCHEX_QUERY_FUSION_RERANK],
            BenchmarkRetrievalOptions(rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        )

        assert warmed_models == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
        assert warmed == ["jina-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"]

    def test_warms_embedder_with_explicit_warm_method(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from archex.benchmark.preflight import warm_benchmark_models

        warmed_embedders: list[str] = []

        class RecordingEmbedder:
            dimension = 768

            def warm(self) -> None:
                warmed_embedders.append("coderank")

        def create_embedder(_index_config: object) -> RecordingEmbedder:
            return RecordingEmbedder()

        from archex.index.embeddings import default_embedder_registry

        monkeypatch.setattr(default_embedder_registry, "load_entry_points", lambda: None)
        monkeypatch.setattr(default_embedder_registry, "create", create_embedder)

        warmed = warm_benchmark_models(
            [Strategy.ARCHEX_QUERY_FUSION],
            BenchmarkRetrievalOptions(embedder="coderank"),
        )

        assert warmed_embedders == ["coderank"]
        assert warmed == ["coderank"]


class TestRunBenchmark:
    def test_run_with_fixture_repo(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert report.task_id == "fixture_test"
        assert report.repo == "test/python_simple"
        assert len(report.results) == 2
        assert report.median_latency_ms > 0
        assert report.p95_latency_ms >= report.median_latency_ms

    def test_baseline_tokens_from_raw_files(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert report.baseline_tokens > 0
        raw_result = next(r for r in report.results if r.strategy == Strategy.RAW_FILES)
        assert report.baseline_tokens == raw_result.tokens_total

    def test_savings_backfill(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        raw = next(r for r in report.results if r.strategy == Strategy.RAW_FILES)
        assert raw.savings_vs_raw == 0.0
        # Other strategies should have savings backfilled
        for r in report.results:
            if r.strategy != Strategy.RAW_FILES:
                # savings_vs_raw is computed; could be negative if query returns more
                assert isinstance(r.savings_vs_raw, float)

    def test_strategy_filtering(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        assert len(report.results) == 1
        assert report.results[0].strategy == Strategy.RAW_FILES

    def test_warms_vector_index_when_vector_strategy_present(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        task, repo_path = fixture_task
        from archex.benchmark import runner as runner_mod
        from archex.benchmark.strategies import default_strategy_registry
        from archex.exceptions import ArchexIndexError

        warmed: list[tuple[Path, BenchmarkRetrievalOptions | None, bool]] = []

        def _record(
            _task: BenchmarkTask,
            path: Path,
            options: BenchmarkRetrievalOptions | None = None,
            *,
            warm_rerank: bool = False,
        ) -> None:
            warmed.append((path, options, warm_rerank))

        def _raise(_task: BenchmarkTask, _path: Path) -> None:
            raise ArchexIndexError("no vector backend in test")

        monkeypatch.setattr(runner_mod, "_check_vector_available", lambda: True)
        monkeypatch.setattr(runner_mod, "_warm_repo_index", _record)

        key = Strategy.ARCHEX_QUERY_FUSION.value
        monkeypatch.setitem(default_strategy_registry._runners, key, _raise)  # pyright: ignore[reportPrivateUsage]

        run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_QUERY_FUSION],
            repo_path=repo_path,
            retrieval_options=BenchmarkRetrievalOptions(embedder="coderank"),
        )
        assert warmed == [(repo_path, BenchmarkRetrievalOptions(embedder="coderank"), False)]

    def test_warms_reranker_when_rerank_strategy_present(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from archex.benchmark import runner as runner_mod
        from archex.benchmark.strategies import default_strategy_registry
        from archex.exceptions import ArchexIndexError

        task, repo_path = fixture_task
        warmed: list[tuple[Path, BenchmarkRetrievalOptions | None, bool]] = []

        def _record(
            _task: BenchmarkTask,
            path: Path,
            options: BenchmarkRetrievalOptions | None = None,
            *,
            warm_rerank: bool = False,
        ) -> None:
            warmed.append((path, options, warm_rerank))

        def _raise(_task: BenchmarkTask, _path: Path) -> None:
            raise ArchexIndexError("no vector backend in test")

        monkeypatch.setattr(runner_mod, "_check_vector_available", lambda: True)
        monkeypatch.setattr(runner_mod, "_warm_repo_index", _record)

        key = Strategy.ARCHEX_QUERY_FUSION_RERANK.value
        monkeypatch.setitem(default_strategy_registry._runners, key, _raise)  # pyright: ignore[reportPrivateUsage]

        options = BenchmarkRetrievalOptions(rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        run_benchmark(
            task,
            strategies=[Strategy.ARCHEX_QUERY_FUSION_RERANK],
            repo_path=repo_path,
            retrieval_options=options,
        )
        assert warmed == [(repo_path, options, True)]

    def test_warm_repo_index_uses_configured_embedder(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        from archex.benchmark import runner as runner_mod
        from archex.models import ContextBundle, IndexConfig

        task, repo_path = fixture_task
        captured: list[tuple[str | None, bool, str | None, str, int]] = []

        def fake_query(
            _source: object,
            question: str,
            *,
            token_budget: int,
            explicit_token_budget: bool,
            config: object,
            index_config: IndexConfig,
        ) -> ContextBundle:
            del config
            del explicit_token_budget
            captured.append(
                (
                    index_config.embedder,
                    index_config.rerank,
                    index_config.rerank_model,
                    index_config.chunker,
                    index_config.rerank_candidate_limit,
                )
            )
            return ContextBundle(
                query=question,
                chunks=[],
                token_count=0,
                token_budget=token_budget,
            )

        with patch("archex.api.query", fake_query):
            runner_mod._warm_repo_index(  # pyright: ignore[reportPrivateUsage]
                task,
                repo_path,
                BenchmarkRetrievalOptions(
                    embedder="coderank",
                    vector_chunker="cast",
                ),
            )
            runner_mod._warm_repo_index(  # pyright: ignore[reportPrivateUsage]
                task,
                repo_path,
                BenchmarkRetrievalOptions(
                    embedder="coderank",
                    vector_chunker="cast",
                    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    rerank_candidate_limit=3,
                ),
                warm_rerank=True,
            )

        assert captured == [
            ("coderank", False, None, "cast", 4),
            ("coderank", True, "cross-encoder/ms-marco-MiniLM-L-6-v2", "cast", 3),
        ]

    def test_warm_repo_index_uses_dual_leg_helper_when_enabled(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        from archex.benchmark import runner as runner_mod
        from archex.models import Config, ContextBundle, IndexConfig, RepoSource

        task, repo_path = fixture_task
        captured: list[tuple[str, str, str, str, bool, bool, bool]] = []

        def fake_dual_leg_query(
            *,
            bm25_source: RepoSource,
            vector_source: RepoSource,
            question: str,
            token_budget: int,
            config: Config,
            bm25_index_config: IndexConfig,
            vector_index_config: IndexConfig,
            scoring_weights: object | None = None,
            timing: object | None = None,
            trace: object | None = None,
            file_stage_orchestration: bool = False,
            direct_file_preservation: bool = False,
        ) -> ContextBundle:
            del question, token_budget, config, scoring_weights, timing, trace
            captured.append(
                (
                    bm25_source.stable_identity or "",
                    vector_source.stable_identity or "",
                    bm25_index_config.chunker,
                    vector_index_config.chunker,
                    vector_index_config.rerank,
                    file_stage_orchestration,
                    direct_file_preservation,
                )
            )
            return ContextBundle(query=task.question, token_budget=task.token_budget)

        with patch("archex.api.query_dual_leg_benchmark", fake_dual_leg_query):
            runner_mod._warm_repo_index(  # pyright: ignore[reportPrivateUsage]
                task,
                repo_path,
                BenchmarkRetrievalOptions(
                    bm25_chunker="default",
                    vector_chunker="cast",
                    dual_leg_orchestration=True,
                    file_stage_orchestration=True,
                    direct_file_preservation=True,
                ),
            )

        assert captured == [
            (
                f"test/python_simple@HEAD#embedder={JINA_V2_CACHE_IDENTITY}+chunker=default",
                f"test/python_simple@HEAD#embedder={JINA_V2_CACHE_IDENTITY}+chunker=cast",
                "default",
                "cast",
                False,
                True,
                True,
            )
        ]

    def test_skips_warmup_for_raw_only_strategies(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        task, repo_path = fixture_task
        from archex.benchmark import runner as runner_mod

        warmed: list[Path] = []

        def _record(_task: BenchmarkTask, path: Path) -> None:
            warmed.append(path)

        monkeypatch.setattr(runner_mod, "_check_vector_available", lambda: True)
        monkeypatch.setattr(runner_mod, "_warm_repo_index", _record)

        run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert warmed == []

    def test_symbol_lookup_skipped_gracefully(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_SYMBOL_LOOKUP],
            repo_path=repo_path,
        )
        # symbol_lookup should be skipped, only raw_files in results
        assert len(report.results) == 1
        assert report.results[0].strategy == Strategy.RAW_FILES

    def test_default_strategies_used_when_none(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        """strategies=None should use DEFAULT_STRATEGIES."""
        task, repo_path = fixture_task
        report = run_benchmark(task, strategies=None, repo_path=repo_path)
        strategy_names = {r.strategy for r in report.results}
        # Only baseline/default strategies should run
        assert Strategy.RAW_FILES in strategy_names
        assert Strategy.RAW_GREPPED in strategy_names
        assert Strategy.ARCHEX_QUERY in strategy_names
        assert Strategy.ARCHEX_QUERY_FUSION not in strategy_names
        assert Strategy.CROSS_LAYER_FUSION not in strategy_names

    def test_no_baseline_when_raw_files_omitted(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        """Without raw_files, baseline_tokens=0 and savings stay 0."""
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert report.baseline_tokens == 0
        assert report.results[0].savings_vs_raw == 0.0

    def test_unknown_strategy_runner_skipped(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        """A strategy missing from the registry is logged and skipped."""
        from archex.benchmark.strategies import default_strategy_registry

        task, repo_path = fixture_task
        key = Strategy.RAW_GREPPED.value
        removed = default_strategy_registry._runners.pop(key)  # pyright: ignore[reportPrivateUsage]
        try:
            report = run_benchmark(
                task,
                strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
                repo_path=repo_path,
            )
        finally:
            default_strategy_registry._runners[key] = removed  # pyright: ignore[reportPrivateUsage]

        # RAW_GREPPED was skipped; only RAW_FILES ran
        assert len(report.results) == 1
        assert report.results[0].strategy == Strategy.RAW_FILES

    def test_retrieval_options_are_scoped_to_strategy_runner(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from archex.benchmark.models import BenchmarkResult
        from archex.benchmark.strategies import (
            current_benchmark_retrieval_options,
            default_strategy_registry,
        )

        task, repo_path = fixture_task
        captured: list[BenchmarkRetrievalOptions] = []

        def _record_options(_task: BenchmarkTask, _path: Path) -> BenchmarkResult:
            captured.append(current_benchmark_retrieval_options())
            return BenchmarkResult(
                task_id=task.task_id,
                strategy=Strategy.RAW_GREPPED,
                tokens_total=1,
                tool_calls=1,
                files_accessed=1,
                recall=0.0,
                precision=0.0,
                savings_vs_raw=0.0,
                wall_time_ms=1.0,
                cached=False,
                timestamp="2026-01-01T00:00:00Z",
            )

        key = Strategy.RAW_GREPPED.value
        monkeypatch.setitem(default_strategy_registry._runners, key, _record_options)  # pyright: ignore[reportPrivateUsage]

        run_benchmark(
            task,
            strategies=[Strategy.RAW_GREPPED],
            repo_path=repo_path,
            retrieval_options=BenchmarkRetrievalOptions(splade=True, module_prefilter=True),
        )

        assert captured == [BenchmarkRetrievalOptions(splade=True, module_prefilter=True)]

    def test_benchmark_repo_source_isolates_opt_in_cache_keys(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        from archex.benchmark.strategies import (
            benchmark_repo_source,
            reset_benchmark_retrieval_options,
            set_benchmark_retrieval_options,
        )
        from archex.index.embeddings import (
            JINA_BERT_CODE_REVISION,
            JINA_V2_MAX_SEQ_LENGTH,
            JINA_V2_MODEL_REVISION,
        )

        task, repo_path = fixture_task
        jina_identity = (
            f"jina-v2@{JINA_V2_MODEL_REVISION}"
            f"+code={JINA_BERT_CODE_REVISION}"
            f"+max_seq={JINA_V2_MAX_SEQ_LENGTH}"
        )
        source = benchmark_repo_source(task, repo_path)
        assert source.stable_identity == (
            f"test/python_simple@HEAD#embedder={jina_identity}+chunker=default"
        )

        token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(splade=True))
        try:
            splade_source = benchmark_repo_source(task, repo_path)
        finally:
            reset_benchmark_retrieval_options(token)

        assert splade_source.stable_identity == (
            f"test/python_simple@HEAD#embedder={jina_identity}+chunker=default+splade"
        )

        coderank_token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(embedder="coderank")
        )
        try:
            coderank_source = benchmark_repo_source(task, repo_path)
        finally:
            reset_benchmark_retrieval_options(coderank_token)

        cast_token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(chunker="cast"))
        try:
            cast_source = benchmark_repo_source(task, repo_path)
        finally:
            reset_benchmark_retrieval_options(cast_token)

        scoped_token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions())
        scoped_task = task.model_copy(update={"include_paths": ["src", "tests"]})
        try:
            scoped_source = benchmark_repo_source(scoped_task, repo_path)
        finally:
            reset_benchmark_retrieval_options(scoped_token)

        assert scoped_source.stable_identity == (
            f"test/python_simple@HEAD#scope=src|tests+embedder={jina_identity}+chunker=default"
        )
        assert cast_source.stable_identity == (
            f"test/python_simple@HEAD#embedder={jina_identity}+chunker=cast"
        )

        assert coderank_source.stable_identity == (
            "test/python_simple@HEAD#embedder=coderank+chunker=default"
        )

        split_token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(bm25_chunker="default", vector_chunker="cast")
        )
        try:
            split_bm25_source = benchmark_repo_source(
                task,
                repo_path,
                strategy=Strategy.ARCHEX_QUERY,
            )
            split_vector_source = benchmark_repo_source(
                task,
                repo_path,
                strategy=Strategy.ARCHEX_QUERY_FUSION,
            )
        finally:
            reset_benchmark_retrieval_options(split_token)

        assert split_bm25_source.stable_identity == (
            f"test/python_simple@HEAD#embedder={jina_identity}+chunker=default"
        )
        assert split_vector_source.stable_identity == (
            f"test/python_simple@HEAD#embedder={jina_identity}+chunker=cast"
        )

    def test_benchmark_index_config_applies_module_prefilter_only_with_bm25(self) -> None:
        from archex.benchmark.strategies import (
            benchmark_cache_enabled,
            benchmark_index_config,
            reset_benchmark_retrieval_options,
            set_benchmark_retrieval_options,
        )
        from archex.models import IndexConfig

        assert benchmark_cache_enabled(default=False) is False

        token = set_benchmark_retrieval_options(
            BenchmarkRetrievalOptions(
                splade=True,
                module_prefilter=True,
                rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        )
        try:
            bm25_config = benchmark_index_config(IndexConfig(vector=True))
            vector_config = benchmark_index_config(IndexConfig(bm25=False, vector=True))
            rerank_config = benchmark_index_config(IndexConfig(vector=True, rerank=True))
            cache_enabled = benchmark_cache_enabled(default=False)
        finally:
            reset_benchmark_retrieval_options(token)

        assert bm25_config.splade is True
        assert bm25_config.module_prefilter is True
        assert vector_config.splade is True
        assert vector_config.module_prefilter is False
        assert bm25_config.embedder == "jina-v2"
        assert bm25_config.rerank_model is None
        assert vector_config.embedder == "jina-v2"
        assert vector_config.rerank_model is None
        assert rerank_config.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert bm25_config.chunker == "default"
        assert cache_enabled is True


class TestCloneAtCommit:
    def testclone_at_commit(self, python_simple_repo: Path) -> None:
        """Exercise clone_at_commit with a local file:// URL substitute."""
        import subprocess

        import archex.benchmark.runner as runner_mod

        calls: list[list[str]] = []
        original_run = subprocess.run

        def mock_run(
            cmd: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        runner_mod.subprocess.run = mock_run  # type: ignore[assignment]
        try:
            path, needs_cleanup = runner_mod.clone_at_commit("owner/repo", "abc123")
        finally:
            runner_mod.subprocess.run = original_run  # type: ignore[assignment]

        assert needs_cleanup is True
        assert path.exists()
        # Shallow clone with --branch succeeds (returncode=0), so only 1 call.
        assert len(calls) == 1
        assert "clone" in calls[0]
        assert "https://github.com/owner/repo.git" in calls[0]
        assert "--depth" in calls[0]
        assert "abc123" in calls[0]

        # Cleanup
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def test_cleanup_on_needs_cleanup(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        """When needs_cleanup=True, repo dir is removed after run_benchmark."""
        import archex.benchmark.runner as runner_mod

        # Create a temp dir that should get cleaned up
        clone_dir = tmp_path / "clone_target"
        clone_dir.mkdir()
        (clone_dir / "main.py").write_text("x = 1\n")

        def fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            return clone_dir, True

        original = runner_mod.clone_at_commit
        runner_mod.clone_at_commit = fake_clone  # type: ignore[assignment]
        try:
            task = BenchmarkTask(
                task_id="cleanup_test",
                repo="test/repo",
                commit="abc",
                question="test",
                expected_files=["main.py"],
            )
            # repo_path=None triggers the clone + cleanup path
            run_benchmark(task, strategies=[Strategy.RAW_FILES], repo_path=None)
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        # clone_dir should have been cleaned up
        assert not clone_dir.exists()

    def test_clone_failure_raises_benchmark_clone_error(self) -> None:
        """Both clones failing raises BenchmarkCloneError carrying git's stderr."""
        import subprocess

        import archex.benchmark.runner as runner_mod
        from archex.exceptions import BenchmarkCloneError

        def fail_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 1, "", "fatal: could not read from remote")

        original = runner_mod.subprocess.run
        runner_mod.subprocess.run = fail_run  # type: ignore[assignment]
        try:
            with pytest.raises(BenchmarkCloneError, match="could not read from remote"):
                runner_mod.clone_at_commit("owner/repo", "v1.0")
        finally:
            runner_mod.subprocess.run = original  # type: ignore[assignment]

    def test_clone_timeout_raises_benchmark_clone_error(self) -> None:
        """A git timeout is converted to BenchmarkCloneError, not propagated raw."""
        import subprocess

        import archex.benchmark.runner as runner_mod
        from archex.exceptions import BenchmarkCloneError

        def timeout_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd, 300)

        original = runner_mod.subprocess.run
        runner_mod.subprocess.run = timeout_run  # type: ignore[assignment]
        try:
            with pytest.raises(BenchmarkCloneError, match="timed out"):
                runner_mod.clone_at_commit("owner/repo", "v1.0")
        finally:
            runner_mod.subprocess.run = original  # type: ignore[assignment]


class TestRunAll:
    def _make_tasks_dir(self, tmp_path: Path) -> Path:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "test.yaml").write_text("""\
task_id: test_all
repo: test/repo
commit: HEAD
question: "How does main work?"
expected_files:
  - main.py
""")
        return tasks_dir

    def test_run_all_with_task_dir(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            return python_simple_repo, False

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert len(reports) == 1
        assert reports[0].task_id == "test_all"
        assert (output_dir / "test_all.json").exists()

    def test_run_all_skips_failed_clone_and_continues(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        """A clone failure on one task is isolated; the batch continues."""
        from archex.benchmark.runner import run_all
        from archex.exceptions import BenchmarkCloneError

        tasks_dir = self._make_tasks_dir(tmp_path)  # task_id=test_all, repo=test/repo
        (tasks_dir / "bad.yaml").write_text("""\
task_id: bad_clone
repo: bad/repo
commit: v1
question: "Bad?"
expected_files:
  - main.py
""")
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            if repo_slug == "bad/repo":
                raise BenchmarkCloneError(f"clone failed for {repo_slug}@{commit}: rate limit")
            return python_simple_repo, False

        original = runner_mod.clone_at_commit
        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        # The good task ran and was written; the bad task was skipped, not crashed.
        assert {r.task_id for r in reports} == {"test_all"}
        assert (output_dir / "test_all.json").exists()
        assert not (output_dir / "bad_clone.json").exists()

    def test_run_all_reuses_external_clone_for_same_repo_ref(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "other.yaml").write_text("""\
task_id: other_task
repo: test/repo
commit: HEAD
question: "Other?"
expected_files:
  - main.py
""")
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        clone_calls: list[tuple[str, str]] = []
        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            clone_calls.append((repo_slug, commit))
            return python_simple_repo, False

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert {report.task_id for report in reports} == {"test_all", "other_task"}
        assert clone_calls == [("test/repo", "HEAD")]

    def test_repo_path_for_task_slices_include_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import archex.benchmark.runner as runner_mod

        clone_dir = tmp_path / "clone"
        (clone_dir / "pkg" / "sub").mkdir(parents=True)
        (clone_dir / "pkg" / "sub" / "kept.py").write_text("kept = True\n")
        (clone_dir / "pkg" / "discarded.py").write_text("discarded = True\n")
        (clone_dir / "other.py").write_text("other = True\n")

        def fake_clone(_repo: str, _commit: str) -> tuple[Path, bool]:
            return clone_dir, True

        monkeypatch.setattr(runner_mod, "clone_at_commit", fake_clone)
        task = BenchmarkTask(
            task_id="slice",
            repo="owner/repo",
            commit="abc",
            question="How?",
            expected_files=["pkg/sub/kept.py"],
            include_paths=["pkg/sub"],
        )
        cleanup_paths: list[Path] = []

        sliced = runner_mod._repo_path_for_task(  # pyright: ignore[reportPrivateUsage]
            task,
            {},
            cleanup_paths,
        )

        try:
            assert (sliced / "pkg" / "sub" / "kept.py").exists()
            assert not (sliced / "pkg" / "discarded.py").exists()
            assert not (sliced / "other.py").exists()
            assert (sliced / ".git").is_dir()
            assert cleanup_paths == [clone_dir, sliced]
        finally:
            shutil.rmtree(sliced, ignore_errors=True)

    def test_run_all_cleans_reused_external_clone(
        self,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        output_dir = tmp_path / "results"
        clone_dir = tmp_path / "clone"
        clone_dir.mkdir()
        (clone_dir / "main.py").write_text("x = 1\n")

        import archex.benchmark.runner as runner_mod

        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            del repo_slug, commit
            return clone_dir, True

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert not clone_dir.exists()

    def test_task_filter_nonexistent_raises(self, tmp_path: Path) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        output_dir = tmp_path / "results"

        with pytest.raises(ValueError, match="No task found with id 'nonexistent'"):
            run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                task_filter="nonexistent",
            )

    def test_task_filter_selects_matching(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        # Add a second task
        (tasks_dir / "other.yaml").write_text("""\
task_id: other_task
repo: test/repo
commit: HEAD
question: "Other?"
expected_files:
  - main.py
""")
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            return python_simple_repo, False

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
                task_filter="test_all",
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert len(reports) == 1
        assert reports[0].task_id == "test_all"

    def test_self_only_filters_to_local_repo_tasks(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "self.yaml").write_text("""\
task_id: self_task
repo: "."
commit: HEAD
question: "Self?"
expected_files:
  - main.py
""")
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        clone_calls: list[tuple[str, str]] = []
        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            clone_calls.append((repo_slug, commit))
            return python_simple_repo, False

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
                self_only=True,
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert [report.task_id for report in reports] == ["self_task"]
        assert clone_calls == []
        assert (output_dir / "self_task.json").exists()
        assert not (output_dir / "test_all.json").exists()

    def test_self_only_requires_local_repo_tasks(self, tmp_path: Path) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        output_dir = tmp_path / "results"

        with pytest.raises(ValueError, match="No self-only tasks"):
            run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
                self_only=True,
            )
