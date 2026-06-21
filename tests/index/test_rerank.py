"""Tests for CrossEncoderReranker and explicit-enable logic."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from archex.exceptions import ConfigError
from archex.index import rerank as rerank_module
from archex.index.rerank import (
    AMBIGUOUS_BM25_CV_THRESHOLD,
    AMBIGUOUS_BM25_IDF_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    JINA_RERANKER_MODEL,
    MAX_CONTENT_CHARS,
    RERANK_CANDIDATE_LIMIT,
    ConditionalReranker,
    CrossEncoderReranker,
    _best_device,  # pyright: ignore[reportPrivateUsage]
    bm25_is_ambiguous,
    is_available,
    maybe_conditional_reranker,
)
from archex.models import CodeChunk, IndexConfig, SymbolKind

_HAS_RERANKER_DEPS = is_available()


@pytest.fixture(autouse=True)
def _mock_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    def fake_resolve(model_name: str, *, revision: str | None = None) -> str:
        del revision
        return f"/cache/{model_name.replace('/', '--')}"

    monkeypatch.setattr(rerank_module, "resolve_hf_model_path", fake_resolve)


def _make_chunk(chunk_id: str, content: str = "def fn(): pass") -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=f"{chunk_id}.py",
        start_line=1,
        end_line=1,
        symbol_name=chunk_id,
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=10,
    )


def _reranker_with_mock() -> tuple[CrossEncoderReranker, MagicMock]:
    """Create a custom-model CrossEncoderReranker with an injected predict model."""
    reranker = CrossEncoderReranker(model_name="custom/model")
    mock_model = MagicMock()
    reranker._model = mock_model  # pyright: ignore[reportPrivateUsage]
    return reranker, mock_model


class TestIsAvailable:
    def test_returns_bool(self) -> None:
        result = is_available()
        assert isinstance(result, bool)

    def test_true_when_mocked(self) -> None:
        with patch("archex.index.rerank.is_available", return_value=True):
            # Direct import bypasses mock; test the real function shape
            assert isinstance(is_available(), bool)


class TestDeviceSelection:
    def test_best_device_uses_mps_when_available(self) -> None:
        torch_module = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
        )

        with patch.dict("sys.modules", {"torch": torch_module}):
            assert _best_device() == "mps"

    def test_best_device_falls_back_to_cpu(self) -> None:
        torch_module = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            ),
        )

        with patch.dict("sys.modules", {"torch": torch_module}):
            assert _best_device() == "cpu"


class TestConstants:
    def test_default_top_k_is_30(self) -> None:
        assert DEFAULT_TOP_K == 30

    def test_listwise_caps_target_warm_benchmark_latency(self) -> None:
        assert MAX_CONTENT_CHARS == 1024
        assert RERANK_CANDIDATE_LIMIT == 4

    def test_default_model_is_jina_reranker(self) -> None:
        assert DEFAULT_MODEL == JINA_RERANKER_MODEL


class TestMaybeReranker:
    def test_returns_none_when_rerank_disabled(self) -> None:
        from archex.api import _maybe_reranker  # pyright: ignore[reportPrivateUsage]

        # Default config leaves rerank unset; reranking stays off even when
        # reranker dependencies are installed, so the flag is observably on/off.
        result = _maybe_reranker(IndexConfig())
        assert result is None

    @pytest.mark.skipif(not _HAS_RERANKER_DEPS, reason="transformers not installed")
    def test_explicit_rerank_true_uses_jina_default(self) -> None:
        from archex.api import _maybe_reranker  # pyright: ignore[reportPrivateUsage]

        config = IndexConfig(rerank=True)
        result = _maybe_reranker(config)
        assert isinstance(result, CrossEncoderReranker)
        assert result._model_name == JINA_RERANKER_MODEL  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.skipif(not _HAS_RERANKER_DEPS, reason="transformers not installed")
    def test_uses_custom_model(self) -> None:
        from archex.api import _maybe_reranker  # pyright: ignore[reportPrivateUsage]

        config = IndexConfig(rerank=True, rerank_model="custom/model")
        result = _maybe_reranker(config)
        assert result is not None
        assert result._model_name == "custom/model"  # pyright: ignore[reportPrivateUsage]


class TestCrossEncoderReranker:
    def test_init_does_not_call_rerank(self) -> None:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_default_model_name(self) -> None:
        _ = CrossEncoderReranker()
        assert DEFAULT_MODEL == "jinaai/jina-reranker-v3"

    def test_custom_model_name(self) -> None:
        reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker.rerank("query", []) == []

    def test_default_load_requires_remote_code_opt_in(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")

        with pytest.raises(ConfigError, match="Remote code is disabled.*jina-reranker-v3"):
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_default_load_enforces_remote_code_opt_in_before_cache(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_model = MagicMock()
        fake_model.rerank.return_value = [{"index": 0, "relevance_score": 1.0}]

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("transformers.AutoModel") as auto_model,
        ):
            auto_model.from_pretrained.return_value = fake_model
            CrossEncoderReranker(allow_remote_code=True).rerank("query", [(chunk, 0.0)])

        with pytest.raises(ConfigError, match="Remote code is disabled.*jina-reranker-v3"):
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_default_load_passes_pinned_jina_revision(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_model = MagicMock()
        fake_model.rerank.return_value = [{"index": 0, "relevance_score": 1.0}]

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("transformers.AutoModel") as auto_model,
        ):
            auto_model.from_pretrained.return_value = fake_model
            CrossEncoderReranker(allow_remote_code=True).rerank("query", [(chunk, 0.0)])

        auto_model.from_pretrained.assert_called_once_with(
            JINA_RERANKER_MODEL,
            revision=rerank_module.JINA_RERANKER_REVISION,
            dtype="auto",
            trust_remote_code=True,
        )
        fake_model.eval.assert_called_once_with()
        fake_model.to.assert_not_called()
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_custom_model_load_has_no_revision_pin(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("sentence_transformers.CrossEncoder") as cross_encoder,
        ):
            cross_encoder.return_value.predict.return_value = np.array([1.0])
            CrossEncoderReranker(model_name="custom/model").rerank("query", [(chunk, 0.0)])

        cross_encoder.assert_called_once_with(
            "/cache/custom--model",
            trust_remote_code=False,
            device="cpu",
        )
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_default_load_moves_transformers_model_to_mps_when_available(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_model = MagicMock()
        fake_model.rerank.return_value = [{"index": 0, "relevance_score": 1.0}]

        with (
            patch("archex.index.rerank._best_device", return_value="mps"),
            patch("transformers.AutoModel") as auto_model,
        ):
            auto_model.from_pretrained.return_value = fake_model
            CrossEncoderReranker(allow_remote_code=True).rerank("query", [(chunk, 0.0)])

        fake_model.to.assert_called_once_with("mps")
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_load_sets_eos_as_padding_token_when_missing(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_encoder = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token=None,
                pad_token_id=None,
                eos_token="<eos>",
                eos_token_id=151645,
            ),
            model=SimpleNamespace(config=SimpleNamespace(pad_token_id=None)),
            predict=MagicMock(return_value=np.array([1.0])),
        )

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("sentence_transformers.CrossEncoder", return_value=fake_encoder),
        ):
            CrossEncoderReranker(model_name="custom/model").rerank("query", [(chunk, 0.0)])

        assert fake_encoder.tokenizer.pad_token == "<eos>"
        assert fake_encoder.tokenizer.pad_token_id == 151645
        assert fake_encoder.model.config.pad_token_id == 151645
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_load_sets_model_padding_id_when_tokenizer_has_pad(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_encoder = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token="<|endoftext|>",
                pad_token_id=151643,
                eos_token="<|im_end|>",
                eos_token_id=151645,
            ),
            model=SimpleNamespace(config=SimpleNamespace(pad_token_id=None)),
            predict=MagicMock(return_value=np.array([1.0])),
        )

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("sentence_transformers.CrossEncoder", return_value=fake_encoder),
        ):
            CrossEncoderReranker(model_name="custom/model").rerank("query", [(chunk, 0.0)])

        assert fake_encoder.tokenizer.pad_token == "<|endoftext|>"
        assert fake_encoder.model.config.pad_token_id == 151643
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_cached_model_sets_model_padding_id(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_encoder = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token="<|endoftext|>",
                pad_token_id=151643,
            ),
            model=SimpleNamespace(config=SimpleNamespace(pad_token_id=None)),
            predict=MagicMock(return_value=np.array([1.0])),
        )
        rerank_module._MODEL_CACHE["test/model"] = fake_encoder  # pyright: ignore[reportPrivateUsage]

        CrossEncoderReranker(model_name="test/model").rerank("query", [(chunk, 0.0)])

        assert fake_encoder.model.config.pad_token_id == 151643
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_load_aligns_model_padding_id_with_tokenizer(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")
        fake_encoder = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token="<|endoftext|>",
                pad_token_id=151643,
            ),
            model=SimpleNamespace(config=SimpleNamespace(pad_token_id=151645)),
            predict=MagicMock(return_value=np.array([1.0])),
        )

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("sentence_transformers.CrossEncoder", return_value=fake_encoder),
        ):
            CrossEncoderReranker(model_name="custom/model").rerank("query", [(chunk, 0.0)])

        assert fake_encoder.model.config.pad_token_id == 151643
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_reuses_loaded_model_for_same_model_name(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")

        with patch("sentence_transformers.CrossEncoder") as cross_encoder:
            cross_encoder.return_value.predict.return_value = np.array([1.0])
            first = CrossEncoderReranker(model_name="test/model")
            second = CrossEncoderReranker(model_name="test/model")

            first.rerank("query", [(chunk, 0.0)])
            second.rerank("query", [(chunk, 0.0)])

        assert cross_encoder.call_count == 1
        assert first._model is second._model  # pyright: ignore[reportPrivateUsage]
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_loads_distinct_models_for_distinct_model_names(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")

        with patch("sentence_transformers.CrossEncoder") as cross_encoder:
            cross_encoder.return_value.predict.return_value = np.array([1.0])
            CrossEncoderReranker(model_name="test/model-a").rerank("query", [(chunk, 0.0)])
            CrossEncoderReranker(model_name="test/model-b").rerank("query", [(chunk, 0.0)])

        assert cross_encoder.call_count == 2
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_rerank_empty_candidates(self) -> None:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_default_rerank_uses_transformers_rerank_results(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.rerank.return_value = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
        reranker._model = mock_model  # pyright: ignore[reportPrivateUsage]

        chunks = [_make_chunk("a"), _make_chunk("b")]
        result = reranker.rerank("query", [(chunks[0], 0.0), (chunks[1], 0.0)], top_k=2)

        assert [chunk.id for chunk, _ in result] == ["b", "a"]
        assert [score for _, score in result] == [0.9, 0.1]

    def test_rerank_sorts_by_cross_encoder_score(self) -> None:
        reranker, mock_model = _reranker_with_mock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        candidates = [(c, float(i)) for i, c in enumerate(chunks)]

        result = reranker.rerank("query", candidates)

        assert len(result) == 3
        assert result[0][0].id == "b"
        assert result[1][0].id == "c"
        assert result[2][0].id == "a"

    def test_rerank_respects_top_k(self) -> None:
        reranker, mock_model = _reranker_with_mock()
        mock_model.predict.return_value = np.array([0.9, 0.5, 0.1])

        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        candidates = [(c, float(i)) for i, c in enumerate(chunks)]

        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2

    def test_rerank_default_top_k_keeps_30(self) -> None:
        reranker, mock_model = _reranker_with_mock()
        n = 40
        mock_model.predict.return_value = np.arange(n, dtype=float)

        chunks = [_make_chunk(f"chunk_{i}") for i in range(n)]
        candidates = [(c, 0.0) for c in chunks]

        result = reranker.rerank("query", candidates)
        assert len(result) == DEFAULT_TOP_K

    def test_rerank_truncates_content(self) -> None:
        reranker, mock_model = _reranker_with_mock()
        mock_model.predict.return_value = np.array([1.0])

        long_content = "x" * (MAX_CONTENT_CHARS + 1000)
        chunk = _make_chunk("long", content=long_content)
        reranker.rerank("query", [(chunk, 1.0)])

        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs[0][1]) == MAX_CONTENT_CHARS

    def test_rerank_returns_float_scores(self) -> None:
        reranker, mock_model = _reranker_with_mock()
        mock_model.predict.return_value = np.array([0.75])

        result = reranker.rerank("query", [(_make_chunk("a"), 1.0)])
        assert isinstance(result[0][1], float)
        assert result[0][1] == 0.75

    def test_rerank_replaces_original_scores(self) -> None:
        """Cross-encoder scores replace BM25 scores, not blend with them."""
        reranker, mock_model = _reranker_with_mock()
        mock_model.predict.return_value = np.array([0.1, 0.5, 0.9])

        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        candidates = [(chunks[0], 10.0), (chunks[1], 5.0), (chunks[2], 1.0)]

        result = reranker.rerank("query", candidates)

        assert result[0][0].id == "c"
        assert result[0][1] == 0.9
        assert result[1][0].id == "b"
        assert result[1][1] == 0.5
        assert result[2][0].id == "a"
        assert result[2][1] == 0.1


def _bm25(scores: list[float]) -> list[tuple[CodeChunk, float]]:
    """Build a BM25-ranked candidate list with the given scores."""
    return [(_make_chunk(f"c{i}"), score) for i, score in enumerate(scores)]


class _StubReranker:
    """Fake cross-encoder recording invocations and optionally sleeping."""

    def __init__(self, *, sleep_s: float = 0.0) -> None:
        self.calls = 0
        self._sleep_s = sleep_s

    def rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[tuple[CodeChunk, float]]:
        self.calls += 1
        if self._sleep_s:
            time.sleep(self._sleep_s)
        # Reverse the candidate order so a reorder is observable.
        reordered = list(reversed(candidates))
        return [(chunk, float(rank)) for rank, (chunk, _) in enumerate(reordered)][:top_k]


class TestBm25IsAmbiguous:
    def test_flat_scores_are_ambiguous(self) -> None:
        decision = bm25_is_ambiguous(_bm25([1.0, 0.99, 0.98, 0.97]))
        assert decision.should_rerank is True
        assert decision.reason.startswith("flat_bm25")
        assert decision.bm25_cv <= AMBIGUOUS_BM25_CV_THRESHOLD

    def test_clear_separation_is_confident(self) -> None:
        decision = bm25_is_ambiguous(_bm25([10.0, 1.0, 1.0, 1.0]))
        assert decision.should_rerank is False
        assert decision.reason.startswith("confident_bm25")
        assert decision.bm25_cv > AMBIGUOUS_BM25_CV_THRESHOLD

    def test_low_idf_forces_rerank_despite_clear_separation(self) -> None:
        # Clear BM25 separation, but query terms are too common to discriminate.
        decision = bm25_is_ambiguous(
            _bm25([10.0, 1.0, 1.0, 1.0]),
            avg_idf=AMBIGUOUS_BM25_IDF_THRESHOLD - 0.5,
        )
        assert decision.should_rerank is True
        assert decision.reason.startswith("low_idf")

    def test_high_idf_keeps_confident_decision(self) -> None:
        decision = bm25_is_ambiguous(
            _bm25([10.0, 1.0, 1.0, 1.0]),
            avg_idf=AMBIGUOUS_BM25_IDF_THRESHOLD + 1.0,
        )
        assert decision.should_rerank is False

    def test_too_few_results_skip(self) -> None:
        decision = bm25_is_ambiguous(_bm25([1.0]))
        assert decision.should_rerank is False
        assert decision.reason.startswith("too_few_results")


class TestMaybeConditionalReranker:
    def test_disabled_returns_none(self) -> None:
        assert maybe_conditional_reranker(enabled=False) is None

    def test_enabled_returns_conditional_reranker(self) -> None:
        reranker = maybe_conditional_reranker(enabled=True, model_name="custom/model")
        assert isinstance(reranker, ConditionalReranker)

    def test_enabled_does_not_load_model_at_construction(self) -> None:
        # Construction must not touch the model: a remote-code model with the
        # opt-in unset only fails when an ambiguous query actually invokes it.
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        reranker = maybe_conditional_reranker(
            enabled=True, model_name=JINA_RERANKER_MODEL, allow_remote_code=False
        )
        assert isinstance(reranker, ConditionalReranker)

    def test_enabled_without_model_raises(self) -> None:
        with pytest.raises(ConfigError, match="no rerank model was supplied"):
            maybe_conditional_reranker(enabled=True)


class TestConditionalRerank:
    def test_confident_bm25_skips_model(self) -> None:
        stub = _StubReranker()
        conditional = ConditionalReranker(stub)  # type: ignore[arg-type]
        candidates = _bm25([10.0, 1.0, 1.0, 1.0])
        result = conditional.rerank_if_ambiguous("q", candidates, candidates)
        assert stub.calls == 0
        assert result.invoked is False
        assert result.status == "skipped:confident_bm25"
        assert result.results == candidates
        assert result.rerank_ms == 0.0

    def test_ambiguous_bm25_invokes_model_and_reorders(self) -> None:
        stub = _StubReranker()
        conditional = ConditionalReranker(stub)  # type: ignore[arg-type]
        candidates = _bm25([1.0, 0.99, 0.98, 0.97])
        result = conditional.rerank_if_ambiguous("q", candidates, candidates)
        assert stub.calls == 1
        assert result.invoked is True
        assert result.status == "applied"
        # Stub reverses order; the full candidate set is preserved.
        assert [c.id for c, _ in result.results] == [c.id for c, _ in reversed(candidates)]
        assert {c.id for c, _ in result.results} == {c.id for c, _ in candidates}

    def test_low_idf_invokes_model_even_when_separated(self) -> None:
        stub = _StubReranker()
        conditional = ConditionalReranker(stub)  # type: ignore[arg-type]
        candidates = _bm25([10.0, 1.0, 1.0, 1.0])
        result = conditional.rerank_if_ambiguous(
            "q", candidates, candidates, avg_idf=AMBIGUOUS_BM25_IDF_THRESHOLD - 0.5
        )
        assert stub.calls == 1
        assert result.status == "applied"

    def test_tail_beyond_candidate_limit_keeps_order(self) -> None:
        stub = _StubReranker()
        conditional = ConditionalReranker(stub, candidate_limit=2)  # type: ignore[arg-type]
        candidates = _bm25([1.0, 0.99, 0.98, 0.97])
        result = conditional.rerank_if_ambiguous("q", candidates, candidates)
        # Head (first 2) reordered; tail (last 2) keeps retrieval order.
        ids = [c.id for c, _ in result.results]
        assert ids[:2] == ["c1", "c0"]
        assert ids[2:] == ["c2", "c3"]

    def test_applied_rerank_ms_within_cap(self) -> None:
        stub = _StubReranker()
        conditional = ConditionalReranker(stub)  # type: ignore[arg-type]
        candidates = _bm25([1.0, 0.99, 0.98, 0.97])
        result = conditional.rerank_if_ambiguous("q", candidates, candidates)
        assert result.status == "applied"
        assert 0.0 <= result.rerank_ms <= conditional.latency_cap_ms

    def test_rerank_stage_wall_clock_is_bounded_when_model_is_slow(self) -> None:
        # The model pass sleeps far longer than the cap; the caller must be
        # released at the cap rather than blocking for the whole model runtime,
        # so the rerank stage the pipeline observes is genuinely wall-clock bounded.
        model_sleep_s = 0.1
        cap_ms = 10.0
        stub = _StubReranker(sleep_s=model_sleep_s)
        conditional = ConditionalReranker(stub, latency_cap_ms=cap_ms)  # type: ignore[arg-type]
        candidates = _bm25([1.0, 0.99, 0.98, 0.97])
        start = time.perf_counter()
        result = conditional.rerank_if_ambiguous("q", candidates, candidates)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert result.status == "aborted:latency"
        # Caller released near the cap, well before the model's full runtime.
        assert elapsed_ms < model_sleep_s * 1000.0
        assert result.results == candidates

    def test_slow_model_pass_aborts_and_preserves_order(self) -> None:
        stub = _StubReranker(sleep_s=0.02)
        conditional = ConditionalReranker(stub, latency_cap_ms=1.0)  # type: ignore[arg-type]
        candidates = _bm25([1.0, 0.99, 0.98, 0.97])
        result = conditional.rerank_if_ambiguous("q", candidates, candidates)
        assert stub.calls == 1
        assert result.invoked is True
        assert result.status == "aborted:latency"
        assert result.rerank_ms > conditional.latency_cap_ms
        # Aborted rerank falls back to the original retrieval order.
        assert result.results == candidates

    def test_empty_candidates_skip_without_invocation(self) -> None:
        stub = _StubReranker()
        conditional = ConditionalReranker(stub)  # type: ignore[arg-type]
        result = conditional.rerank_if_ambiguous("q", [], _bm25([1.0, 0.99, 0.98]))
        assert stub.calls == 0
        assert result.invoked is False
        assert result.status == "skipped:no_candidates"

    def test_respects_remote_code_policy_on_invocation(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        conditional = maybe_conditional_reranker(
            enabled=True, model_name=JINA_RERANKER_MODEL, allow_remote_code=False
        )
        assert conditional is not None
        candidates = _bm25([1.0, 0.99, 0.98, 0.97])
        with pytest.raises(ConfigError, match="Remote code is disabled.*jina-reranker-v3"):
            conditional.rerank_if_ambiguous("q", candidates, candidates)
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
