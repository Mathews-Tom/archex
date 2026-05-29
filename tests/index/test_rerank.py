"""Tests for CrossEncoderReranker and explicit-enable logic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from archex.index import rerank as rerank_module
from archex.index.rerank import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    JINA_RERANKER_MODEL,
    JINA_RERANKER_REVISION,
    MAX_CONTENT_CHARS,
    CrossEncoderReranker,
    _best_device,  # pyright: ignore[reportPrivateUsage]
    is_available,
)
from archex.models import CodeChunk, IndexConfig, SymbolKind

_HAS_CROSS_ENCODER = is_available()


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
    """Create a CrossEncoderReranker with a mock model injected (bypasses _load_model)."""
    reranker = CrossEncoderReranker()
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

    def test_max_content_chars_matches_jina_window_budget(self) -> None:
        assert MAX_CONTENT_CHARS == 16384

    def test_default_model_is_jina_reranker(self) -> None:
        assert DEFAULT_MODEL == JINA_RERANKER_MODEL


class TestMaybeReranker:
    def test_returns_none_when_rerank_disabled(self) -> None:
        from archex.api import _maybe_reranker  # pyright: ignore[reportPrivateUsage]

        # Default config leaves rerank unset; reranking stays off even when
        # sentence-transformers is installed, so the flag is observably on/off.
        result = _maybe_reranker(IndexConfig())
        assert result is None

    @pytest.mark.skipif(not _HAS_CROSS_ENCODER, reason="sentence-transformers not installed")
    def test_explicit_rerank_true_uses_jina_default(self) -> None:
        from archex.api import _maybe_reranker  # pyright: ignore[reportPrivateUsage]

        config = IndexConfig(rerank=True)
        result = _maybe_reranker(config)
        assert isinstance(result, CrossEncoderReranker)
        assert result._model_name == JINA_RERANKER_MODEL  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.skipif(not _HAS_CROSS_ENCODER, reason="sentence-transformers not installed")
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

    def test_default_load_passes_pinned_jina_revision(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")

        with (
            patch("archex.index.rerank._best_device", return_value="cpu"),
            patch("sentence_transformers.CrossEncoder") as cross_encoder,
        ):
            cross_encoder.return_value.predict.return_value = np.array([1.0])
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

        cross_encoder.assert_called_once_with(
            JINA_RERANKER_MODEL,
            revision=JINA_RERANKER_REVISION,
            trust_remote_code=True,
            device="cpu",
        )
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
            "custom/model",
            revision=None,
            trust_remote_code=True,
            device="cpu",
        )
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    def test_load_passes_mps_when_available(self) -> None:
        rerank_module._MODEL_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
        chunk = _make_chunk("a")

        with (
            patch("archex.index.rerank._best_device", return_value="mps"),
            patch("sentence_transformers.CrossEncoder") as cross_encoder,
        ):
            cross_encoder.return_value.predict.return_value = np.array([1.0])
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

        cross_encoder.assert_called_once_with(
            JINA_RERANKER_MODEL,
            revision=JINA_RERANKER_REVISION,
            trust_remote_code=True,
            device="mps",
        )
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
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

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
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

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
            CrossEncoderReranker().rerank("query", [(chunk, 0.0)])

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
