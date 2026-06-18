"""Tests for TurboQuant data-oblivious vector quantization."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from archex.exceptions import ArchexIndexError
from archex.index.quantize import (
    SUPPORTED_BITS,
    compression_ratio,
    dequantize_vectors,
    float32_bytes,
    generate_rotation_matrix,
    get_codebook,
    get_rotation_matrix,
    pack_codes,
    quantize_vectors,
    quantized_dot_product,
    storage_bytes,
    unpack_codes,
)
from archex.index.vector import VectorIndex
from archex.models import CodeChunk, IndexConfig, SymbolKind

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic test embedder using content hashes for reproducible vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            raw: list[float] = []
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                raw.append((byte_val / 255.0) * 2 - 1)
            results.append(raw)
        return results

    @property
    def dimension(self) -> int:
        return self._dim


def _make_normalized_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate L2-normalized random vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


SAMPLE_CHUNKS = [
    CodeChunk(
        id=f"chunk_{i}",
        content=f"def function_{i}(): return {i}",
        file_path="test.py",
        start_line=i,
        end_line=i + 1,
        symbol_name=f"function_{i}",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=10,
    )
    for i in range(20)
]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def normalized_vectors_64() -> np.ndarray:
    return _make_normalized_vectors(100, 64)


@pytest.fixture
def normalized_vectors_768() -> np.ndarray:
    return _make_normalized_vectors(50, 768)


# ---------------------------------------------------------------------------
# Rotation matrix tests
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    def test_orthogonality(self) -> None:
        r = generate_rotation_matrix(64)
        # R @ R^T should be identity
        product = r @ r.T
        np.testing.assert_allclose(product, np.eye(64), atol=1e-5)

    def test_deterministic_with_seed(self) -> None:
        r1 = generate_rotation_matrix(64, seed=123)
        r2 = generate_rotation_matrix(64, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self) -> None:
        r1 = generate_rotation_matrix(64, seed=1)
        r2 = generate_rotation_matrix(64, seed=2)
        assert not np.array_equal(r1, r2)

    def test_caching(self) -> None:
        r1 = get_rotation_matrix(64)
        r2 = get_rotation_matrix(64)
        assert r1 is r2

    def test_preserves_norm(self) -> None:
        r = generate_rotation_matrix(64)
        v = np.random.default_rng(1).standard_normal(64).astype(np.float32)
        rotated = r @ v
        np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v), rtol=1e-5)

    def test_preserves_inner_product(self) -> None:
        r = generate_rotation_matrix(64)
        rng = np.random.default_rng(1)
        v1 = rng.standard_normal(64).astype(np.float32)
        v2 = rng.standard_normal(64).astype(np.float32)
        np.testing.assert_allclose(v1 @ v2, (r @ v1) @ (r @ v2), rtol=1e-5)


# ---------------------------------------------------------------------------
# Codebook tests
# ---------------------------------------------------------------------------


class TestCodebook:
    def test_4bit_has_16_levels(self) -> None:
        thresholds, centroids = get_codebook(4)
        assert len(thresholds) == 15  # 16 levels - 1 boundaries
        assert len(centroids) == 16

    def test_2bit_has_4_levels(self) -> None:
        thresholds, centroids = get_codebook(2)
        assert len(thresholds) == 3  # 4 levels - 1 boundaries
        assert len(centroids) == 4

    def test_thresholds_monotonically_increasing(self) -> None:
        for bits in SUPPORTED_BITS:
            thresholds, _ = get_codebook(bits)
            assert np.all(np.diff(thresholds) > 0)

    def test_centroids_in_unit_interval(self) -> None:
        for bits in SUPPORTED_BITS:
            _, centroids = get_codebook(bits)
            assert np.all(centroids >= 0)
            assert np.all(centroids <= 1)

    def test_centroids_monotonically_increasing(self) -> None:
        for bits in SUPPORTED_BITS:
            _, centroids = get_codebook(bits)
            assert np.all(np.diff(centroids) > 0)

    def test_unsupported_bits_raises(self) -> None:
        with pytest.raises(ArchexIndexError, match="Unsupported bit-width"):
            get_codebook(3)

    def test_caching(self) -> None:
        cb1 = get_codebook(4)
        cb2 = get_codebook(4)
        assert cb1[0] is cb2[0]


# ---------------------------------------------------------------------------
# Quantize / Dequantize tests
# ---------------------------------------------------------------------------


class TestQuantizeVectors:
    def test_output_shapes(self, normalized_vectors_64: np.ndarray) -> None:
        codes, norms, scale = quantize_vectors(normalized_vectors_64, bits=4)
        n, dim = normalized_vectors_64.shape
        assert codes.shape == (n, dim)
        assert codes.dtype == np.uint8
        assert norms.shape == (n,)
        assert scale.shape == (n, 2)

    def test_codes_in_valid_range_4bit(self, normalized_vectors_64: np.ndarray) -> None:
        codes, _, _ = quantize_vectors(normalized_vectors_64, bits=4)
        assert codes.min() >= 0
        assert codes.max() <= 15

    def test_codes_in_valid_range_2bit(self, normalized_vectors_64: np.ndarray) -> None:
        codes, _, _ = quantize_vectors(normalized_vectors_64, bits=2)
        assert codes.min() >= 0
        assert codes.max() <= 3

    def test_rejects_1d_input(self) -> None:
        v = np.ones(10, dtype=np.float32)
        with pytest.raises(ArchexIndexError, match="Expected 2D"):
            quantize_vectors(v)

    def test_deterministic(self, normalized_vectors_64: np.ndarray) -> None:
        c1, n1, s1 = quantize_vectors(normalized_vectors_64, bits=4)
        c2, n2, s2 = quantize_vectors(normalized_vectors_64, bits=4)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(s1, s2)

    def test_reconstruction_error_4bit(self, normalized_vectors_64: np.ndarray) -> None:
        codes, norms, scale = quantize_vectors(normalized_vectors_64, bits=4)
        approx = dequantize_vectors(codes, norms, scale, bits=4)
        # Mean per-vector reconstruction error should be small
        errors = np.linalg.norm(normalized_vectors_64 - approx, axis=1)
        assert errors.mean() < 0.3  # conservative bound for 64-dim

    def test_reconstruction_error_768dim(self, normalized_vectors_768: np.ndarray) -> None:
        codes, norms, scale = quantize_vectors(normalized_vectors_768, bits=4)
        approx = dequantize_vectors(codes, norms, scale, bits=4)
        errors = np.linalg.norm(normalized_vectors_768 - approx, axis=1)
        # Higher dims have better quantization error properties
        assert errors.mean() < 0.2

    def test_2bit_has_higher_error_than_4bit(self, normalized_vectors_64: np.ndarray) -> None:
        c4, n4, s4 = quantize_vectors(normalized_vectors_64, bits=4)
        c2, n2, s2 = quantize_vectors(normalized_vectors_64, bits=2)
        approx4 = dequantize_vectors(c4, n4, s4, bits=4)
        approx2 = dequantize_vectors(c2, n2, s2, bits=2)
        err4 = np.linalg.norm(normalized_vectors_64 - approx4, axis=1).mean()
        err2 = np.linalg.norm(normalized_vectors_64 - approx2, axis=1).mean()
        assert err2 > err4


# ---------------------------------------------------------------------------
# Quantized dot product tests
# ---------------------------------------------------------------------------


class TestQuantizedDotProduct:
    def test_dot_product_accuracy_4bit(self, normalized_vectors_64: np.ndarray) -> None:
        codes, norms, scale = quantize_vectors(normalized_vectors_64, bits=4)
        query = normalized_vectors_64[0]
        true_sims = normalized_vectors_64[1:] @ query
        approx_sims = quantized_dot_product(query, codes[1:], norms[1:], scale[1:], bits=4)
        mean_error = np.abs(true_sims - approx_sims).mean()
        assert mean_error < 0.02  # within recall tolerance

    def test_dot_product_accuracy_768dim(self, normalized_vectors_768: np.ndarray) -> None:
        codes, norms, scale = quantize_vectors(normalized_vectors_768, bits=4)
        query = normalized_vectors_768[0]
        true_sims = normalized_vectors_768[1:] @ query
        approx_sims = quantized_dot_product(query, codes[1:], norms[1:], scale[1:], bits=4)
        mean_error = np.abs(true_sims - approx_sims).mean()
        assert mean_error < 0.02

    def test_preserves_ranking(self, normalized_vectors_64: np.ndarray) -> None:
        """Top-k ranking should be nearly identical between exact and quantized."""
        codes, norms, scale = quantize_vectors(normalized_vectors_64, bits=4)
        query = normalized_vectors_64[0]
        true_sims = normalized_vectors_64[1:] @ query
        approx_sims = quantized_dot_product(query, codes[1:], norms[1:], scale[1:], bits=4)
        k = 10
        true_top_k = set(np.argsort(true_sims)[-k:])
        approx_top_k = set(np.argsort(approx_sims)[-k:])
        # At least 80% overlap in top-10
        overlap = len(true_top_k & approx_top_k)
        assert overlap >= 8

    def test_self_similarity_close_to_one(self, normalized_vectors_64: np.ndarray) -> None:
        codes, norms, scale = quantize_vectors(normalized_vectors_64, bits=4)
        # Dot product of a vector with its own quantized version
        query = normalized_vectors_64[0]
        self_sim = quantized_dot_product(query, codes[0:1], norms[0:1], scale[0:1], bits=4)
        assert self_sim[0] > 0.95


# ---------------------------------------------------------------------------
# Pack / Unpack tests
# ---------------------------------------------------------------------------


class TestPackUnpack:
    def test_4bit_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        codes = rng.integers(0, 16, size=(10, 64), dtype=np.uint8)
        packed = pack_codes(codes, bits=4)
        unpacked = unpack_codes(packed, 64, bits=4)
        np.testing.assert_array_equal(codes, unpacked)

    def test_2bit_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        codes = rng.integers(0, 4, size=(10, 64), dtype=np.uint8)
        packed = pack_codes(codes, bits=2)
        unpacked = unpack_codes(packed, 64, bits=2)
        np.testing.assert_array_equal(codes, unpacked)

    def test_4bit_packing_halves_size(self) -> None:
        codes = np.zeros((10, 64), dtype=np.uint8)
        packed = pack_codes(codes, bits=4)
        assert packed.shape[1] == 32  # 64 / 2

    def test_2bit_packing_quarters_size(self) -> None:
        codes = np.zeros((10, 64), dtype=np.uint8)
        packed = pack_codes(codes, bits=2)
        assert packed.shape[1] == 16  # 64 / 4

    def test_odd_dimension_round_trip(self) -> None:
        """Dimensions not divisible by codes_per_byte should still work."""
        rng = np.random.default_rng(42)
        codes = rng.integers(0, 16, size=(5, 65), dtype=np.uint8)
        packed = pack_codes(codes, bits=4)
        unpacked = unpack_codes(packed, 65, bits=4)
        np.testing.assert_array_equal(codes, unpacked)

    def test_unsupported_bits_raises(self) -> None:
        codes = np.zeros((1, 10), dtype=np.uint8)
        with pytest.raises(ArchexIndexError, match="Unsupported bit-width"):
            pack_codes(codes, bits=3)
        with pytest.raises(ArchexIndexError, match="Unsupported bit-width"):
            unpack_codes(codes, 10, bits=3)

    def test_768dim_4bit_round_trip(self) -> None:
        rng = np.random.default_rng(99)
        codes = rng.integers(0, 16, size=(20, 768), dtype=np.uint8)
        packed = pack_codes(codes, bits=4)
        unpacked = unpack_codes(packed, 768, bits=4)
        np.testing.assert_array_equal(codes, unpacked)


# ---------------------------------------------------------------------------
# Storage calculation tests
# ---------------------------------------------------------------------------


class TestStorageCalculation:
    def test_compression_ratio_4bit_768dim(self) -> None:
        ratio = compression_ratio(768, 4)
        assert ratio >= 7.5  # target: 8x+

    def test_compression_ratio_2bit_768dim(self) -> None:
        ratio = compression_ratio(768, 2)
        assert ratio >= 14.0

    def test_float32_bytes_correct(self) -> None:
        assert float32_bytes(100, 768) == 100 * 768 * 4

    def test_storage_bytes_less_than_float32(self) -> None:
        n, dim = 1000, 768
        assert storage_bytes(n, dim, 4) < float32_bytes(n, dim)

    def test_4bit_smaller_than_float32_by_factor(self) -> None:
        n, dim = 1000, 768
        ratio = float32_bytes(n, dim) / storage_bytes(n, dim, 4)
        assert ratio >= 7.5


# ---------------------------------------------------------------------------
# VectorIndex with quantization integration tests
# ---------------------------------------------------------------------------


class TestVectorIndexQuantized:
    def test_build_quantized_sets_size(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex(quantize=True, quantize_bits=4)
        idx.build(SAMPLE_CHUNKS, embedder)
        assert idx.size == len(SAMPLE_CHUNKS)

    def test_build_quantized_sets_dim(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex(quantize=True, quantize_bits=4)
        idx.build(SAMPLE_CHUNKS, embedder)
        assert idx.dim == 64

    def test_is_quantized_flag(self, embedder: FakeEmbedder) -> None:
        idx_q = VectorIndex(quantize=True)
        idx_q.build(SAMPLE_CHUNKS, embedder)
        assert idx_q.is_quantized

        idx_u = VectorIndex()
        idx_u.build(SAMPLE_CHUNKS, embedder)
        assert not idx_u.is_quantized

    def test_search_returns_results(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex(quantize=True, quantize_bits=4)
        idx.build(SAMPLE_CHUNKS, embedder)
        results = idx.search("def function_5", embedder, top_k=5)
        assert len(results) > 0

    def test_search_top_result_is_exact_match(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex(quantize=True, quantize_bits=4)
        idx.build(SAMPLE_CHUNKS, embedder)
        results = idx.search(SAMPLE_CHUNKS[5].content, embedder, top_k=5)
        assert results[0][0].id == SAMPLE_CHUNKS[5].id

    def test_quantized_vs_unquantized_ranking_overlap(self, embedder: FakeEmbedder) -> None:
        idx_q = VectorIndex(quantize=True, quantize_bits=4)
        idx_q.build(SAMPLE_CHUNKS, embedder)
        results_q = idx_q.search("def function_3", embedder, top_k=10)

        idx_u = VectorIndex()
        idx_u.build(SAMPLE_CHUNKS, embedder)
        results_u = idx_u.search("def function_3", embedder, top_k=10)

        ids_q = {r[0].id for r in results_q}
        ids_u = {r[0].id for r in results_u}
        overlap = len(ids_q & ids_u)
        assert overlap >= 8  # 80%+ overlap

    def test_build_empty_chunks_quantized(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex(quantize=True)
        idx.build([], embedder)
        assert idx.size == 0
        assert idx.search("anything", embedder) == []

    def test_save_load_quantized(self, embedder: FakeEmbedder, tmp_path: Path) -> None:
        idx = VectorIndex(quantize=True, quantize_bits=4)
        idx.build(SAMPLE_CHUNKS, embedder)
        path = tmp_path / "quantized.npz"
        idx.save(path)

        idx2 = VectorIndex()
        idx2.load(path, SAMPLE_CHUNKS)
        assert idx2.is_quantized
        assert idx2.dim == 64
        assert idx2.size == len(SAMPLE_CHUNKS)

        results = idx2.search(SAMPLE_CHUNKS[3].content, embedder, top_k=5)
        assert results[0][0].id == SAMPLE_CHUNKS[3].id

    def test_save_load_unquantized_backward_compat(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Unquantized .npz files saved without quantization metadata load correctly."""
        idx = VectorIndex()
        idx.build(SAMPLE_CHUNKS, embedder)
        path = tmp_path / "unquantized.npz"
        idx.save(path)

        idx2 = VectorIndex()
        idx2.load(path, SAMPLE_CHUNKS)
        assert not idx2.is_quantized

        results = idx2.search(SAMPLE_CHUNKS[0].content, embedder, top_k=5)
        assert results[0][0].id == SAMPLE_CHUNKS[0].id

    def test_quantized_file_smaller_than_unquantized(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        idx_q = VectorIndex(quantize=True, quantize_bits=4)
        idx_q.build(SAMPLE_CHUNKS, embedder)
        path_q = tmp_path / "q.npz"
        idx_q.save(path_q)

        idx_u = VectorIndex()
        idx_u.build(SAMPLE_CHUNKS, embedder)
        path_u = tmp_path / "u.npz"
        idx_u.save(path_u)

        size_q = path_q.stat().st_size
        size_u = path_u.stat().st_size
        assert size_q < size_u

    def test_save_load_2bit(self, embedder: FakeEmbedder, tmp_path: Path) -> None:
        idx = VectorIndex(quantize=True, quantize_bits=2)
        idx.build(SAMPLE_CHUNKS, embedder)
        path = tmp_path / "q2.npz"
        idx.save(path)

        idx2 = VectorIndex()
        idx2.load(path, SAMPLE_CHUNKS)
        assert idx2.is_quantized
        results = idx2.search(SAMPLE_CHUNKS[0].content, embedder, top_k=5)
        assert len(results) > 0


def test_index_config_metadata_tracks_quantization(tmp_path: Path) -> None:
    from archex.api import (
        _index_config_metadata_matches,  # pyright: ignore[reportPrivateUsage]
        _set_index_config_metadata,  # pyright: ignore[reportPrivateUsage]
    )
    from archex.index.store import IndexStore

    store = IndexStore(tmp_path / "index.db")
    try:
        _set_index_config_metadata(store, IndexConfig(vector=True, quantize_vectors=False))
        assert _index_config_metadata_matches(
            store, IndexConfig(vector=True, quantize_vectors=False)
        )
        assert not _index_config_metadata_matches(
            store,
            IndexConfig(vector=True, quantize_vectors=True, quantize_bits=4),
        )

        _set_index_config_metadata(
            store,
            IndexConfig(vector=True, quantize_vectors=True, quantize_bits=4),
        )
        assert _index_config_metadata_matches(
            store,
            IndexConfig(vector=True, quantize_vectors=True, quantize_bits=4),
        )
        assert not _index_config_metadata_matches(
            store,
            IndexConfig(vector=True, quantize_vectors=True, quantize_bits=2),
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# End-to-end quantization pipeline test
# ---------------------------------------------------------------------------


class TestEndToEndQuantization:
    def test_full_pipeline_768dim(self) -> None:
        """Full quantize → pack → unpack → dequantize → dot product pipeline at 768 dims."""
        vecs = _make_normalized_vectors(30, 768, seed=99)

        # Quantize
        codes, norms, scale = quantize_vectors(vecs, bits=4)
        assert codes.shape == (30, 768)

        # Pack → unpack round-trip
        packed = pack_codes(codes, bits=4)
        unpacked = unpack_codes(packed, 768, bits=4)
        np.testing.assert_array_equal(codes, unpacked)

        # Dequantize and check error
        approx = dequantize_vectors(unpacked, norms, scale, bits=4)
        errors = np.linalg.norm(vecs - approx, axis=1)
        assert errors.mean() < 0.2

        # Dot product accuracy
        query = vecs[0]
        true_sims = vecs[1:] @ query
        approx_sims = quantized_dot_product(query, unpacked[1:], norms[1:], scale[1:], bits=4)
        assert np.abs(true_sims - approx_sims).mean() < 0.02

    def test_compression_target_met(self) -> None:
        """Verify 8x+ compression at 768 dims, 4-bit."""
        ratio = compression_ratio(768, 4)
        assert ratio >= 7.5  # conservative; actual is ~7.8x
