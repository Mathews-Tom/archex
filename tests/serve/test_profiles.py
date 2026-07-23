"""Unit tests for archex.serve.profiles: named retrieval-profile IndexConfig presets."""

from __future__ import annotations

from archex.models import IndexConfig, RetrievalProfile
from archex.serve.profiles import index_config_for_profile


class TestIndexConfigForProfile:
    def test_fast_disables_vector_module_prefilter_and_rerank(self) -> None:
        result = index_config_for_profile(RetrievalProfile.FAST, IndexConfig())
        assert result.vector is False
        assert result.module_prefilter is False
        assert result.rerank is False
        assert result.bm25 is True

    def test_fast_matches_index_config_defaults(self) -> None:
        """FAST is documented as equivalent to IndexConfig()'s own defaults."""
        result = index_config_for_profile(RetrievalProfile.FAST, IndexConfig())
        assert result == IndexConfig()

    def test_balanced_enables_module_prefilter_only(self) -> None:
        result = index_config_for_profile(RetrievalProfile.BALANCED, IndexConfig())
        assert result.vector is False
        assert result.module_prefilter is True
        assert result.rerank is False

    def test_deep_enables_vector_module_prefilter_and_rerank(self) -> None:
        result = index_config_for_profile(RetrievalProfile.DEEP, IndexConfig())
        assert result.vector is True
        assert result.module_prefilter is True
        assert result.rerank is True

    def test_preserves_base_fields_not_covered_by_profile(self) -> None:
        base = IndexConfig(
            embedder="jina-v2",
            chunker="cast",
            chunk_max_tokens=256,
            token_encoding="o200k_base",
            rerank_candidate_limit=8,
        )
        for profile in RetrievalProfile:
            result = index_config_for_profile(profile, base)
            assert result.embedder == "jina-v2"
            assert result.chunker == "cast"
            assert result.chunk_max_tokens == 256
            assert result.token_encoding == "o200k_base"
            assert result.rerank_candidate_limit == 8

    def test_deep_from_fast_base_overrides_correctly(self) -> None:
        """A profile fully replaces the prior profile's toggles rather than merging."""
        fast_config = index_config_for_profile(RetrievalProfile.FAST, IndexConfig())
        deep_from_fast = index_config_for_profile(RetrievalProfile.DEEP, fast_config)
        assert deep_from_fast.vector is True
        assert deep_from_fast.module_prefilter is True
        assert deep_from_fast.rerank is True
