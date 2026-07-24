"""Unit tests for archex.serve.generation: canonical index generation identity."""

from __future__ import annotations

from pathlib import Path

from archex.index.store import IndexStore
from archex.models import IndexConfig, RetrievalPolicy, VectorMode
from archex.serve.generation import (
    compute_generation_id,
    finalize_generation_id,
    read_generation_id,
)


def _make_store(tmp_path: Path, name: str = "gen.db") -> IndexStore:
    return IndexStore(tmp_path / name)


class TestComputeGenerationId:
    def test_same_inputs_produce_same_id(self) -> None:
        config = IndexConfig()
        first = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=config,
        )
        second = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=config,
        )
        assert first == second

    def test_is_sha256_hex_digest(self) -> None:
        generation_id = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature=None,
            file_count=1,
            chunk_count=1,
            index_config=IndexConfig(),
        )
        assert len(generation_id) == 64
        assert all(c in "0123456789abcdef" for c in generation_id)

    def test_commit_hash_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="commit-a",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="commit-b",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        assert a != b

    def test_working_tree_signature_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="sig-a",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="sig-b",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        assert a != b

    def test_chunk_count_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=11,
            index_config=IndexConfig(),
        )
        assert a != b

    def test_file_count_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=4,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        assert a != b

    def test_schema_version_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        b = compute_generation_id(
            schema_version="6",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(),
        )
        assert a != b

    def test_retrieval_affecting_config_field_changes_id(self) -> None:
        bm25_only = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(retrieval_policy=RetrievalPolicy.BM25_ONLY),
        )
        vector_only = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(retrieval_policy=RetrievalPolicy.VECTOR_ONLY),
        )
        assert bm25_only != vector_only

    def test_chunker_setting_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(chunk_max_tokens=256),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(chunk_max_tokens=512),
        )
        assert a != b

    def test_semantic_evidence_providers_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(semantic_evidence_providers=[]),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(semantic_evidence_providers=["scip"]),
        )
        assert a != b

    def test_runtime_evidence_providers_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(runtime_evidence_providers=[]),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(runtime_evidence_providers=["coverage"]),
        )
        assert a != b

    def test_history_evidence_providers_change_changes_id(self) -> None:
        a = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(history_evidence_providers=[]),
        )
        b = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(history_evidence_providers=["git_log"]),
        )
        assert a != b

    def test_vector_mode_changes_id(self) -> None:
        raw = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(vector_mode=VectorMode.RAW),
        )
        surrogate = compute_generation_id(
            schema_version="5",
            commit_hash="abc123",
            working_tree_signature="clean",
            file_count=3,
            chunk_count=10,
            index_config=IndexConfig(vector_mode=VectorMode.SURROGATE),
        )
        assert raw != surrogate

    def test_none_commit_and_signature_do_not_crash(self) -> None:
        generation_id = compute_generation_id(
            schema_version="5",
            commit_hash=None,
            working_tree_signature=None,
            file_count=0,
            chunk_count=0,
            index_config=IndexConfig(),
        )
        assert isinstance(generation_id, str) and generation_id


class TestFinalizeAndReadGenerationId:
    def test_read_generation_id_before_finalize_is_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            assert read_generation_id(store) is None
        finally:
            store.close()

    def test_finalize_persists_a_readable_generation_id(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            store.set_metadata("schema_version", "5")
            store.set_metadata("commit_hash", "abc123")
            generation_id = finalize_generation_id(store, IndexConfig())
            assert read_generation_id(store) == generation_id
        finally:
            store.close()

    def test_finalize_reflects_live_chunk_count_not_a_stale_metadata_field(
        self, tmp_path: Path
    ) -> None:
        """A caller that forgets to update a chunk_count metadata field elsewhere
        still gets a correct identity, because finalize reads counts live from
        the chunks table rather than trusting a separately maintained field."""
        from archex.models import CodeChunk

        store = _make_store(tmp_path)
        try:
            store.set_metadata("schema_version", "5")
            store.set_metadata("commit_hash", "abc123")
            # Deliberately wrong/stale metadata field — must not affect the result.
            store.set_metadata("chunk_count", "999")
            before = finalize_generation_id(store, IndexConfig())

            store.insert_chunks(
                [
                    CodeChunk(
                        id="c1",
                        file_path="a.py",
                        content="def f(): pass",
                        start_line=1,
                        end_line=1,
                        language="python",
                        token_count=3,
                    )
                ]
            )
            after = finalize_generation_id(store, IndexConfig())
            assert before != after
        finally:
            store.close()

    def test_finalize_is_stable_when_nothing_changes(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            store.set_metadata("schema_version", "5")
            store.set_metadata("commit_hash", "abc123")
            first = finalize_generation_id(store, IndexConfig())
            second = finalize_generation_id(store, IndexConfig())
            assert first == second
        finally:
            store.close()

    def test_finalize_changes_when_index_config_changes(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            store.set_metadata("schema_version", "5")
            store.set_metadata("commit_hash", "abc123")
            default_id = finalize_generation_id(store, IndexConfig())
            reranked_id = finalize_generation_id(store, IndexConfig(rerank=True))
            assert default_id != reranked_id
        finally:
            store.close()
