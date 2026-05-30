"""Tests for SPLADEIndex: learned sparse retrieval over CodeChunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from archex.exceptions import ArchexIndexError
from archex.index.graph import DependencyGraph
from archex.index.splade import SPLADEEncoder, SPLADEIndex
from archex.index.store import IndexStore
from archex.models import CodeChunk, SymbolKind
from archex.serve.context import assemble_context

SAMPLE_CHUNKS = [
    CodeChunk(
        id="utils.py:calculate_sum:5",
        content="def calculate_sum(a: int, b: int) -> int:\n    return a + b",
        file_path="utils.py",
        start_line=5,
        end_line=6,
        symbol_name="calculate_sum",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=20,
    ),
    CodeChunk(
        id="auth.py:authenticate:10",
        content=(
            "def authenticate(username: str, password: str) -> bool:\n"
            "    return check_credentials(username, password)"
        ),
        file_path="auth.py",
        start_line=10,
        end_line=11,
        symbol_name="authenticate",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=25,
    ),
    CodeChunk(
        id="session.py:SessionManager:1",
        content=(
            "class SessionManager:\n"
            "    def __init__(self, db: Database) -> None:\n"
            "        self.db = db\n"
            "        self.session = scoped_session(sessionmaker(bind=db.engine))\n"
            "\n"
            "    def commit(self) -> None:\n"
            "        self.session.commit()"
        ),
        file_path="session.py",
        start_line=1,
        end_line=7,
        symbol_name="SessionManager",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=50,
    ),
]


class FakeSPLADEEncoder:
    """Deterministic encoder for unit tests — no model download needed.

    Simulates SPLADE by assigning weights based on character hashing.
    Each unique word in the text gets a non-zero entry in the sparse vector.
    """

    def __init__(self) -> None:
        self._vocab_size = 30522  # BERT vocab size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        results: list[dict[int, float]] = []
        for text in texts:
            sparse: dict[int, float] = {}
            words = text.lower().split()
            for word in set(words):
                # Deterministic hash to term_id
                term_id = hash(word) % self._vocab_size
                # Weight based on word frequency
                weight = 1.0 + words.count(word) * 0.5
                sparse[abs(term_id)] = weight
            results.append(sparse)
        return results

    def decode_token_ids(self, token_ids: list[int]) -> list[str]:
        return [f"token_{tid}" for tid in token_ids]


@pytest.fixture
def fake_encoder() -> FakeSPLADEEncoder:
    return FakeSPLADEEncoder()


@pytest.fixture
def store_and_index(
    tmp_path: Path, fake_encoder: FakeSPLADEEncoder
) -> Generator[tuple[IndexStore, SPLADEIndex], None, None]:
    db = tmp_path / "splade_test.db"
    store = IndexStore(db)
    idx = SPLADEIndex(store, encoder=fake_encoder)
    store.insert_chunks(SAMPLE_CHUNKS)
    idx.build(SAMPLE_CHUNKS)
    yield store, idx
    store.close()


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------


def test_build_populates_index(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    assert idx.has_data
    assert idx.size == 3


def test_build_empty_chunks(tmp_path: Path, fake_encoder: FakeSPLADEEncoder) -> None:
    db = tmp_path / "empty.db"
    store = IndexStore(db)
    idx = SPLADEIndex(store, encoder=fake_encoder)
    idx.build([])
    assert not idx.has_data
    assert idx.size == 0
    store.close()


def test_build_replaces_previous_data(
    store_and_index: tuple[IndexStore, SPLADEIndex],
) -> None:
    _, idx = store_and_index
    assert idx.size == 3
    # Rebuild with a subset
    idx.build(SAMPLE_CHUNKS[:1])
    assert idx.size == 1


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


def test_search_returns_results(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    results = idx.search("calculate sum")
    assert len(results) > 0
    chunk_ids = [c.id for c, _ in results]
    assert "utils.py:calculate_sum:5" in chunk_ids


def test_search_ranks_relevant_chunk_first(
    store_and_index: tuple[IndexStore, SPLADEIndex],
) -> None:
    _, idx = store_and_index
    results = idx.search("authenticate username password")
    assert len(results) > 0
    assert results[0][0].id == "auth.py:authenticate:10"


def test_search_session_management(
    store_and_index: tuple[IndexStore, SPLADEIndex],
) -> None:
    """The core SPLADE value prop: NL query matches code identifiers."""
    _, idx = store_and_index
    results = idx.search("session commit database")
    assert len(results) > 0
    chunk_ids = [c.id for c, _ in results]
    assert "session.py:SessionManager:1" in chunk_ids


def test_search_empty_query(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    assert idx.search("") == []
    assert idx.search("   ") == []


def test_search_respects_top_k(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    results = idx.search("def", top_k=1)
    assert len(results) <= 1


def test_search_scores_are_positive(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    results = idx.search("authenticate")
    for _, score in results:
        assert score > 0


def test_search_scores_descending(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    results = idx.search("session manager")
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_return_contract_matches_bm25(
    store_and_index: tuple[IndexStore, SPLADEIndex],
) -> None:
    _, idx = store_and_index
    results = idx.search("authenticate", top_k=2)

    assert len(results) <= 2
    for chunk, score in results:
        assert isinstance(chunk, CodeChunk)
        assert isinstance(score, float)
        assert score > 0


def test_build_then_query_vocabulary_mismatch_candidate(
    tmp_path: Path,
    fake_encoder: FakeSPLADEEncoder,
) -> None:
    db = tmp_path / "vocab_mismatch.db"
    store = IndexStore(db)
    chunk = CodeChunk(
        id="tasks.py:celery_task_dispatch:1",
        content=(
            "def celery_task_dispatch(payload: dict[str, str]) -> None:\n    worker.send(payload)"
        ),
        file_path="tasks.py",
        start_line=1,
        end_line=2,
        symbol_name="celery_task_dispatch",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=24,
    )
    store.insert_chunks([chunk])
    idx = SPLADEIndex(store, encoder=fake_encoder)

    idx.build([chunk])
    results = idx.search("dispatch worker", top_k=5)

    assert [result_chunk.id for result_chunk, _ in results] == [chunk.id]
    store.close()


def test_assemble_context_uses_splade_candidate_when_opted_in(
    store_and_index: tuple[IndexStore, SPLADEIndex],
) -> None:
    _, idx = store_and_index
    splade_results = idx.search("session commit database", top_k=5)
    graph = DependencyGraph()
    for chunk, _ in splade_results:
        graph.add_file_node(chunk.file_path)

    bundle = assemble_context(
        search_results=[],
        graph=graph,
        all_chunks=[chunk for chunk, _ in splade_results],
        question="session lifecycle",
        token_budget=1000,
        splade_results=splade_results,
    )

    assert bundle.retrieval_metadata.splade_used is True
    assert bundle.retrieval_metadata.splade_results == len(splade_results)
    assert {rc.chunk.id for rc in bundle.chunks} == {chunk.id for chunk, _ in splade_results}


# ---------------------------------------------------------------------------
# Sparse vector inspection
# ---------------------------------------------------------------------------


def test_get_sparse_vector(store_and_index: tuple[IndexStore, SPLADEIndex]) -> None:
    _, idx = store_and_index
    vec = idx.get_sparse_vector("utils.py:calculate_sum:5")
    assert isinstance(vec, dict)
    assert len(vec) > 0
    for term_id, weight in vec.items():
        assert isinstance(term_id, int)
        assert isinstance(weight, float)
        assert weight > 0


def test_get_sparse_vector_missing_chunk(
    store_and_index: tuple[IndexStore, SPLADEIndex],
) -> None:
    _, idx = store_and_index
    with pytest.raises(ArchexIndexError, match="No SPLADE vector"):
        idx.get_sparse_vector("nonexistent_chunk")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path, fake_encoder: FakeSPLADEEncoder) -> None:
    # Build index
    db1 = tmp_path / "save_test.db"
    store1 = IndexStore(db1)
    idx1 = SPLADEIndex(store1, encoder=fake_encoder)
    store1.insert_chunks(SAMPLE_CHUNKS)
    idx1.build(SAMPLE_CHUNKS)

    save_path = tmp_path / "splade_index.npz"
    idx1.save(save_path)
    store1.close()

    # Load into fresh store
    db2 = tmp_path / "load_test.db"
    store2 = IndexStore(db2)
    idx2 = SPLADEIndex(store2, encoder=fake_encoder)
    store2.insert_chunks(SAMPLE_CHUNKS)
    idx2.load(save_path, SAMPLE_CHUNKS)

    assert idx2.size == 3
    results = idx2.search("authenticate")
    assert len(results) > 0
    store2.close()


def test_save_empty_raises(tmp_path: Path, fake_encoder: FakeSPLADEEncoder) -> None:
    db = tmp_path / "empty_save.db"
    store = IndexStore(db)
    idx = SPLADEIndex(store, encoder=fake_encoder)
    with pytest.raises(ArchexIndexError, match="Cannot save empty"):
        idx.save(tmp_path / "empty.npz")
    store.close()


def test_load_missing_file_raises(tmp_path: Path, fake_encoder: FakeSPLADEEncoder) -> None:
    db = tmp_path / "missing.db"
    store = IndexStore(db)
    idx = SPLADEIndex(store, encoder=fake_encoder)
    with pytest.raises(ArchexIndexError, match="not found"):
        idx.load(tmp_path / "nonexistent.npz", SAMPLE_CHUNKS)
    store.close()


def test_load_model_mismatch_raises(tmp_path: Path, fake_encoder: FakeSPLADEEncoder) -> None:
    db1 = tmp_path / "mismatch_save.db"
    store1 = IndexStore(db1)
    idx1 = SPLADEIndex(store1, encoder=fake_encoder)
    store1.insert_chunks(SAMPLE_CHUNKS)
    idx1.build(SAMPLE_CHUNKS)

    save_path = tmp_path / "mismatch.npz"
    idx1.save(save_path)
    store1.close()

    db2 = tmp_path / "mismatch_load.db"
    store2 = IndexStore(db2)
    idx2 = SPLADEIndex(store2, model_name="different/model", encoder=fake_encoder)
    with pytest.raises(ArchexIndexError, match="model mismatch"):
        idx2.load(save_path, SAMPLE_CHUNKS)
    store2.close()


# ---------------------------------------------------------------------------
# Schema safety
# ---------------------------------------------------------------------------


def test_schema_created_on_init(tmp_path: Path, fake_encoder: FakeSPLADEEncoder) -> None:
    db = tmp_path / "schema_test.db"
    store = IndexStore(db)
    _ = SPLADEIndex(store, encoder=fake_encoder)
    tables = {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "splade_vectors" in tables
    assert "splade_inverted" in tables
    assert "splade_meta" in tables
    store.close()


# ---------------------------------------------------------------------------
# Encoder unit tests
# ---------------------------------------------------------------------------


def test_fake_encoder_deterministic() -> None:
    enc = FakeSPLADEEncoder()
    v1 = enc.encode(["hello world"])
    v2 = enc.encode(["hello world"])
    assert v1 == v2


def test_fake_encoder_different_texts_differ() -> None:
    enc = FakeSPLADEEncoder()
    v1 = enc.encode(["authentication login"])
    v2 = enc.encode(["database schema migration"])
    assert v1 != v2


# ---------------------------------------------------------------------------
# Integration test (requires model download — marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_encoder_produces_sparse_vectors() -> None:
    """Smoke test with the real SPLADE model."""
    encoder = SPLADEEncoder()
    results = encoder.encode(["session management in Flask"])
    assert len(results) == 1
    sparse = results[0]
    assert len(sparse) > 0
    # SPLADE should activate many vocabulary terms
    assert len(sparse) > 50


@pytest.mark.slow
def test_real_encoder_vocab_expansion() -> None:
    """SPLADE should expand NL query to include related code tokens."""
    encoder = SPLADEEncoder()
    sparse = encoder.encode(["session management"])[0]
    # Decode top-weighted tokens to verify expansion
    top_terms = sorted(sparse.items(), key=lambda x: x[1], reverse=True)[:20]
    top_ids = [tid for tid, _ in top_terms]
    decoded = encoder.decode_token_ids(top_ids)
    decoded_lower = [t.lower() for t in decoded]
    # "session" should appear in top terms
    assert any("session" in t for t in decoded_lower)
