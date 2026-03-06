"""Tests for the LlamaIndex ArchexRetriever integration."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("llama_index.core", reason="llama-index-core not installed")

from archex.exceptions import ArchexIndexError
from archex.models import (
    CodeChunk,
    Config,
    ContextBundle,
    RankedChunk,
    RepoSource,
    SymbolKind,
)


def _make_bundle() -> ContextBundle:
    chunk = CodeChunk(
        id="chunk-1",
        content="fn bar() {}",
        file_path="src/bar.rs",
        start_line=5,
        end_line=7,
        symbol_name="bar",
        symbol_kind=SymbolKind.FUNCTION,
        language="rust",
        token_count=6,
    )
    ranked = RankedChunk(
        chunk=chunk,
        relevance_score=0.88,
        structural_score=0.75,
        type_coverage_score=0.6,
        final_score=0.82,
    )
    return ContextBundle(query="bar function", chunks=[ranked])


class TestArchexRetrieverLlamaIndex:
    def test_instantiation(self) -> None:
        from archex.integrations.llamaindex import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        assert retriever._repo_source == source
        assert retriever._token_budget == 8192
        assert retriever._config is None

    def test_instantiation_with_config(self) -> None:
        from archex.integrations.llamaindex import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        config = Config(languages=["rust"])
        retriever = ArchexRetriever(repo_source=source, config=config, token_budget=2048)
        assert retriever._config == config
        assert retriever._token_budget == 2048

    def test_retrieve_calls_archex_query(self) -> None:
        from llama_index.core.schema import QueryBundle

        from archex.integrations.llamaindex import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        bundle = _make_bundle()
        qb = QueryBundle(query_str="bar function")

        with patch("archex.api.query", return_value=bundle) as mock_query:
            nodes = retriever._retrieve(qb)

        mock_query.assert_called_once_with(
            source,
            "bar function",
            token_budget=8192,
            config=None,
        )
        assert len(nodes) == 1

    def test_node_mapping(self) -> None:
        from llama_index.core.schema import QueryBundle

        from archex.integrations.llamaindex import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        bundle = _make_bundle()
        qb = QueryBundle(query_str="bar")

        with patch("archex.api.query", return_value=bundle):
            nodes = retriever._retrieve(qb)

        nws = nodes[0]
        assert nws.score == pytest.approx(0.82)
        assert nws.node.text == "fn bar() {}"
        assert nws.node.metadata["file_path"] == "src/bar.rs"
        assert nws.node.metadata["start_line"] == 5
        assert nws.node.metadata["end_line"] == 7
        assert nws.node.metadata["symbol_name"] == "bar"
        assert nws.node.metadata["language"] == "rust"

    def test_empty_bundle_returns_empty_list(self) -> None:
        from llama_index.core.schema import QueryBundle

        from archex.integrations.llamaindex import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        empty_bundle = ContextBundle(query="nothing", chunks=[])
        qb = QueryBundle(query_str="nothing")

        with patch("archex.api.query", return_value=empty_bundle):
            nodes = retriever._retrieve(qb)

        assert nodes == []

    def test_import_guard_raises_when_llamaindex_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import archex.integrations.llamaindex as li_module

        original = li_module._llamaindex_available
        monkeypatch.setattr(li_module, "_llamaindex_available", False)
        try:
            with pytest.raises(ArchexIndexError, match="llama-index-core"):
                li_module.ArchexRetriever(repo_source=RepoSource(local_path="/tmp"))
        finally:
            monkeypatch.setattr(li_module, "_llamaindex_available", original)

    def test_multiple_chunks_all_mapped(self) -> None:
        from llama_index.core.schema import QueryBundle

        from archex.integrations.llamaindex import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)

        chunks = [
            RankedChunk(
                chunk=CodeChunk(
                    id=f"chunk-{i}",
                    content=f"fn f{i}() {{}}",
                    file_path=f"src/f{i}.rs",
                    start_line=i,
                    end_line=i + 2,
                    language="rust",
                    token_count=4,
                ),
                final_score=float(i) / 10,
            )
            for i in range(3)
        ]
        bundle = ContextBundle(query="q", chunks=chunks)
        qb = QueryBundle(query_str="q")

        with patch("archex.api.query", return_value=bundle):
            nodes = retriever._retrieve(qb)

        assert len(nodes) == 3
        assert nodes[2].node.text == "fn f2() {}"
        assert nodes[2].score == pytest.approx(0.2)


def test_llamaindex_stub_when_unavailable() -> None:
    """Cover the ImportError branch that defines the _LIBase stub class."""
    import builtins
    import importlib
    import sys
    from typing import Any

    # Save llama_index-related modules
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if "llama_index" in key or "llamaindex" in key:
            saved[key] = sys.modules.pop(key)

    # Also remove the archex integration module so reload works
    archex_li_key = "archex.integrations.llamaindex"
    if archex_li_key in sys.modules:
        saved[archex_li_key] = sys.modules.pop(archex_li_key)

    original_import = builtins.__import__

    def mock_import(  # pyright: ignore[reportExplicitAny]
        name: str, *args: Any, **kwargs: Any
    ) -> object:
        if "llama_index" in name:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = mock_import  # type: ignore[assignment]
        li = importlib.import_module("archex.integrations.llamaindex")
        assert li._llamaindex_available is False  # pyright: ignore[reportPrivateUsage]
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]
        # Restore saved modules
        sys.modules.update(saved)  # type: ignore[arg-type]
        # Reload to restore normal state
        if archex_li_key in sys.modules:
            importlib.reload(sys.modules[archex_li_key])
