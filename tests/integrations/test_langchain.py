"""Tests for the LangChain ArchexRetriever integration."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

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
        content="def foo(): pass",
        file_path="src/foo.py",
        start_line=1,
        end_line=1,
        symbol_name="foo",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=5,
    )
    ranked = RankedChunk(
        chunk=chunk,
        relevance_score=0.9,
        structural_score=0.8,
        type_coverage_score=0.7,
        final_score=0.85,
    )
    return ContextBundle(query="test query", chunks=[ranked])


class TestArchexRetrieverLangChain:
    def test_instantiation(self) -> None:
        from archex.integrations.langchain import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        assert retriever.repo_source == source
        assert retriever.token_budget == 8192
        assert retriever.config is None

    def test_instantiation_with_config(self) -> None:
        from archex.integrations.langchain import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        config = Config(languages=["python"])
        retriever = ArchexRetriever(repo_source=source, config=config, token_budget=4096)
        assert retriever.config == config
        assert retriever.token_budget == 4096

    def test_get_relevant_documents_calls_archex_query(self) -> None:
        from archex.integrations.langchain import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        bundle = _make_bundle()

        with patch("archex.api.query", return_value=bundle) as mock_query:
            docs = retriever.invoke("what does foo do?")

        mock_query.assert_called_once_with(
            source,
            "what does foo do?",
            token_budget=8192,
            config=None,
        )
        assert len(docs) == 1

    def test_document_mapping(self) -> None:
        from archex.integrations.langchain import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        bundle = _make_bundle()

        with patch("archex.api.query", return_value=bundle):
            docs = retriever.invoke("foo")

        doc = docs[0]
        assert doc.page_content == "def foo(): pass"
        assert doc.metadata["file_path"] == "src/foo.py"
        assert doc.metadata["start_line"] == 1
        assert doc.metadata["end_line"] == 1
        assert doc.metadata["symbol_name"] == "foo"
        assert doc.metadata["language"] == "python"
        assert doc.metadata["relevance_score"] == pytest.approx(0.85)

    def test_empty_bundle_returns_empty_list(self) -> None:
        from archex.integrations.langchain import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)
        empty_bundle = ContextBundle(query="nothing", chunks=[])

        with patch("archex.api.query", return_value=empty_bundle):
            docs = retriever.invoke("nothing")

        assert docs == []

    def test_import_guard_raises_when_langchain_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate langchain_core not being installed by patching the flag
        import archex.integrations.langchain as lc_module

        original = lc_module._langchain_available
        monkeypatch.setattr(lc_module, "_langchain_available", False)
        try:
            with pytest.raises(ArchexIndexError, match="langchain-core"):
                lc_module.ArchexRetriever(repo_source=RepoSource(local_path="/tmp"))
        finally:
            monkeypatch.setattr(lc_module, "_langchain_available", original)

    def test_multiple_chunks_all_mapped(self) -> None:
        from archex.integrations.langchain import ArchexRetriever

        source = RepoSource(local_path="/tmp/repo")
        retriever = ArchexRetriever(repo_source=source)

        chunks = [
            RankedChunk(
                chunk=CodeChunk(
                    id=f"chunk-{i}",
                    content=f"def fn_{i}(): pass",
                    file_path=f"src/f{i}.py",
                    start_line=i,
                    end_line=i,
                    language="python",
                    token_count=4,
                ),
                final_score=float(i) / 10,
            )
            for i in range(3)
        ]
        bundle = ContextBundle(query="q", chunks=chunks)

        with patch("archex.api.query", return_value=bundle):
            docs = retriever.invoke("q")

        assert len(docs) == 3
        assert docs[1].page_content == "def fn_1(): pass"


def test_langchain_stub_when_unavailable() -> None:
    """Cover the ImportError branch that defines the _BaseRetriever stub class."""
    import builtins
    import importlib
    import sys
    from typing import Any

    # Save langchain-related modules
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if "langchain" in key:
            saved[key] = sys.modules.pop(key)

    # Also remove the archex integration module so reload works
    archex_lc_key = "archex.integrations.langchain"
    if archex_lc_key in sys.modules:
        saved[archex_lc_key] = sys.modules.pop(archex_lc_key)

    original_import = builtins.__import__

    def mock_import(  # pyright: ignore[reportExplicitAny]
        name: str, *args: Any, **kwargs: Any
    ) -> object:
        if "langchain_core" in name:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = mock_import  # type: ignore[assignment]
        lc = importlib.import_module("archex.integrations.langchain")
        assert lc._langchain_available is False  # pyright: ignore[reportPrivateUsage]
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]
        # Restore saved modules
        sys.modules.update(saved)  # type: ignore[arg-type]
        # Reload to restore normal state
        if archex_lc_key in sys.modules:
            importlib.reload(sys.modules[archex_lc_key])
