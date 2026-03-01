"""LangChain integration: expose archex query pipeline as a LangChain retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError
from archex.models import Config, RepoSource  # noqa: TCH001

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document

try:
    from langchain_core.retrievers import (
        BaseRetriever as _BaseRetriever,  # type: ignore[assignment]
    )

    _langchain_available = True
except ImportError:
    _langchain_available = False

    class _BaseRetriever:  # type: ignore[no-redef]
        pass


class ArchexRetriever(_BaseRetriever):  # type: ignore[misc]
    """LangChain retriever backed by the archex query pipeline.

    Maps each RankedChunk from a ContextBundle to a LangChain Document.

    Requires ``langchain-core``: ``uv add archex[langchain]``.
    """

    repo_source: RepoSource
    config: Config | None = None
    token_budget: int = 8192

    def __init__(self, **data: Any) -> None:
        if not _langchain_available:
            raise ArchexIndexError("Install langchain-core: uv add archex[langchain]")
        super().__init__(**data)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # type: ignore[override]
    ) -> list[Document]:
        from langchain_core.documents import Document as _Document

        from archex.api import query as archex_query

        bundle = archex_query(
            self.repo_source,
            query,
            token_budget=self.token_budget,
            config=self.config,
        )
        return [
            _Document(
                page_content=rc.chunk.content,
                metadata={
                    "file_path": rc.chunk.file_path,
                    "start_line": rc.chunk.start_line,
                    "end_line": rc.chunk.end_line,
                    "symbol_name": rc.chunk.symbol_name,
                    "language": rc.chunk.language,
                    "relevance_score": rc.final_score,
                },
            )
            for rc in bundle.chunks
        ]
