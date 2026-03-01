"""LlamaIndex integration: wrap archex query pipeline as a LlamaIndex retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError
from archex.models import Config, RepoSource  # noqa: TCH001

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore, QueryBundle

try:
    from llama_index.core.retrievers import BaseRetriever as _LIBase  # type: ignore[assignment]

    _llamaindex_available: bool = True
except ImportError:
    _llamaindex_available = False

    class _LIBase:  # type: ignore[no-redef]
        """Stub used when llama-index-core is not installed."""


class ArchexRetriever(_LIBase):  # type: ignore[misc]
    """LlamaIndex retriever backed by the archex query pipeline.

    Maps each RankedChunk from a ContextBundle to a NodeWithScore.

    Requires ``llama-index-core``: ``uv add archex[llamaindex]``.
    """

    def __init__(
        self,
        repo_source: RepoSource,
        config: Config | None = None,
        token_budget: int = 8192,
        **kwargs: Any,
    ) -> None:
        if not _llamaindex_available:
            raise ArchexIndexError("Install llama-index-core: uv add archex[llamaindex]")
        super().__init__(**kwargs)
        self._repo_source = repo_source
        self._config = config
        self._token_budget = token_budget

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:  # type: ignore[override]
        from llama_index.core.schema import NodeWithScore as _NodeWithScore
        from llama_index.core.schema import TextNode as _TextNode

        from archex.api import query as archex_query

        bundle = archex_query(
            self._repo_source,
            query_bundle.query_str,
            token_budget=self._token_budget,
            config=self._config,
        )
        return [
            _NodeWithScore(
                node=_TextNode(
                    text=rc.chunk.content,
                    metadata={
                        "file_path": rc.chunk.file_path,
                        "start_line": rc.chunk.start_line,
                        "end_line": rc.chunk.end_line,
                        "symbol_name": rc.chunk.symbol_name,
                        "language": rc.chunk.language,
                    },
                ),
                score=rc.final_score,
            )
            for rc in bundle.chunks
        ]
