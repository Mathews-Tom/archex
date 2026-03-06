"""LSAP integration: enrich archex symbols with LSP type information."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from archex.exceptions import LSAPError
from archex.integrations.lsap_models import (
    DefinitionLocation,
    HoverInfo,
    LSAPEnrichment,
    ReferenceLocation,
)
from archex.models import SymbolSource  # noqa: TCH001

if TYPE_CHECKING:
    from archex.models import DetectedPattern, ParsedFile

try:
    from lsp_client import Client  # type: ignore[import-untyped]

    _lsap_available = True
except ImportError:
    _lsap_available = False

    class Client:  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)

# Keywords indicating data-store interaction in hover content.
_DATASTORE_INDICATORS = frozenset(
    {
        "session",
        "connection",
        "cursor",
        "transaction",
        "engine",
        "sessionmaker",
        "asyncsession",
        "pool",
        "database",
        "db",
        "repository",
        "collection",
        "table",
        "query",
        "sqlalchemy",
        "asyncpg",
        "psycopg",
        "pymongo",
        "redis",
    }
)


class LSAPEnrichedLookup:
    """Wraps an initialized ``lsp_client.Client`` to enrich archex symbols.

    The caller is responsible for starting and stopping the language server.
    This class performs three LSP operations (hover, references, definition)
    and maps the results onto archex data models.
    """

    def __init__(self, lsp_client: Client) -> None:  # type: ignore[type-arg]
        if not _lsap_available:
            raise LSAPError("Install lsp-client: uv add archex[lsap]")
        self._client: Any = lsp_client

    async def get_hover(self, file_path: str, line: int, character: int = 0) -> HoverInfo:
        """Return hover information for a position."""
        result = await self._client.request_hover(file_path, line, character)
        if result is None:
            return HoverInfo()
        raw = str(result.get("contents", ""))
        # Extract type signature from the first line of hover content.
        lines = raw.strip().splitlines()
        type_sig = lines[0] if lines else ""
        doc = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        return HoverInfo(type_signature=type_sig, documentation=doc, raw_content=raw)

    async def get_references(
        self, file_path: str, line: int, character: int = 0, max_items: int = 50
    ) -> list[ReferenceLocation]:
        """Return reference locations for a position."""
        results = await self._client.request_references(file_path, line, character)
        if not results:
            return []
        refs: list[ReferenceLocation] = []
        for ref in results[:max_items]:
            refs.append(
                ReferenceLocation(
                    file_path=ref.get("uri", ""),
                    line=ref.get("range", {}).get("start", {}).get("line", 0),
                    character=ref.get("range", {}).get("start", {}).get("character", 0),
                    context_line=ref.get("context", ""),
                )
            )
        return refs

    async def get_definition(
        self, file_path: str, line: int, character: int = 0
    ) -> DefinitionLocation | None:
        """Return the definition location for a position."""
        result = await self._client.request_definition(file_path, line, character)
        if not result:
            return None
        # LSP may return a list; take the first entry.
        entry: dict[str, Any] = cast(
            "dict[str, Any]",
            result[0] if isinstance(result, list) else result,
        )
        return DefinitionLocation(
            file_path=str(entry.get("uri", "")),
            line=int(entry.get("range", {}).get("start", {}).get("line", 0)),
            character=int(entry.get("range", {}).get("start", {}).get("character", 0)),
            context_line=str(entry.get("context", "")),
        )

    async def enrich_symbol(self, symbol: SymbolSource) -> SymbolSource:
        """Enrich a single SymbolSource with hover, references, and definition.

        Each LSP call is isolated — a failure in one does not block the others.
        Returns a new ``SymbolSource`` via ``model_copy()``; the original is unmodified.
        """
        hover: HoverInfo | None = None
        refs: list[ReferenceLocation] = []
        defn: DefinitionLocation | None = None

        try:
            hover = await self.get_hover(symbol.file_path, symbol.start_line)
        except Exception:
            logger.debug("hover failed for %s", symbol.symbol_id, exc_info=True)

        try:
            refs = await self.get_references(symbol.file_path, symbol.start_line)
        except Exception:
            logger.debug("references failed for %s", symbol.symbol_id, exc_info=True)

        try:
            defn = await self.get_definition(symbol.file_path, symbol.start_line)
        except Exception:
            logger.debug("definition failed for %s", symbol.symbol_id, exc_info=True)

        enrichment = LSAPEnrichment(
            hover=hover,
            references=refs,
            definition=defn,
            reference_count=len(refs),
        )
        return symbol.model_copy(update={"lsap_enrichment": enrichment})

    async def enrich_symbols_batch(
        self, symbols: list[SymbolSource], concurrency: int = 5
    ) -> list[SymbolSource]:
        """Enrich multiple symbols concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _enrich(sym: SymbolSource) -> SymbolSource:
            async with semaphore:
                return await self.enrich_symbol(sym)

        return list(await asyncio.gather(*[_enrich(s) for s in symbols]))


async def verify_repository_pattern(
    lookup: LSAPEnrichedLookup,
    pattern: DetectedPattern,
    parsed_files: list[ParsedFile],
) -> float | None:
    """Verify a repository pattern using LSP hover data.

    For ``repository`` patterns only: checks whether CRUD methods reference
    data-store types (session, connection, cursor, etc.) via hover info.

    Returns an adjusted confidence, or ``None`` to keep the original.
    """
    if pattern.name.lower() != "repository":
        return None

    if not pattern.evidence:
        return None

    indicator_hits = 0
    total_checks = 0

    for evidence in pattern.evidence:
        file_path = evidence.file_path
        # Find the parsed file to get the symbol's start line.
        pf = next((f for f in parsed_files if f.path == file_path), None)
        if pf is None:
            continue
        for sym in pf.symbols:
            if sym.name in evidence.explanation:
                total_checks += 1
                try:
                    hover = await lookup.get_hover(file_path, sym.start_line)
                except Exception:
                    continue
                raw_lower = hover.raw_content.lower()
                if any(kw in raw_lower for kw in _DATASTORE_INDICATORS):
                    indicator_hits += 1

    if total_checks == 0:
        return None

    hit_ratio = indicator_hits / total_checks
    if hit_ratio >= 0.5:
        return min(pattern.confidence + 0.10, 1.0)
    if hit_ratio == 0.0:
        return max(pattern.confidence - 0.15, 0.0)
    return None
