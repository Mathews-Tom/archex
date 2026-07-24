"""LSAP/LSP evidence provider: queries a live language server for evidence.

Reuses ``archex.integrations.lsap.LSAPEnrichedLookup`` (the existing LSP
integration wrapper around the optional ``lsp-client`` package,
``archex[lsap]``) to run definition/references/implementation requests per
symbol and turn cross-file results into semantic edge evidence.

This provider does not spawn or manage a language server process — archex has
no server orchestration layer for arbitrary languages. A caller supplies an
already-connected ``lsp_client.Client``; without one, the provider reports
``UNAVAILABLE`` rather than guessing at a connection or falling back to any
other evidence source.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from archex.integrations.lsap import LSAPEnrichedLookup, lsap_available
from archex.integrations.semantic.models import (
    ProviderAvailability,
    SemanticEdgeEvidence,
    SemanticEdgeKind,
    SemanticEvidenceLocation,
    SemanticProviderName,
    SemanticProviderReceipt,
)

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.lsap_models import (
        DefinitionLocation,
        ImplementationLocation,
        ReferenceLocation,
    )
    from archex.models import ParsedFile

logger = logging.getLogger(__name__)

#: Hard cap on symbols queried per run — an LSP round trip per symbol is
#: costly, so a large repository is bounded rather than exhaustively walked.
#: A capped run is reported PARTIAL, never silently truncated without saying
#: so.
_MAX_SYMBOLS = 200

_LSP_TOOL_VERSION = "lsp-client"


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


class LspEvidenceProvider:
    """Reads definition/reference/implementation evidence from a live LSP server."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        max_symbols: int = _MAX_SYMBOLS,
        confidence_definition: float = 0.85,
        confidence_reference: float = 0.80,
        confidence_implementation: float = 0.80,
    ) -> None:
        self._client: Any = client
        self._max_symbols = max_symbols
        self._confidence = {
            SemanticEdgeKind.DEFINITION: confidence_definition,
            SemanticEdgeKind.REFERENCE: confidence_reference,
            SemanticEdgeKind.IMPLEMENTATION: confidence_implementation,
        }

    @property
    def name(self) -> SemanticProviderName:
        return SemanticProviderName.LSP

    def probe(self, repo_root: Path) -> SemanticProviderReceipt:
        del repo_root  # unused: availability depends on the injected client, not the repo path
        if not lsap_available():
            return SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason="lsp-client not installed: install archex[lsap]",
                collected_at=_now_iso(),
            )
        if self._client is None:
            return SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=(
                    "no LSP client configured: construct LspEvidenceProvider with a "
                    "connected lsp_client.Client"
                ),
                collected_at=_now_iso(),
            )
        return SemanticProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            tool_name="lsp-client",
            collected_at=_now_iso(),
        )

    def collect(
        self, parsed_files: list[ParsedFile], repo_root: Path
    ) -> tuple[list[SemanticEdgeEvidence], SemanticProviderReceipt]:
        probe_receipt = self.probe(repo_root)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt
        assert self._client is not None  # narrowed by probe() returning AVAILABLE
        return asyncio.run(self._collect_async(parsed_files, self._client))

    async def _collect_async(
        self, parsed_files: list[ParsedFile], client: Any
    ) -> tuple[list[SemanticEdgeEvidence], SemanticProviderReceipt]:
        lookup = LSAPEnrichedLookup(client)
        symbols = [
            (sym.name, _normalize(pf.path), sym.start_line)
            for pf in parsed_files
            for sym in pf.symbols
        ]
        truncated = len(symbols) > self._max_symbols
        symbols = symbols[: self._max_symbols]

        evidence: list[SemanticEdgeEvidence] = []
        attempted_files: set[str] = set()
        succeeded_files: set[str] = set()

        for symbol_name, file_path, line in symbols:
            attempted_files.add(file_path)
            source = SemanticEvidenceLocation(
                file_path=file_path, line=line, character=0, symbol=symbol_name
            )

            definition: DefinitionLocation | None = None
            try:
                definition = await lookup.get_definition(file_path, line)
            except Exception:
                logger.debug(
                    "LSP definition lookup failed for %s:%d", file_path, line, exc_info=True
                )
            if definition is not None and definition.file_path:
                target_path = _normalize(definition.file_path)
                if target_path != file_path:
                    evidence.append(
                        SemanticEdgeEvidence(
                            provider=self.name,
                            provider_version=_LSP_TOOL_VERSION,
                            kind=SemanticEdgeKind.DEFINITION,
                            source=source,
                            target=SemanticEvidenceLocation(
                                file_path=target_path,
                                line=definition.line,
                                character=definition.character,
                                symbol=symbol_name,
                            ),
                            confidence=self._confidence[SemanticEdgeKind.DEFINITION],
                        )
                    )
                    succeeded_files.add(file_path)
                    succeeded_files.add(target_path)

            references: list[ReferenceLocation] = []
            try:
                references = await lookup.get_references(file_path, line)
            except Exception:
                logger.debug(
                    "LSP references lookup failed for %s:%d", file_path, line, exc_info=True
                )
            for ref in references:
                if not ref.file_path:
                    continue
                target_path = _normalize(ref.file_path)
                if target_path == file_path:
                    continue
                evidence.append(
                    SemanticEdgeEvidence(
                        provider=self.name,
                        provider_version=_LSP_TOOL_VERSION,
                        kind=SemanticEdgeKind.REFERENCE,
                        source=source,
                        target=SemanticEvidenceLocation(
                            file_path=target_path,
                            line=ref.line,
                            character=ref.character,
                            symbol=symbol_name,
                        ),
                        confidence=self._confidence[SemanticEdgeKind.REFERENCE],
                    )
                )
                succeeded_files.add(file_path)
                succeeded_files.add(target_path)

            implementation: ImplementationLocation | None = None
            try:
                implementation = await lookup.get_implementation(file_path, line)
            except Exception:
                logger.debug(
                    "LSP implementation lookup failed for %s:%d", file_path, line, exc_info=True
                )
            if implementation is not None and implementation.file_path:
                target_path = _normalize(implementation.file_path)
                if target_path != file_path:
                    evidence.append(
                        SemanticEdgeEvidence(
                            provider=self.name,
                            provider_version=_LSP_TOOL_VERSION,
                            kind=SemanticEdgeKind.IMPLEMENTATION,
                            source=source,
                            target=SemanticEvidenceLocation(
                                file_path=target_path,
                                line=implementation.line,
                                character=implementation.character,
                                symbol=symbol_name,
                            ),
                            confidence=self._confidence[SemanticEdgeKind.IMPLEMENTATION],
                        )
                    )
                    succeeded_files.add(file_path)
                    succeeded_files.add(target_path)

        availability = ProviderAvailability.PARTIAL if truncated else ProviderAvailability.AVAILABLE
        reason = f"symbol queries capped at {self._max_symbols}" if truncated else ""
        receipt = SemanticProviderReceipt(
            provider=self.name,
            availability=availability,
            reason=reason,
            tool_name="lsp-client",
            files_attempted=len(attempted_files),
            files_succeeded=len(succeeded_files & attempted_files),
            evidence_count=len(evidence),
            collected_at=_now_iso(),
        )
        return evidence, receipt
