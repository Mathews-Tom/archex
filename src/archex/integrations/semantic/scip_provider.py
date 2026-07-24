"""SCIP evidence provider: reads a compiler-produced SCIP index off disk.

SCIP (github.com/sourcegraph/scip) indexes are produced by external,
language-specific indexers (scip-python, scip-typescript, scip-clang, ...)
and describe compiler-verified definition/reference/implementation
relationships as a protobuf-encoded ``Index`` message. This provider never
runs an indexer itself — archex only *consumes* a pre-built index at a
configured, repo-relative path. When no index is present, or the runtime
``protobuf`` package (the ``archex[scip]`` extra) is not installed, the
provider reports ``UNAVAILABLE`` rather than falling back to any other
evidence source.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING

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

    from archex.models import ParsedFile

try:
    from archex.integrations.semantic import scip_pb2

    _scip_runtime_available = True
except ImportError:
    _scip_runtime_available = False
    scip_pb2 = None  # type: ignore[assignment]

#: Symbol-role bit for a definition occurrence (SymbolRole.Definition = 0x1).
_DEFINITION_ROLE_BIT = 0x1

#: Hard cap on emitted evidence records per run — bounds the cost of a very
#: large SCIP index the same way DependencyGraph.add_co_directory_edges()
#: bounds dense-directory fan-out. A capped run is reported PARTIAL, never
#: silently truncated without saying so.
_MAX_EVIDENCE_RECORDS = 20_000

#: Below this fraction of parsed_files having any SCIP document coverage,
#: the index is treated as stale relative to the current checkout rather
#: than trusted as available evidence.
_MIN_COVERAGE_RATIO = 0.10


class ScipEvidenceProviderError(Exception):
    """Raised only for programmer errors, never for ordinary unavailability."""


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _occurrence_position(occ: object) -> tuple[int, int] | None:
    """Return (line, character) for an occurrence's start position, or None."""
    typed_range = occ.WhichOneof("typed_range")  # type: ignore[attr-defined]
    if typed_range == "single_line_range":
        r = occ.single_line_range  # type: ignore[attr-defined]
        return int(r.line), int(r.start_character)
    if typed_range == "multi_line_range":
        r = occ.multi_line_range  # type: ignore[attr-defined]
        return int(r.start_line), int(r.start_character)
    legacy = list(occ.range)  # type: ignore[attr-defined]
    if len(legacy) == 3:
        return int(legacy[0]), int(legacy[1])
    if len(legacy) == 4:
        return int(legacy[0]), int(legacy[1])
    return None


class ScipEvidenceProvider:
    """Reads definition/reference/implementation evidence from a SCIP index."""

    def __init__(
        self,
        *,
        index_path: str = "index.scip",
        confidence_definition: float = 0.97,
        confidence_reference: float = 0.93,
        confidence_implementation: float = 0.95,
    ) -> None:
        self._index_path = index_path
        self._confidence = {
            SemanticEdgeKind.DEFINITION: confidence_definition,
            SemanticEdgeKind.REFERENCE: confidence_reference,
            SemanticEdgeKind.IMPLEMENTATION: confidence_implementation,
        }

    @property
    def name(self) -> SemanticProviderName:
        return SemanticProviderName.SCIP

    def _resolved_index_path(self, repo_root: Path) -> Path:
        return repo_root / self._index_path

    def probe(self, repo_root: Path) -> SemanticProviderReceipt:
        if not _scip_runtime_available:
            return SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason="protobuf runtime unavailable: install archex[scip]",
                collected_at=_now_iso(),
            )
        index_file = self._resolved_index_path(repo_root)
        if not index_file.is_file():
            return SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"no SCIP index found at {self._index_path}",
                collected_at=_now_iso(),
            )
        if index_file.stat().st_size == 0:
            return SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"SCIP index at {self._index_path} is empty",
                collected_at=_now_iso(),
            )
        return SemanticProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            collected_at=_now_iso(),
        )

    def collect(
        self, parsed_files: list[ParsedFile], repo_root: Path
    ) -> tuple[list[SemanticEdgeEvidence], SemanticProviderReceipt]:
        probe_receipt = self.probe(repo_root)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt

        index_file = self._resolved_index_path(repo_root)
        index = scip_pb2.Index()  # type: ignore[union-attr]
        try:
            index.ParseFromString(index_file.read_bytes())
        except Exception as exc:  # noqa: BLE001 — any decode failure is a stale/corrupt index
            return [], SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.STALE,
                reason=f"could not decode SCIP index at {self._index_path}: {exc}",
                collected_at=_now_iso(),
            )

        tool_name = index.metadata.tool_info.name or None
        tool_version = index.metadata.tool_info.version or "unknown"

        parsed_paths = {_normalize(f.path) for f in parsed_files}
        scip_paths = {_normalize(doc.relative_path) for doc in index.documents}
        covered = parsed_paths & scip_paths
        coverage_ratio = (len(covered) / len(parsed_paths)) if parsed_paths else 0.0
        if parsed_paths and coverage_ratio < _MIN_COVERAGE_RATIO:
            return [], SemanticProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.STALE,
                reason=(
                    f"SCIP index at {self._index_path} covers only "
                    f"{len(covered)}/{len(parsed_paths)} indexed files "
                    f"({coverage_ratio:.0%}); likely stale relative to the current checkout"
                ),
                tool_name=tool_name,
                tool_version=tool_version,
                files_attempted=len(parsed_paths),
                collected_at=_now_iso(),
            )

        definitions: dict[str, SemanticEvidenceLocation] = {}
        for doc in index.documents:
            doc_path = _normalize(doc.relative_path)
            for occ in doc.occurrences:
                if not occ.symbol or not (occ.symbol_roles & _DEFINITION_ROLE_BIT):
                    continue
                pos = _occurrence_position(occ)
                if pos is None:
                    continue
                line, character = pos
                definitions.setdefault(
                    occ.symbol,
                    SemanticEvidenceLocation(
                        file_path=doc_path, line=line, character=character, symbol=occ.symbol
                    ),
                )

        evidence: list[SemanticEdgeEvidence] = []
        truncated = False

        def _emit(
            kind: SemanticEdgeKind,
            source: SemanticEvidenceLocation,
            target: SemanticEvidenceLocation,
        ) -> bool:
            nonlocal truncated
            if source.file_path == target.file_path:
                return True
            if len(evidence) >= _MAX_EVIDENCE_RECORDS:
                truncated = True
                return False
            evidence.append(
                SemanticEdgeEvidence(
                    provider=self.name,
                    provider_version=tool_version,
                    kind=kind,
                    source=source,
                    target=target,
                    confidence=self._confidence[kind],
                )
            )
            return True

        succeeded_files: set[str] = set()
        for doc in index.documents:
            doc_path = _normalize(doc.relative_path)
            for occ in doc.occurrences:
                if not occ.symbol or (occ.symbol_roles & _DEFINITION_ROLE_BIT):
                    continue
                definition = definitions.get(occ.symbol)
                if definition is None:
                    continue
                pos = _occurrence_position(occ)
                if pos is None:
                    continue
                line, character = pos
                usage = SemanticEvidenceLocation(
                    file_path=doc_path, line=line, character=character, symbol=occ.symbol
                )
                if usage.file_path != definition.file_path and _emit(
                    SemanticEdgeKind.DEFINITION, usage, definition
                ):
                    succeeded_files.add(usage.file_path)
                    succeeded_files.add(definition.file_path)
                if _emit(SemanticEdgeKind.REFERENCE, definition, usage):
                    succeeded_files.add(usage.file_path)
                    succeeded_files.add(definition.file_path)

        all_symbol_infos = list(index.external_symbols)
        for doc in index.documents:
            all_symbol_infos.extend(doc.symbols)
        for sym_info in all_symbol_infos:
            implementer = definitions.get(sym_info.symbol)
            if implementer is None:
                continue
            for rel in sym_info.relationships:
                if not rel.is_implementation:
                    continue
                base = definitions.get(rel.symbol)
                if base is None:
                    continue
                if _emit(SemanticEdgeKind.IMPLEMENTATION, implementer, base):
                    succeeded_files.add(implementer.file_path)
                    succeeded_files.add(base.file_path)

        availability = ProviderAvailability.PARTIAL if truncated else ProviderAvailability.AVAILABLE
        reason = f"evidence capped at {_MAX_EVIDENCE_RECORDS} records" if truncated else ""
        receipt = SemanticProviderReceipt(
            provider=self.name,
            availability=availability,
            reason=reason,
            tool_name=tool_name,
            tool_version=tool_version,
            files_attempted=len(parsed_paths),
            files_succeeded=len(succeeded_files & parsed_paths),
            evidence_count=len(evidence),
            collected_at=_now_iso(),
        )
        return evidence, receipt
