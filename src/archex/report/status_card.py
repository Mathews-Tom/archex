"""StatusCard: a dimensioned, evidence-linked documentation/release status summary (M9).

Deliberately not a scored artifact: every dimension carries its own
factual, evidence-linked state (``evidenced`` or ``unknown``) and there is
no field anywhere in this module -- on ``StatusDimension`` or on
``StatusCard`` itself -- that aggregates dimensions into a single score,
letter grade, or health rating. A reader must weigh each dimension
independently; archex never claims an overall verdict about a repository's
documentation or release posture.

Unlike M7's runtime/coverage evidence and M8's history evidence, which are
diff-scoped (attached to a specific changed file or symbol), this card is
repository-scoped: it summarizes the whole M9 documentation-evidence
channel (doc links, ADR records, ownership records) plus locally
verifiable release/CI evidence, read from the same read-only index every
other report command uses. Building the card never mutates the analyzed
repository -- it only reads a previously built index and files already on
disk (``CHANGELOG.md``, ``.github/workflows/``).
"""

from __future__ import annotations

import json
import re
import time
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, model_validator

from archex.api import index_repository
from archex.cache import CacheManager
from archex.config import load_config, load_index_config
from archex.integrations.docs.models import (
    AdrRecord,
    DocEvidenceProviderName,
    DocProviderReceipt,
    DocumentationLink,
    OwnershipRecord,
    ProviderAvailability,
)
from archex.models import RepoSource

STATUS_CARD_SCHEMA_VERSION = "1.0.0"

#: Bounds how many example evidence entries one dimension lists -- keeps the
#: card small and reviewable rather than dumping every collected record.
MAX_DIMENSION_EVIDENCE = 10

#: Conventional read-only CI workflow locations checked for the "Release &
#: CI evidence" dimension, in order. Presence is evidence of a pinned,
#: read-only example existing -- never a claim about its current run state.
_CI_WORKFLOW_CANDIDATES = (".github/workflows/report-diff.yml",)

_CHANGELOG_VERSION_PATTERN = re.compile(r"^## \[([^\]]+)\](?:\s*-\s*(.+))?$", re.MULTILINE)


class StatusCardError(ValueError):
    """Raised when a status card cannot be built."""


class StatusDimensionState(StrEnum):
    """A dimension's own evidence state. Never combined across dimensions."""

    EVIDENCED = "evidenced"
    UNKNOWN = "unknown"


class StatusDimensionEvidence(BaseModel):
    """One concrete, locally verifiable pointer backing a dimension's state."""

    description: str
    location: str

    @model_validator(mode="after")
    def _validate_evidence(self) -> StatusDimensionEvidence:
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if not self.location.strip():
            raise ValueError("location must not be empty")
        return self


class StatusDimension(BaseModel):
    """One independent, evidence-linked axis of the status card.

    ``detail`` is always a factual statement (a count, a path, a verbatim
    declared value) -- never a subjective rating. ``evidence`` must be
    non-empty whenever ``state`` is ``EVIDENCED``: a dimension claiming
    evidence exists must point at it.
    """

    name: str
    state: StatusDimensionState
    detail: str
    provider: str
    evidence: list[StatusDimensionEvidence] = []

    @model_validator(mode="after")
    def _validate_dimension(self) -> StatusDimension:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.detail.strip():
            raise ValueError("detail must not be empty")
        if self.state == StatusDimensionState.EVIDENCED and not self.evidence:
            raise ValueError("evidence must not be empty when state is EVIDENCED")
        return self


class StatusCard(BaseModel):
    """The canonical, read-only dimensioned status card every renderer projects.

    There is intentionally no ``score``, ``grade``, or ``health`` field on
    this model, on ``StatusDimension``, or anywhere in this module: the
    absence is structural, not a runtime check, so no code path can
    construct a composite rating even by accident.
    """

    schema_version: str = STATUS_CARD_SCHEMA_VERSION
    source_identity: str
    revision: str
    generated_at: str
    dimensions: list[StatusDimension]

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


def _provider_receipt(
    receipts: list[DocProviderReceipt], provider: DocEvidenceProviderName
) -> DocProviderReceipt | None:
    return next((receipt for receipt in receipts if receipt.provider == provider), None)


def _unknown_dimension(name: str, provider: str, reason: str) -> StatusDimension:
    return StatusDimension(
        name=name, state=StatusDimensionState.UNKNOWN, detail=reason, provider=provider
    )


def _doc_link_dimension(
    doc_links: list[DocumentationLink], receipts: list[DocProviderReceipt]
) -> StatusDimension:
    receipt = _provider_receipt(receipts, DocEvidenceProviderName.DOC_LINK)
    if receipt is None:
        return _unknown_dimension(
            "Documentation linkage", "doc_link", "doc_link provider not configured for this index"
        )
    if receipt.availability != ProviderAvailability.AVAILABLE or not doc_links:
        return _unknown_dimension(
            "Documentation linkage",
            "doc_link",
            receipt.reason or "doc_link provider produced no evidence",
        )
    documented_files = sorted({link.target_path for link in doc_links})
    return StatusDimension(
        name="Documentation linkage",
        state=StatusDimensionState.EVIDENCED,
        detail=(
            f"{len(doc_links)} documentation link(s) reference {len(documented_files)} "
            "distinct source path(s)"
        ),
        provider="doc_link",
        evidence=[
            StatusDimensionEvidence(
                description=f"linked from {link.doc_path}", location=link.target_path
            )
            for link in doc_links[:MAX_DIMENSION_EVIDENCE]
        ],
    )


def _adr_dimension(
    adr_records: list[AdrRecord], receipts: list[DocProviderReceipt]
) -> StatusDimension:
    receipt = _provider_receipt(receipts, DocEvidenceProviderName.ADR)
    if receipt is None:
        return _unknown_dimension(
            "ADR provenance", "adr", "adr provider not configured for this index"
        )
    if receipt.availability != ProviderAvailability.AVAILABLE or not adr_records:
        return _unknown_dimension(
            "ADR provenance", "adr", receipt.reason or "adr provider produced no evidence"
        )
    return StatusDimension(
        name="ADR provenance",
        state=StatusDimensionState.EVIDENCED,
        detail=f"{len(adr_records)} architecture-decision-record(s) found",
        provider="adr",
        evidence=[
            StatusDimensionEvidence(
                description=f"{record.adr_id}: {record.title} (status: {record.status})",
                location=record.doc_path,
            )
            for record in adr_records[:MAX_DIMENSION_EVIDENCE]
        ],
    )


def _ownership_dimension(
    ownership_records: list[OwnershipRecord], receipts: list[DocProviderReceipt]
) -> StatusDimension:
    receipt = _provider_receipt(receipts, DocEvidenceProviderName.OWNERSHIP)
    if receipt is None:
        return _unknown_dimension(
            "Ownership coverage", "ownership", "ownership provider not configured for this index"
        )
    if receipt.availability != ProviderAvailability.AVAILABLE or not ownership_records:
        return _unknown_dimension(
            "Ownership coverage",
            "ownership",
            receipt.reason or "ownership provider produced no evidence",
        )
    return StatusDimension(
        name="Ownership coverage",
        state=StatusDimensionState.EVIDENCED,
        detail=f"{len(ownership_records)} ownership pattern(s) declared",
        provider="ownership",
        evidence=[
            StatusDimensionEvidence(
                description=f"{record.path_pattern} -> {', '.join(record.owners)}",
                location=record.source_path,
            )
            for record in ownership_records[:MAX_DIMENSION_EVIDENCE]
        ],
    )


def _latest_released_changelog_entry(repo_root: Path) -> tuple[str, str] | None:
    """Return ``(version, date)`` for the most recent *released* CHANGELOG entry.

    Skips a leading ``## [Unreleased]`` heading -- that section describes
    staged, not-yet-published changes, so it is never presented as release
    evidence.
    """
    changelog_path = repo_root / "CHANGELOG.md"
    if not changelog_path.is_file():
        return None
    try:
        text = changelog_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for match in _CHANGELOG_VERSION_PATTERN.finditer(text):
        version = match.group(1).strip()
        if version.lower() == "unreleased":
            continue
        date = (match.group(2) or "").strip()
        return version, date
    return None


def _release_dimension(repo_root: Path) -> StatusDimension:
    changelog_entry = _latest_released_changelog_entry(repo_root)
    ci_workflow = next(
        (candidate for candidate in _CI_WORKFLOW_CANDIDATES if (repo_root / candidate).is_file()),
        None,
    )
    if changelog_entry is None and ci_workflow is None:
        return _unknown_dimension(
            "Release & CI evidence",
            "release",
            "no released CHANGELOG.md entry and no pinned read-only CI workflow found",
        )
    evidence: list[StatusDimensionEvidence] = []
    detail_parts: list[str] = []
    if changelog_entry is not None:
        version, date = changelog_entry
        label = f"{version} ({date})" if date else version
        detail_parts.append(f"latest released version is {label}")
        evidence.append(
            StatusDimensionEvidence(description=f"CHANGELOG entry {label}", location="CHANGELOG.md")
        )
    if ci_workflow is not None:
        detail_parts.append("a pinned read-only CI workflow is present")
        evidence.append(
            StatusDimensionEvidence(
                description="pinned read-only CI workflow", location=ci_workflow
            )
        )
    return StatusDimension(
        name="Release & CI evidence",
        state=StatusDimensionState.EVIDENCED,
        detail="; ".join(detail_parts),
        provider="release",
        evidence=evidence,
    )


def build_status_card(source: str | Path) -> StatusCard:
    """Build the canonical dimensioned status card for *source*.

    Ensures a current index via ``index_repository`` (the same read-side
    contract every other analysis command uses). Every dimension is
    ``UNKNOWN`` unless its corresponding provider is configured on the
    index and produced real evidence -- there is no default-enabled
    dimension.
    """
    repo_root = Path(source).expanduser().resolve()
    repo_source = RepoSource(local_path=str(source))
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)

    store = index_repository(repo_source, config=config, index_config=index_config)
    try:
        doc_links = store.get_documentation_links()
        adr_records = store.get_documentation_adr_records()
        ownership_records = store.get_documentation_ownership_records()
        receipts = store.get_documentation_provider_receipts()
        source_revision = (
            store.get_metadata("commit_hash") or CacheManager.git_head(str(repo_root)) or ""
        )
    finally:
        store.close()

    dimensions = [
        _doc_link_dimension(doc_links, receipts),
        _adr_dimension(adr_records, receipts),
        _ownership_dimension(ownership_records, receipts),
        _release_dimension(repo_root),
    ]

    return StatusCard(
        source_identity=repo_source.url or repo_source.local_path or str(repo_root),
        revision=source_revision,
        generated_at=str(time.time()),
        dimensions=dimensions,
    )
