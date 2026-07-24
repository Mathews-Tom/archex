"""Conditional documentation-graph evidence collection (M9): dispatch configured providers.

Zero-cost when no provider names are requested (the default): no provider
module import happens beyond this module's own top-level imports, and no
provider runs. Enabling a provider by name runs it against the given
repository root and revision, and folds every non-``AVAILABLE`` outcome
(unavailable or partial) into an explicit receipt rather than a silent gap.

Kept structurally separate from ``archex.index.semantic_evidence`` (M6),
``archex.index.runtime_evidence`` (M7), and ``archex.index.history_evidence``
(M8): every conditional evidence channel is independently disableable and
independently rollback-able, so none of them import or depend on each
other. Collected documentation, ADR, and ownership evidence is never
folded into ``DependencyGraph`` -- it is stored and surfaced separately
from code-dependency edges.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import TYPE_CHECKING

from archex.integrations.docs.adr_provider import AdrProvider
from archex.integrations.docs.doc_link_provider import DocLinkProvider
from archex.integrations.docs.models import (
    DocEvidenceProviderName,
    DocProviderReceipt,
    ProviderAvailability,
)
from archex.integrations.docs.ownership_provider import OwnershipProvider

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.docs.models import AdrRecord, DocumentationLink, OwnershipRecord
    from archex.integrations.docs.provider import (
        AdrEvidenceProvider,
        DocLinkEvidenceProvider,
        OwnershipEvidenceProvider,
    )

logger = logging.getLogger(__name__)

#: Provider names accepted by ``IndexConfig.documentation_evidence_providers``.
KNOWN_DOCUMENTATION_EVIDENCE_PROVIDERS = frozenset({"doc_link", "adr", "ownership"})


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _collect_doc_link(
    provider: DocLinkEvidenceProvider, repo_root: Path, expected_revision: str
) -> tuple[list[DocumentationLink], DocProviderReceipt]:
    try:
        return provider.collect(repo_root, expected_revision=expected_revision)
    except Exception as exc:
        logger.warning(
            "documentation evidence provider 'doc_link' raised during collect()", exc_info=True
        )
        return [], DocProviderReceipt(
            provider=DocEvidenceProviderName.DOC_LINK,
            availability=ProviderAvailability.UNAVAILABLE,
            reason=f"provider raised {type(exc).__name__}: {exc}",
            expected_revision=expected_revision,
            collected_at=_now_iso(),
        )


def _collect_adr(
    provider: AdrEvidenceProvider, repo_root: Path, expected_revision: str
) -> tuple[list[AdrRecord], DocProviderReceipt]:
    try:
        return provider.collect(repo_root, expected_revision=expected_revision)
    except Exception as exc:
        logger.warning(
            "documentation evidence provider 'adr' raised during collect()", exc_info=True
        )
        return [], DocProviderReceipt(
            provider=DocEvidenceProviderName.ADR,
            availability=ProviderAvailability.UNAVAILABLE,
            reason=f"provider raised {type(exc).__name__}: {exc}",
            expected_revision=expected_revision,
            collected_at=_now_iso(),
        )


def _collect_ownership(
    provider: OwnershipEvidenceProvider, repo_root: Path, expected_revision: str
) -> tuple[list[OwnershipRecord], DocProviderReceipt]:
    try:
        return provider.collect(repo_root, expected_revision=expected_revision)
    except Exception as exc:
        logger.warning(
            "documentation evidence provider 'ownership' raised during collect()", exc_info=True
        )
        return [], DocProviderReceipt(
            provider=DocEvidenceProviderName.OWNERSHIP,
            availability=ProviderAvailability.UNAVAILABLE,
            reason=f"provider raised {type(exc).__name__}: {exc}",
            expected_revision=expected_revision,
            collected_at=_now_iso(),
        )


def collect_documentation_evidence(
    repo_root: Path,
    provider_names: list[str],
    *,
    expected_revision: str,
    doc_link_provider: DocLinkEvidenceProvider | None = None,
    adr_provider: AdrEvidenceProvider | None = None,
    ownership_provider: OwnershipEvidenceProvider | None = None,
) -> tuple[
    list[DocumentationLink],
    list[AdrRecord],
    list[OwnershipRecord],
    list[DocProviderReceipt],
]:
    """Run every provider named in *provider_names* against *repo_root*.

    Returns ``([], [], [], [])`` immediately when *provider_names* is empty
    -- the default path adds no cost. *expected_revision* labels every
    collected record; every provider reads only the current working tree,
    so no staleness comparison against a separately-collected manifest
    applies here (unlike M7's coverage/profile evidence or M8's operator
    rationale). *doc_link_provider*/*adr_provider*/*ownership_provider* let
    a caller inject an already-configured provider; the corresponding
    stock provider is used when omitted.

    A provider must never raise for ordinary unavailability, but this is an
    external boundary (a built-in provider's own bug, or a caller-injected
    custom provider), so every ``collect()`` call is defended here: an
    unexpected exception degrades that one provider to an explicit
    ``UNAVAILABLE`` receipt rather than aborting the entire index build.
    """
    if not provider_names:
        return [], [], [], []
    unknown = set(provider_names) - KNOWN_DOCUMENTATION_EVIDENCE_PROVIDERS
    if unknown:
        raise ValueError(f"unknown documentation evidence providers: {sorted(unknown)}")

    doc_links: list[DocumentationLink] = []
    adr_records: list[AdrRecord] = []
    ownership_records: list[OwnershipRecord] = []
    receipts: list[DocProviderReceipt] = []

    if "doc_link" in provider_names:
        resolved_doc_link = doc_link_provider or DocLinkProvider()
        doc_links, doc_link_receipt = _collect_doc_link(
            resolved_doc_link, repo_root, expected_revision
        )
        receipts.append(doc_link_receipt)
    if "adr" in provider_names:
        resolved_adr = adr_provider or AdrProvider()
        adr_records, adr_receipt = _collect_adr(resolved_adr, repo_root, expected_revision)
        receipts.append(adr_receipt)
    if "ownership" in provider_names:
        resolved_ownership = ownership_provider or OwnershipProvider()
        ownership_records, ownership_receipt = _collect_ownership(
            resolved_ownership, repo_root, expected_revision
        )
        receipts.append(ownership_receipt)

    return doc_links, adr_records, ownership_records, receipts
