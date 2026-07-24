"""Conditional semantic-evidence collection (M6): dispatch configured SCIP/LSP providers.

Zero-cost when ``IndexConfig.semantic_evidence_providers`` is empty (the
default): no provider module is imported, no provider runs, and the syntax
graph is completely unaffected. Enabling a provider by name runs it and
folds every non-``AVAILABLE`` outcome into an explicit receipt rather than a
silent gap.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import TYPE_CHECKING

from archex.integrations.semantic.lsp_provider import LspEvidenceProvider
from archex.integrations.semantic.models import ProviderAvailability, SemanticProviderReceipt
from archex.integrations.semantic.scip_provider import ScipEvidenceProvider

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.semantic.models import SemanticEdgeEvidence
    from archex.integrations.semantic.provider import SemanticEvidenceProvider
    from archex.models import IndexConfig, ParsedFile

logger = logging.getLogger(__name__)


def _default_provider(name: str) -> SemanticEvidenceProvider:
    if name == "scip":
        return ScipEvidenceProvider()
    if name == "lsp":
        return LspEvidenceProvider()
    # Unreachable via IndexConfig, whose validator restricts values to
    # _KNOWN_PROVIDERS; guarded here for direct callers of this function.
    raise ValueError(f"unknown semantic evidence provider: {name!r}")


def collect_semantic_evidence(
    parsed_files: list[ParsedFile],
    repo_root: Path,
    index_config: IndexConfig,
    *,
    providers: dict[str, SemanticEvidenceProvider] | None = None,
) -> tuple[list[SemanticEdgeEvidence], list[SemanticProviderReceipt]]:
    """Run every provider named in ``index_config.semantic_evidence_providers``.

    Returns ``([], [])`` immediately when no provider is configured — the
    default path adds no cost and touches no optional dependency. ``providers``
    lets a caller inject an already-configured provider (for example an
    ``LspEvidenceProvider`` bound to a live ``lsp_client.Client``); entries not
    supplied fall back to a stock provider constructed with default settings.

    A provider must never raise for ordinary unavailability, but this is an
    external-tool boundary (a built-in provider's own bug, or a caller-
    injected custom provider), so every ``collect()`` call is defended here:
    an unexpected exception degrades that one provider to an explicit
    ``UNAVAILABLE`` receipt rather than aborting the entire index build.
    """
    if not index_config.semantic_evidence_providers:
        return [], []

    resolved = providers or {}
    evidence: list[SemanticEdgeEvidence] = []
    receipts: list[SemanticProviderReceipt] = []
    for name in index_config.semantic_evidence_providers:
        provider = resolved.get(name) or _default_provider(name)
        try:
            item_evidence, receipt = provider.collect(parsed_files, repo_root)
        except Exception as exc:
            logger.warning(
                "semantic evidence provider %r raised during collect()", name, exc_info=True
            )
            receipts.append(
                SemanticProviderReceipt(
                    provider=provider.name,
                    availability=ProviderAvailability.UNAVAILABLE,
                    reason=f"provider raised {type(exc).__name__}: {exc}",
                    collected_at=_dt.datetime.now(tz=_dt.UTC).isoformat(),
                )
            )
            continue
        evidence.extend(item_evidence)
        receipts.append(receipt)
    return evidence, receipts
