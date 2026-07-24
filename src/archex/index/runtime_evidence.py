"""Conditional runtime/coverage evidence collection (M7): dispatch configured providers.

Zero-cost when no provider names are requested (the default): no provider
module import happens beyond this module's own top-level imports, and no
provider runs. Enabling a provider by name runs it against the given
repository root and revision, and folds every non-``AVAILABLE`` outcome
(unavailable, partial, or revision-stale) into an explicit receipt rather
than a silent gap or a mismatched-revision record being silently applied.

Kept structurally separate from ``archex.index.semantic_evidence`` (M6): the
two conditional evidence channels are independently disableable and
independently rollback-able, so neither imports or depends on the other.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import TYPE_CHECKING

from archex.integrations.runtime.coverage_provider import CoverageXmlProvider
from archex.integrations.runtime.models import (
    ProviderAvailability,
    RuntimeEvidenceProviderName,
    RuntimeProviderReceipt,
)
from archex.integrations.runtime.profile_provider import RuntimeProfileProvider

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.runtime.models import CoverageFileEvidence, RuntimeProfileEvidence
    from archex.integrations.runtime.provider import (
        CoverageEvidenceProvider,
        RuntimeProfileEvidenceProvider,
    )

logger = logging.getLogger(__name__)

#: Provider names accepted by ``IndexConfig.runtime_evidence_providers``.
KNOWN_RUNTIME_EVIDENCE_PROVIDERS = frozenset({"coverage", "runtime_profile"})


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _collect_coverage(
    provider: CoverageEvidenceProvider, repo_root: Path, expected_revision: str
) -> tuple[list[CoverageFileEvidence], RuntimeProviderReceipt]:
    try:
        return provider.collect(repo_root, expected_revision=expected_revision)
    except Exception as exc:
        logger.warning(
            "runtime evidence provider 'coverage' raised during collect()", exc_info=True
        )
        return [], RuntimeProviderReceipt(
            provider=RuntimeEvidenceProviderName.COVERAGE,
            availability=ProviderAvailability.UNAVAILABLE,
            reason=f"provider raised {type(exc).__name__}: {exc}",
            expected_revision=expected_revision,
            collected_at=_now_iso(),
        )


def _collect_profile(
    provider: RuntimeProfileEvidenceProvider, repo_root: Path, expected_revision: str
) -> tuple[list[RuntimeProfileEvidence], RuntimeProviderReceipt]:
    try:
        return provider.collect(repo_root, expected_revision=expected_revision)
    except Exception as exc:
        logger.warning(
            "runtime evidence provider 'runtime_profile' raised during collect()", exc_info=True
        )
        return [], RuntimeProviderReceipt(
            provider=RuntimeEvidenceProviderName.RUNTIME_PROFILE,
            availability=ProviderAvailability.UNAVAILABLE,
            reason=f"provider raised {type(exc).__name__}: {exc}",
            expected_revision=expected_revision,
            collected_at=_now_iso(),
        )


def collect_runtime_evidence(
    repo_root: Path,
    provider_names: list[str],
    *,
    expected_revision: str,
    coverage_provider: CoverageEvidenceProvider | None = None,
    profile_provider: RuntimeProfileEvidenceProvider | None = None,
) -> tuple[
    list[CoverageFileEvidence],
    list[RuntimeProfileEvidence],
    list[RuntimeProviderReceipt],
]:
    """Run every provider named in *provider_names* against *repo_root*.

    Returns ``([], [], [])`` immediately when *provider_names* is empty --
    the default path adds no cost. *expected_revision* is the current index
    build's resolved git revision; each provider validates its own
    evidence's declared revision against it and reports ``STALE`` rather
    than applying mismatched evidence. *coverage_provider*/*profile_provider*
    let a caller inject an already-configured provider (for example one
    pointed at a non-default evidence directory); the corresponding stock
    provider is used when omitted.

    A provider must never raise for ordinary unavailability, but this is an
    external-tool boundary (a built-in provider's own bug, or a caller-
    injected custom provider), so every ``collect()`` call is defended here:
    an unexpected exception degrades that one provider to an explicit
    ``UNAVAILABLE`` receipt rather than aborting the entire index build.
    """
    if not provider_names:
        return [], [], []
    unknown = set(provider_names) - KNOWN_RUNTIME_EVIDENCE_PROVIDERS
    if unknown:
        raise ValueError(f"unknown runtime evidence providers: {sorted(unknown)}")

    coverage_evidence: list[CoverageFileEvidence] = []
    profile_evidence: list[RuntimeProfileEvidence] = []
    receipts: list[RuntimeProviderReceipt] = []

    if "coverage" in provider_names:
        resolved_coverage = coverage_provider or CoverageXmlProvider()
        coverage_evidence, coverage_receipt = _collect_coverage(
            resolved_coverage, repo_root, expected_revision
        )
        receipts.append(coverage_receipt)
    if "runtime_profile" in provider_names:
        resolved_profile = profile_provider or RuntimeProfileProvider()
        profile_evidence, profile_receipt = _collect_profile(
            resolved_profile, repo_root, expected_revision
        )
        receipts.append(profile_receipt)

    return coverage_evidence, profile_evidence, receipts
