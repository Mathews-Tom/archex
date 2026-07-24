"""Ownership evidence provider: reads a local CODEOWNERS-style manifest already on disk.

Reads a conventional CODEOWNERS file (``.github/CODEOWNERS``, ``CODEOWNERS``,
or ``docs/CODEOWNERS``, checked in that order -- GitHub's own precedence) and
parses ``pattern owner1 owner2 ...`` lines into ``OwnershipRecord``s. Owners
are recorded verbatim from the file; archex never infers ownership from
commit authorship or any other heuristic. A "no owner" override line (a
bare pattern with no owners, used by GitHub to unset inherited ownership)
is skipped -- it asserts an absence, not a positive ownership claim this
provider can carry as evidence.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from archex.integrations.docs.models import (
    DocEvidenceProviderName,
    DocProviderReceipt,
    OwnershipRecord,
    ProviderAvailability,
)

#: Conventional CODEOWNERS locations, checked in GitHub's own precedence
#: order, relative to the repository root.
_CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")

#: Hard cap on ownership lines parsed per run.
MAX_OWNERSHIP_RECORDS = 500


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _find_codeowners(repo_root: Path) -> Path | None:
    for relative in _CODEOWNERS_PATHS:
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return None


class OwnershipProvider:
    """Reads local CODEOWNERS-style ownership records from the working tree."""

    @property
    def name(self) -> DocEvidenceProviderName:
        return DocEvidenceProviderName.OWNERSHIP

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        codeowners = _find_codeowners(repo_root)
        if codeowners is None:
            checked = ", ".join(_CODEOWNERS_PATHS)
            return DocProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"no CODEOWNERS file found at any of: {checked} (under {repo_root})",
                expected_revision=expected_revision,
                observed_revision=expected_revision,
                collected_at=_now_iso(),
            )
        return DocProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=expected_revision,
            sources_scanned=1,
            collected_at=_now_iso(),
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[OwnershipRecord], DocProviderReceipt]:
        probe_receipt = self.probe(repo_root, expected_revision=expected_revision)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt

        codeowners = _find_codeowners(repo_root)
        if codeowners is None:
            return [], DocProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"CODEOWNERS file disappeared during collection (under {repo_root})",
                expected_revision=expected_revision,
                observed_revision=expected_revision,
                collected_at=_now_iso(),
            )

        try:
            text = codeowners.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return [], DocProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"could not read {codeowners}: {exc}",
                expected_revision=expected_revision,
                observed_revision=expected_revision,
                collected_at=_now_iso(),
            )

        source_relative = codeowners.relative_to(repo_root).as_posix()
        records: list[OwnershipRecord] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                continue  # bare pattern with no owners: an override, not evidence
            pattern, owners = tokens[0], tokens[1:]
            records.append(
                OwnershipRecord(
                    path_pattern=pattern,
                    owners=owners,
                    source_path=source_relative,
                    revision=expected_revision,
                )
            )
            if len(records) >= MAX_OWNERSHIP_RECORDS:
                break

        receipt = DocProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=expected_revision,
            sources_scanned=1,
            records_collected=len(records),
            collected_at=_now_iso(),
        )
        return records, receipt
