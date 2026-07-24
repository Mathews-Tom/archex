"""ADR evidence provider: reads local architecture-decision-records already on disk.

Scans a bounded, conventional set of ADR directories (``docs/adr``,
``.docs/adr``, ``doc/adr``, ``adr``) for markdown files and reads each
file's title, declared status text, and referenced source paths. Status is
read verbatim from the document (for example ``"Accepted"`` or
``"Superseded"``) -- never inferred, normalized, or scored. A repository
with no ADR directory is a fully expected, honest ``UNAVAILABLE`` outcome,
not an error.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

from archex.integrations.docs._markdown import extract_markdown_links
from archex.integrations.docs.models import (
    AdrRecord,
    DocEvidenceProviderName,
    DocProviderReceipt,
    ProviderAvailability,
)

#: Conventional ADR directories, checked in this order relative to the
#: repository root.
_ADR_DIR_NAMES = ("docs/adr", ".docs/adr", "doc/adr", "adr")

#: Hard cap on ADR files scanned per run.
MAX_ADR_FILES = 200

_TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_STATUS_PATTERN = re.compile(r"^\s*(?:##\s*)?status\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LEADING_ID_PATTERN = re.compile(r"^(\d+)")


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _find_adr_dir(repo_root: Path) -> Path | None:
    for dirname in _ADR_DIR_NAMES:
        candidate = repo_root / dirname
        if candidate.is_dir() and any(candidate.glob("*.md")):
            return candidate
    return None


def _adr_id_for(path: Path, fallback_index: int) -> str:
    match = _LEADING_ID_PATTERN.match(path.stem)
    if match:
        return match.group(1)
    return path.stem or f"adr-{fallback_index}"


def _title_for(text: str, path: Path) -> str:
    match = _TITLE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.stem


def _status_for(text: str) -> str:
    match = _STATUS_PATTERN.search(text)
    if match:
        return match.group(1).strip().rstrip(".")
    return "unknown"


class AdrProvider:
    """Reads local architecture-decision-records from a conventional directory."""

    @property
    def name(self) -> DocEvidenceProviderName:
        return DocEvidenceProviderName.ADR

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        adr_dir = _find_adr_dir(repo_root)
        if adr_dir is None:
            checked = ", ".join(_ADR_DIR_NAMES)
            return DocProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"no ADR directory found at any of: {checked} (under {repo_root})",
                expected_revision=expected_revision,
                observed_revision=expected_revision,
                collected_at=_now_iso(),
            )
        files = sorted(adr_dir.glob("*.md"))[:MAX_ADR_FILES]
        return DocProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=expected_revision,
            sources_scanned=len(files),
            collected_at=_now_iso(),
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[AdrRecord], DocProviderReceipt]:
        probe_receipt = self.probe(repo_root, expected_revision=expected_revision)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt

        adr_dir = _find_adr_dir(repo_root)
        if adr_dir is None:
            # Directory removed between probe() and collect() (unlikely
            # filesystem race) -- handled explicitly rather than assumed.
            return [], DocProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"ADR directory disappeared during collection (under {repo_root})",
                expected_revision=expected_revision,
                observed_revision=expected_revision,
                collected_at=_now_iso(),
            )
        files = sorted(adr_dir.glob("*.md"))[:MAX_ADR_FILES]

        records: list[AdrRecord] = []
        for index, path in enumerate(files):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            doc_relative = path.relative_to(repo_root).as_posix()
            referenced = sorted({target for _, target in extract_markdown_links(path, repo_root)})
            records.append(
                AdrRecord(
                    adr_id=_adr_id_for(path, index),
                    title=_title_for(text, path),
                    status=_status_for(text),
                    doc_path=doc_relative,
                    referenced_paths=referenced,
                    revision=expected_revision,
                )
            )

        receipt = DocProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=expected_revision,
            sources_scanned=len(files),
            records_collected=len(records),
            collected_at=_now_iso(),
        )
        return records, receipt
