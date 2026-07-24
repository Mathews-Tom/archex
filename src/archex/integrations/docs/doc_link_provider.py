"""Doc-link evidence provider: reads local markdown documentation already on disk.

Scans a bounded, conventional set of markdown roots (``README.md`` at the
repository root, plus ``docs/`` and ``.docs/`` trees) for markdown links
that resolve to a real file under the repository root. Never fetches a
remote page, never resolves a reference-style link, and never records a
link whose target does not exist locally -- an aspirational or broken link
is not documentation evidence.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from archex.integrations.docs._markdown import extract_markdown_links
from archex.integrations.docs.models import (
    DocEvidenceProviderName,
    DocProviderReceipt,
    DocumentationLink,
    ProviderAvailability,
)

#: Conventional markdown documentation roots, checked in this order.
#: ``README.md`` is a single file; the other two are directory trees.
_DOC_ROOT_NAMES = ("README.md", "docs", ".docs")

#: Hard cap on markdown files scanned per run -- bounds cost and receipt
#: size the same way other providers cap emitted records.
MAX_DOC_FILES = 200

#: Hard cap on emitted documentation links per run.
MAX_DOCUMENTATION_LINKS = 1000


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _discover_markdown_files(repo_root: Path) -> list[Path]:
    readme = repo_root / "README.md"
    files: list[Path] = [readme] if readme.is_file() else []
    for dirname in ("docs", ".docs"):
        root = repo_root / dirname
        if root.is_dir():
            files.extend(sorted(p for p in root.rglob("*.md") if p.is_file()))
    return files[:MAX_DOC_FILES]


class DocLinkProvider:
    """Reads markdown documentation links from the working tree already on disk."""

    @property
    def name(self) -> DocEvidenceProviderName:
        return DocEvidenceProviderName.DOC_LINK

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        files = _discover_markdown_files(repo_root)
        if not files:
            return DocProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=(
                    f"no markdown documentation found under {repo_root} "
                    f"(checked {', '.join(_DOC_ROOT_NAMES)})"
                ),
                expected_revision=expected_revision,
                observed_revision=expected_revision,
                collected_at=_now_iso(),
            )
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
    ) -> tuple[list[DocumentationLink], DocProviderReceipt]:
        probe_receipt = self.probe(repo_root, expected_revision=expected_revision)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt

        files = _discover_markdown_files(repo_root)
        links: list[DocumentationLink] = []
        for doc_path in files:
            doc_relative = doc_path.relative_to(repo_root).as_posix()
            for link_text, target_path in extract_markdown_links(doc_path, repo_root):
                if target_path == doc_relative:
                    continue
                links.append(
                    DocumentationLink(
                        doc_path=doc_relative,
                        target_path=target_path,
                        link_text=link_text,
                        revision=expected_revision,
                    )
                )
                if len(links) >= MAX_DOCUMENTATION_LINKS:
                    break
            if len(links) >= MAX_DOCUMENTATION_LINKS:
                break

        receipt = DocProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=expected_revision,
            sources_scanned=len(files),
            records_collected=len(links),
            collected_at=_now_iso(),
        )
        return links, receipt
