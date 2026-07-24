"""Coverage evidence provider: reads a Cobertura-format ``coverage.xml`` report.

Reads line-coverage evidence from a previously generated coverage report --
the same Cobertura XML shape ``coverage xml`` / ``pytest --cov-report=xml``
already produces -- plus a manifest recording the git revision the report
was collected against. Coverage collection itself never runs automatically:
an operator (or a CI job) runs the test suite with coverage enabled and
writes the evidence directory; this provider only reads it.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from archex.integrations.runtime.models import (
    CoverageFileEvidence,
    CoverageLineRecord,
    ProviderAvailability,
    RuntimeEvidenceProviderName,
    RuntimeProviderReceipt,
)

#: Evidence directory layout, relative to the repository root: a manifest
#: recording provenance plus the Cobertura report itself. Mirrors the
#: existing ``.archex/`` local-workspace-state convention already used for
#: benchmark baselines.
_MANIFEST_FILENAME = "manifest.json"
_REPORT_FILENAME = "coverage.xml"
_DEFAULT_EVIDENCE_DIRNAME = ".archex/runtime-evidence/coverage"


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


class CoverageXmlProvider:
    """Reads pre-collected Cobertura ``coverage.xml`` line-coverage evidence."""

    def __init__(self, evidence_dir: Path | str | None = None) -> None:
        self._evidence_dir = Path(evidence_dir) if evidence_dir is not None else None

    @property
    def name(self) -> RuntimeEvidenceProviderName:
        return RuntimeEvidenceProviderName.COVERAGE

    def _resolve_evidence_dir(self, repo_root: Path) -> Path:
        if self._evidence_dir is not None:
            return self._evidence_dir
        return repo_root / _DEFAULT_EVIDENCE_DIRNAME

    def probe(self, repo_root: Path, *, expected_revision: str) -> RuntimeProviderReceipt:
        evidence_dir = self._resolve_evidence_dir(repo_root)
        manifest_path = evidence_dir / _MANIFEST_FILENAME
        report_path = evidence_dir / _REPORT_FILENAME
        if not manifest_path.is_file() or not report_path.is_file():
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"no coverage evidence found at {evidence_dir}",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"could not read coverage manifest at {manifest_path}: {exc}",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        observed_revision = str(manifest.get("revision") or "") or None
        if not observed_revision:
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"coverage manifest at {manifest_path} has no revision",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        if observed_revision != expected_revision:
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.STALE,
                reason=(
                    f"coverage evidence was collected at revision {observed_revision[:12]}, "
                    f"current revision is {expected_revision[:12]}"
                ),
                expected_revision=expected_revision,
                observed_revision=observed_revision,
                collected_at=_now_iso(),
            )
        tool_version_raw = manifest.get("tool_version")
        return RuntimeProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            tool_name=str(manifest.get("tool") or "coverage.py"),
            tool_version=str(tool_version_raw) if tool_version_raw else None,
            expected_revision=expected_revision,
            observed_revision=observed_revision,
            collected_at=_now_iso(),
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[CoverageFileEvidence], RuntimeProviderReceipt]:
        probe_receipt = self.probe(repo_root, expected_revision=expected_revision)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt

        evidence_dir = self._resolve_evidence_dir(repo_root)
        report_path = evidence_dir / _REPORT_FILENAME
        try:
            # noqa comment kept short: local, operator-collected evidence, never untrusted network XML.
            tree = ET.parse(report_path)  # noqa: S314
        except ET.ParseError as exc:
            return [], RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.STALE,
                reason=f"could not parse coverage report at {report_path}: {exc}",
                expected_revision=expected_revision,
                observed_revision=probe_receipt.observed_revision,
                collected_at=_now_iso(),
            )

        repo_root_resolved = repo_root.resolve()
        records: list[CoverageFileEvidence] = []
        for class_elem in tree.getroot().iter("class"):
            filename = class_elem.get("filename")
            if not filename:
                continue
            normalized = filename.replace("\\", "/").strip().lstrip("/")
            resolved = (repo_root / normalized).resolve()
            try:
                resolved.relative_to(repo_root_resolved)
            except ValueError:
                continue  # never attach evidence for a path outside the repository root
            lines: list[CoverageLineRecord] = []
            lines_elem = class_elem.find("lines")
            if lines_elem is not None:
                for line_elem in lines_elem.findall("line"):
                    number = line_elem.get("number")
                    hits = line_elem.get("hits")
                    if number is None or hits is None:
                        continue
                    lines.append(CoverageLineRecord(line=int(number), hits=int(hits)))
            line_rate_raw = class_elem.get("line-rate")
            line_rate = (
                min(1.0, max(0.0, float(line_rate_raw))) if line_rate_raw is not None else 0.0
            )
            records.append(
                CoverageFileEvidence(
                    file_path=normalized,
                    lines=lines,
                    line_rate=line_rate,
                    revision=expected_revision,
                )
            )

        receipt = RuntimeProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            tool_name=probe_receipt.tool_name,
            tool_version=probe_receipt.tool_version,
            expected_revision=expected_revision,
            observed_revision=probe_receipt.observed_revision,
            records_collected=len(records),
            collected_at=_now_iso(),
        )
        return records, receipt


def current_repo_revision(repo_root: Path) -> str | None:
    """Resolve the current git HEAD of *repo_root*, or ``None`` if it cannot be resolved."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
