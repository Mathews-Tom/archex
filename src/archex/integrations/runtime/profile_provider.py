"""Runtime-profile evidence provider: reads a folded-stack sample file.

Reads revision-bound "folded stack" runtime samples -- the widely used
flamegraph input format (``frame;frame;...;frame count`` per line, root
frame first) -- from a previously collected profiling run plus a manifest
recording the git revision it was collected against. Profiling itself never
runs automatically: an operator runs a profiler (for example
``python -m cProfile``, folded into this format by a companion script) and
writes the evidence directory; this provider only reads it.

Each frame must be ``<repo-relative file path>:<qualified symbol name>`` so
a frame can always be attributed to an indexed file/symbol without guessing
at an external tool's own naming convention. A malformed frame drops the
whole sample rather than being partially applied.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from archex.integrations.runtime.models import (
    ProviderAvailability,
    RuntimeEvidenceProviderName,
    RuntimeProfileEvidence,
    RuntimeProviderReceipt,
    RuntimeStackSample,
)

_MANIFEST_FILENAME = "manifest.json"
_REPORT_FILENAME = "profile.folded"
_DEFAULT_EVIDENCE_DIRNAME = ".archex/runtime-evidence/profile"


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _normalize_frame(frame: str, repo_root_resolved: Path, repo_root: Path) -> str | None:
    file_part, sep, symbol = frame.partition(":")
    if not sep or not file_part.strip() or not symbol.strip():
        return None
    normalized = file_part.replace("\\", "/").strip().lstrip("/")
    resolved = (repo_root / normalized).resolve()
    try:
        resolved.relative_to(repo_root_resolved)
    except ValueError:
        return None  # never attach evidence for a path outside the repository root
    return f"{normalized}:{symbol.strip()}"


def _parse_folded_line(line: str, repo_root: Path) -> RuntimeStackSample | None:
    stripped = line.strip()
    if not stripped:
        return None
    stack_part, _, count_part = stripped.rpartition(" ")
    if not stack_part or not count_part.isdigit():
        return None
    sample_count = int(count_part)
    if sample_count < 1:
        return None
    repo_root_resolved = repo_root.resolve()
    frames: list[str] = []
    for raw_frame in stack_part.split(";"):
        normalized = _normalize_frame(raw_frame, repo_root_resolved, repo_root)
        if normalized is None:
            return None
        frames.append(normalized)
    if not frames:
        return None
    return RuntimeStackSample(frames=tuple(frames), sample_count=sample_count)


class RuntimeProfileProvider:
    """Reads pre-collected folded-stack runtime-profile evidence."""

    def __init__(self, evidence_dir: Path | str | None = None) -> None:
        self._evidence_dir = Path(evidence_dir) if evidence_dir is not None else None

    @property
    def name(self) -> RuntimeEvidenceProviderName:
        return RuntimeEvidenceProviderName.RUNTIME_PROFILE

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
                reason=f"no runtime-profile evidence found at {evidence_dir}",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"could not read runtime-profile manifest at {manifest_path}: {exc}",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        observed_revision = str(manifest.get("revision") or "") or None
        if not observed_revision:
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"runtime-profile manifest at {manifest_path} has no revision",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        if observed_revision != expected_revision:
            return RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.STALE,
                reason=(
                    f"runtime-profile evidence was collected at revision "
                    f"{observed_revision[:12]}, current revision is {expected_revision[:12]}"
                ),
                expected_revision=expected_revision,
                observed_revision=observed_revision,
                collected_at=_now_iso(),
            )
        tool_version_raw = manifest.get("tool_version")
        return RuntimeProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            tool_name=str(manifest.get("tool") or "cProfile"),
            tool_version=str(tool_version_raw) if tool_version_raw else None,
            expected_revision=expected_revision,
            observed_revision=observed_revision,
            collected_at=_now_iso(),
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[RuntimeProfileEvidence], RuntimeProviderReceipt]:
        probe_receipt = self.probe(repo_root, expected_revision=expected_revision)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], probe_receipt

        evidence_dir = self._resolve_evidence_dir(repo_root)
        report_path = evidence_dir / _REPORT_FILENAME
        try:
            raw_lines = report_path.read_text().splitlines()
        except OSError as exc:
            return [], RuntimeProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.STALE,
                reason=f"could not read runtime-profile report at {report_path}: {exc}",
                expected_revision=expected_revision,
                observed_revision=probe_receipt.observed_revision,
                collected_at=_now_iso(),
            )

        samples: list[RuntimeStackSample] = []
        dropped = 0
        for raw_line in raw_lines:
            sample = _parse_folded_line(raw_line, repo_root)
            if sample is None:
                if raw_line.strip():
                    dropped += 1
                continue
            samples.append(sample)

        total_samples = sum(sample.sample_count for sample in samples)
        evidence = [
            RuntimeProfileEvidence(
                samples=samples,
                total_samples=total_samples,
                revision=expected_revision,
            )
        ]
        reason = f"{dropped} malformed frame line(s) dropped" if dropped else ""
        receipt = RuntimeProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            reason=reason,
            tool_name=probe_receipt.tool_name,
            tool_version=probe_receipt.tool_version,
            expected_revision=expected_revision,
            observed_revision=probe_receipt.observed_revision,
            records_collected=len(samples),
            collected_at=_now_iso(),
        )
        return evidence, receipt
