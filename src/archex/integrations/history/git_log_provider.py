"""Git-log evidence provider: reads local commit history already on disk.

Collects revision-bounded ``ChangeCard`` and ``TemporalCouplingObservation``
evidence from ``git log`` against the commit window ending at a caller-given
revision. Never contacts a remote service, never fetches an issue or pull
request record -- linked-reference extraction is a local regex match
against each commit's own subject line, never a lookup. Commits touching an
excessive number of files are excluded from temporal-coupling accounting
(they dilute co-change signal without adding real association evidence, the
same "dense commit" caveat as any co-change analysis).
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
from collections import Counter
from pathlib import Path

from archex.integrations.history.models import (
    ChangeCard,
    HistoryEvidenceProviderName,
    HistoryProviderReceipt,
    LinkedReference,
    ProviderAvailability,
    TemporalCouplingObservation,
)

#: STX/US control characters delimit git log records/fields unambiguously --
#: neither can appear in a commit subject or ISO date.
_RECORD_SEP = "\x02"
_FIELD_SEP = "\x1f"
_LOG_FORMAT = f"{_RECORD_SEP}%H{_FIELD_SEP}%s{_FIELD_SEP}%cI"

#: Commits touching more files than this are excluded from coupling
#: accounting -- a mass rename/formatting commit is not real co-change
#: evidence between any two of its files.
_MAX_FILES_PER_COUPLING_COMMIT = 30

#: Coupled pairs must co-occur at least this many times to be reported.
_MIN_COUPLING_COUNT = 2

#: Hard cap on emitted coupling pairs per run, sorted by strongest first --
#: bounds receipt size the same way other providers cap emitted records.
_MAX_COUPLING_PAIRS = 500

_REFERENCE_PATTERN = re.compile(r"(?:GH-|#)(\d+)\b")


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _extract_references(subject: str) -> list[LinkedReference]:
    references: list[LinkedReference] = []
    seen: set[str] = set()
    for match in _REFERENCE_PATTERN.finditer(subject):
        identifier = match.group(1)
        if identifier in seen:
            continue
        seen.add(identifier)
        references.append(LinkedReference(raw_text=match.group(0), identifier=identifier))
    return references


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    parts = lowered.split("/")
    if any(part in ("test", "tests", "__tests__") for part in parts[:-1]):
        return True
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name.endswith(".test.ts")


class GitLogHistoryProvider:
    """Reads revision-bounded local commit history via ``git log``."""

    @property
    def name(self) -> HistoryEvidenceProviderName:
        return HistoryEvidenceProviderName.GIT_LOG

    def probe(self, repo_root: Path, *, expected_revision: str) -> HistoryProviderReceipt:

        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{expected_revision}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return HistoryProviderReceipt(
                provider=self.name,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=(
                    f"{repo_root} is not a git repository, or revision "
                    f"{expected_revision!r} could not be resolved"
                ),
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            )
        return HistoryProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=result.stdout.strip(),
            collected_at=_now_iso(),
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str, max_commits: int
    ) -> tuple[list[ChangeCard], list[TemporalCouplingObservation], HistoryProviderReceipt]:

        probe_receipt = self.probe(repo_root, expected_revision=expected_revision)
        if probe_receipt.availability != ProviderAvailability.AVAILABLE:
            return [], [], probe_receipt

        result = subprocess.run(
            [
                "git",
                "log",
                expected_revision,
                f"-n{max_commits}",
                "--no-merges",
                f"--pretty=format:{_LOG_FORMAT}",
                "--name-only",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return (
                [],
                [],
                HistoryProviderReceipt(
                    provider=self.name,
                    availability=ProviderAvailability.UNAVAILABLE,
                    reason=f"git log failed: {result.stderr.strip() or 'unknown error'}",
                    expected_revision=expected_revision,
                    observed_revision=probe_receipt.observed_revision,
                    collected_at=_now_iso(),
                ),
            )

        change_cards: list[ChangeCard] = []
        coupling_counts: Counter[tuple[str, str]] = Counter()
        commits_considered = 0

        for raw_record in result.stdout.split(_RECORD_SEP):
            if not raw_record.strip():
                continue
            lines = raw_record.split("\n")
            header = lines[0]
            parts = header.split(_FIELD_SEP)
            if len(parts) != 3:
                continue
            commit_sha, subject, committed_at = parts
            changed_files = [line.strip() for line in lines[1:] if line.strip()]
            commits_considered += 1

            touched_test_files = [path for path in changed_files if _is_test_path(path)]
            change_cards.append(
                ChangeCard(
                    commit_sha=commit_sha,
                    commit_subject=subject,
                    committed_at=committed_at,
                    changed_files=changed_files,
                    touched_test_files=touched_test_files,
                    linked_references=_extract_references(subject),
                    revision=expected_revision,
                )
            )

            if len(changed_files) <= _MAX_FILES_PER_COUPLING_COMMIT:
                unique_files = sorted(set(changed_files))
                for i, file_a in enumerate(unique_files):
                    for file_b in unique_files[i + 1 :]:
                        coupling_counts[(file_a, file_b)] += 1

        coupling_observations = [
            TemporalCouplingObservation(
                file_a=file_a,
                file_b=file_b,
                co_change_count=count,
                window_commit_count=commits_considered,
                revision=expected_revision,
            )
            for (file_a, file_b), count in coupling_counts.most_common(_MAX_COUPLING_PAIRS)
            if count >= _MIN_COUPLING_COUNT
        ]

        receipt = HistoryProviderReceipt(
            provider=self.name,
            availability=ProviderAvailability.AVAILABLE,
            expected_revision=expected_revision,
            observed_revision=probe_receipt.observed_revision,
            window_commit_count=commits_considered,
            records_collected=len(change_cards) + len(coupling_observations),
            collected_at=_now_iso(),
        )
        return change_cards, coupling_observations, receipt
