"""Bounded, read-only diff-review delta: a compact summary of AnalysisArtifactV1.

Purpose-built for CI: small and deterministic enough to post as a job
summary or PR-comment-sized artifact without exceeding a log budget, while
the full `AnalysisArtifactV1` JSON remains available as the canonical,
complete record. Derived entirely from an already-built artifact -- no new
analysis, no state mutation, no remote fetch.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from archex.report.artifact import AnalysisArtifactV1

MAX_DELTA_RISK_CANDIDATES = 10


class ReportDelta(BaseModel):
    """A bounded summary of one AnalysisArtifactV1 build."""

    schema_version: str
    archex_version: str
    source_identity: str
    source_revision: str
    base_ref: str
    base_resolved_sha: str
    freshness: str
    completeness: str
    confidence: str
    risk_level: str
    changed_files_total: int
    symbol_candidates_total: int
    high_risk_symbol_count: int
    affected_interfaces_total: int
    test_candidates_total: int
    unsupported_files_total: int
    top_risk_candidates: list[str] = []
    generated_at: str

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "## Diff Review Delta",
            "",
            f"- **Source:** `{self.source_identity}` @ `{self.source_revision}`",
            f"- **Base:** `{self.base_ref}` (`{self.base_resolved_sha or 'unresolved'}`)",
            f"- **Freshness / Completeness / Confidence:** "
            f"`{self.freshness}` / `{self.completeness}` / `{self.confidence}`",
            f"- **Risk:** `{self.risk_level}`",
            f"- **Changed files:** {self.changed_files_total}",
            f"- **Symbol candidates:** {self.symbol_candidates_total} "
            f"({self.high_risk_symbol_count} high risk)",
            f"- **Affected interfaces:** {self.affected_interfaces_total}",
            f"- **Test candidates:** {self.test_candidates_total}",
            f"- **Unsupported files:** {self.unsupported_files_total}",
        ]
        if self.top_risk_candidates:
            lines.append("")
            lines.append("**Top risk candidates:**")
            lines.extend(f"- `{candidate}`" for candidate in self.top_risk_candidates)
        return "\n".join(lines).rstrip() + "\n"


def build_report_delta(artifact: AnalysisArtifactV1) -> ReportDelta:
    """Compact an AnalysisArtifactV1 into a bounded, CI-log-sized delta summary."""
    # symbol_candidates is already bounded by MAX_SYMBOL_CANDIDATES on the
    # source artifact; high_risk_symbol_count reflects that same window, not
    # necessarily every high-risk symbol in an unbounded diff.
    high_risk = [c for c in artifact.diff.symbol_candidates if c.risk_level == "high"]
    top_candidates = [
        f"{c.risk_level}: {c.qualified_name or c.symbol_name or '<unnamed>'} ({c.handle})"
        for c in sorted(high_risk, key=lambda c: (c.file_path, c.start_line))[
            :MAX_DELTA_RISK_CANDIDATES
        ]
    ]
    return ReportDelta(
        schema_version=artifact.schema_version.value,
        archex_version=artifact.archex_version,
        source_identity=artifact.source_identity,
        source_revision=artifact.source_revision,
        base_ref=artifact.diff.base_ref,
        base_resolved_sha=artifact.diff.base_resolved_sha,
        freshness=artifact.freshness.value,
        completeness=artifact.completeness.value,
        confidence=artifact.confidence.value,
        risk_level=artifact.diff.risk_level.value,
        changed_files_total=artifact.diff.changed_files_total,
        symbol_candidates_total=artifact.diff.symbol_candidates_total,
        high_risk_symbol_count=len(high_risk),
        affected_interfaces_total=artifact.diff.affected_interfaces_total,
        test_candidates_total=artifact.diff.test_candidates_total,
        unsupported_files_total=artifact.diff.unsupported_files_total,
        top_risk_candidates=top_candidates,
        generated_at=artifact.generated_at,
    )
