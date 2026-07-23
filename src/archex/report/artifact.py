"""AnalysisArtifactV1: the canonical, read-only diff-review artifact.

`build_analysis_artifact` is the single place that projects the existing
index/graph/impact/status surfaces into one versioned artifact. It performs
no parsing of its own, no edge construction, no reranking, and no security
inference -- it reuses `archex.impact`'s already-reviewed deterministic risk
classifier verbatim and adds identity, provenance, freshness, completeness,
confidence, evidence, exclusion, and unknown-count fields so every renderer
(JSON, Markdown, static HTML) can project the same semantics without
reinterpreting them.

Redaction is structural, not a toggle: the artifact never stores raw source
text, only path/line ranges, symbol identifiers, and stable handles. There is
therefore nothing for a renderer to redact -- `redaction_mode` documents that
guarantee rather than gating it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import subprocess
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from archex import __version__
from archex.api import index_repository
from archex.cache import CacheManager
from archex.config import load_config, load_index_config
from archex.impact import (
    ImpactError,
    ImpactFileChange,
    ImpactRiskLevel,
    analyze_diff_impact,
    git_changed_files,
    git_diff_hunks,
    resolve_diff_symbols,
)
from archex.languages import EXTENSION_LANGUAGE_MAP
from archex.models import ContextCompletenessStatus, ContextFreshness, RepoSource
from archex.scout import chunk_handle, file_handle, symbol_handle
from archex.serve.generation import read_generation_id

if TYPE_CHECKING:
    from archex.impact import ImpactReport, SymbolImpact
    from archex.models import CodeChunk

REPORT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_REPORT_SCHEMA_MAJOR = 1

# Bounds applied while building the artifact so every producer -- CLI, CI
# example, or a future MCP tool -- emits a size-predictable artifact. Each
# bounded list carries a companion `*_total` field recording the true count.
MAX_CHANGED_FILES = 200
MAX_SYMBOL_CANDIDATES = 150
MAX_INTERFACE_CANDIDATES = 100
MAX_TEST_CANDIDATES = 100
MAX_UNSUPPORTED_FILES = 100
MAX_EVIDENCE_LOCATIONS = 100

_PARSER_DISTRIBUTIONS: dict[str, str] = {
    "python": "tree-sitter-python",
    "javascript": "tree-sitter-javascript",
    "typescript": "tree-sitter-typescript",
    "tsx": "tree-sitter-typescript",
    "go": "tree-sitter-go",
    "rust": "tree-sitter-rust",
    "java": "tree-sitter-java",
    "kotlin": "tree-sitter-kotlin",
    "csharp": "tree-sitter-c-sharp",
}
_FALLBACK_PARSER_DISTRIBUTION = "tree-sitter-language-pack"


class ReportArtifactError(ValueError):
    """Raised when a report artifact cannot be built, loaded, or is malformed."""


class AnalysisConfidence(StrEnum):
    """Deterministic confidence in a finding, derived from evidence presence only."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class RedactionMode(StrEnum):
    """The artifact never carries raw source text; `REDACTED` is the only mode."""

    REDACTED = "redacted"


class ReportSchemaVersion(BaseModel):
    value: str = REPORT_SCHEMA_VERSION

    @property
    def major(self) -> int:
        return int(self.value.split(".", maxsplit=1)[0])


class EvidenceLocation(BaseModel):
    """A structural evidence pointer: where a finding's supporting data lives."""

    path: str
    start_line: int | None = None
    end_line: int | None = None
    handle: str | None = None
    description: str


class DiffHunk(BaseModel):
    start_line: int
    end_line: int


class DiffFileChange(BaseModel):
    path: str
    status: str
    old_path: str | None = None
    handle: str
    hunks: list[DiffHunk] = []


class SymbolCandidate(BaseModel):
    """A structural risk candidate for one symbol touched by the diff's hunks.

    Labeled a *candidate*: the risk level and confidence come from
    deterministic graph signals (fan-in, public-interface membership,
    cross-module reach), not a claim that the symbol will break.
    """

    handle: str
    file_path: str
    symbol_name: str | None = None
    qualified_name: str | None = None
    symbol_kind: str | None = None
    start_line: int
    end_line: int
    risk_level: str
    confidence: AnalysisConfidence
    signals: list[str] = []
    evidence: list[EvidenceLocation] = []


class InterfaceCandidate(BaseModel):
    path: str
    handle: str
    confidence: AnalysisConfidence = AnalysisConfidence.HIGH
    evidence: list[EvidenceLocation] = []


class TestCandidate(BaseModel):
    path: str
    handle: str
    reason: str
    confidence: AnalysisConfidence = AnalysisConfidence.HIGH
    evidence: list[EvidenceLocation] = []


class UnsupportedFile(BaseModel):
    path: str
    reason: Literal["unmapped_from_index"] = "unmapped_from_index"


class DiffAnalysis(BaseModel):
    """The diff-scoped payload: what changed and its deterministic structural impact."""

    base_ref: str
    base_resolved_sha: str = ""
    head_ref: str = ""

    changed_files: list[DiffFileChange] = []
    changed_files_total: int = 0

    symbol_candidates: list[SymbolCandidate] = []
    symbol_candidates_total: int = 0

    affected_interfaces: list[InterfaceCandidate] = []
    affected_interfaces_total: int = 0

    test_candidates: list[TestCandidate] = []
    test_candidates_total: int = 0

    unsupported_files: list[UnsupportedFile] = []
    unsupported_files_total: int = 0

    risk_level: ImpactRiskLevel = ImpactRiskLevel.LOW
    risk_reasons: list[str] = []


class AnalysisArtifactV1(BaseModel):
    """The canonical, versioned diff-review artifact every renderer projects.

    A read-only projection over the verified index, dependency graph, and
    deterministic impact classifier: it never parses source, constructs
    graph edges, reranks retrieval, infers security findings, mutates
    project state, or fetches remote resources.
    """

    schema_version: ReportSchemaVersion = Field(default_factory=ReportSchemaVersion)
    archex_version: str = __version__
    generated_at: str

    source_identity: str
    source_revision: str
    working_tree_fingerprint: str
    index_generation: str
    index_schema_version: str
    parser_versions: dict[str, str] = {}
    chunker_revision: str
    retrieval_profile: str | None = None
    config_fingerprint: str

    freshness: ContextFreshness = ContextFreshness.UNKNOWN
    completeness: ContextCompletenessStatus = ContextCompletenessStatus.UNKNOWN

    producer: Literal["archex-cli"] = "archex-cli"
    producer_version: str = __version__

    confidence: AnalysisConfidence = AnalysisConfidence.UNKNOWN
    evidence_locations: list[EvidenceLocation] = []
    excluded_counts: dict[str, int] = {}
    unknown_counts: dict[str, int] = {}
    redaction_mode: RedactionMode = RedactionMode.REDACTED

    diff: DiffAnalysis

    @model_validator(mode="after")
    def _validate_schema_version(self) -> AnalysisArtifactV1:
        assert_supported_report_schema_version(self.schema_version)
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


def assert_supported_report_schema_version(version: ReportSchemaVersion) -> None:
    if version.major != SUPPORTED_REPORT_SCHEMA_MAJOR:
        raise ReportArtifactError(
            f"Unsupported report schema major version {version.major}; "
            f"this archex build supports major {SUPPORTED_REPORT_SCHEMA_MAJOR}."
        )


def load_analysis_artifact(path: Path) -> AnalysisArtifactV1:
    """Load and schema-validate a previously exported `AnalysisArtifactV1`."""
    try:
        data = json.loads(path.read_text())
        artifact = AnalysisArtifactV1.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReportArtifactError(f"Malformed report artifact {path}: {exc}") from exc
    return artifact


def build_analysis_artifact(source: str | Path, *, base_ref: str) -> AnalysisArtifactV1:
    """Build the canonical diff-review artifact for `source` against `base_ref`.

    Ensures a current index via `index_repository` (the same read-side
    contract every other analysis command uses), then reuses
    `archex.impact`'s existing deterministic diff-symbol resolution and risk
    classifier without adding new parsing or inference.
    """
    repo_root = Path(source).expanduser().resolve()
    repo_source = RepoSource(local_path=str(source))
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)

    try:
        changes = git_changed_files(repo_root, base_ref)
        hunks = git_diff_hunks(repo_root, base_ref)
    except ImpactError as exc:
        raise ReportArtifactError(str(exc)) from exc

    store = index_repository(repo_source, config=config, index_config=index_config)
    try:
        report = analyze_diff_impact(store, repo_root, changes, hunks, base_ref)
        touched_chunks = resolve_diff_symbols(store, changes, hunks)
        index_generation = read_generation_id(store) or ""
        index_schema_version = store.get_metadata("schema_version") or ""
        chunker_revision = store.get_metadata("chunker_revision") or ""
        working_tree_fingerprint = store.get_metadata("working_tree_signature") or ""
        source_revision = (
            store.get_metadata("commit_hash") or CacheManager.git_head(str(repo_root)) or ""
        )
    finally:
        store.close()

    base_resolved_sha = _resolve_ref(repo_root, base_ref)
    diff = _build_diff_analysis(
        base_ref=base_ref,
        base_resolved_sha=base_resolved_sha,
        head_ref=source_revision,
        report=report,
        touched_chunks=touched_chunks,
        hunks=hunks,
    )
    parser_versions = _parser_versions_for_changes(changes)
    excluded_counts = {"unmapped_changed_files": diff.unsupported_files_total}
    unknown_counts = {
        "changed_files_without_symbol_coverage": _files_without_symbol_coverage(
            report, touched_chunks, hunks
        )
    }
    completeness = (
        ContextCompletenessStatus.INCOMPLETE
        if diff.unsupported_files_total > 0
        else ContextCompletenessStatus.COMPLETE
    )
    confidence = _artifact_confidence(diff)
    evidence_locations = _top_level_evidence(diff)

    return AnalysisArtifactV1(
        generated_at=str(time.time()),
        source_identity=repo_source.url or repo_source.local_path or str(repo_root),
        source_revision=source_revision,
        working_tree_fingerprint=working_tree_fingerprint,
        index_generation=index_generation,
        index_schema_version=index_schema_version,
        parser_versions=parser_versions,
        chunker_revision=chunker_revision,
        retrieval_profile=None,
        config_fingerprint=_config_fingerprint(index_config),
        freshness=ContextFreshness.CLEAN,
        completeness=completeness,
        confidence=confidence,
        evidence_locations=evidence_locations,
        excluded_counts=excluded_counts,
        unknown_counts=unknown_counts,
        diff=diff,
    )


def _resolve_ref(repo_root: Path, ref: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _config_fingerprint(index_config: object) -> str:
    payload = json.dumps(
        index_config.model_dump(mode="json") if isinstance(index_config, BaseModel) else {},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _language_for_path(path: str) -> str | None:
    return EXTENSION_LANGUAGE_MAP.get(Path(path).suffix.lower())


def _parser_versions_for_changes(changes: list[ImpactFileChange]) -> dict[str, str]:
    languages = {
        language for change in changes if (language := _language_for_path(change.path)) is not None
    }
    versions: dict[str, str] = {}
    for language in sorted(languages):
        distribution = _PARSER_DISTRIBUTIONS.get(language, _FALLBACK_PARSER_DISTRIBUTION)
        try:
            versions[language] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[language] = "unknown"
    return versions


def _symbol_candidate(impact: SymbolImpact, chunk: CodeChunk | None) -> SymbolCandidate:
    if chunk is not None and chunk.symbol_id is not None:
        handle = symbol_handle(chunk.symbol_id)
    elif chunk is not None:
        handle = chunk_handle(chunk.id)
    else:
        handle = file_handle(impact.file_path)
    # `_classify_file_risk` always evaluates all three deterministic signals
    # (never a partial/fallback path), so every emitted candidate carries the
    # same evidentiary completeness; genuine uncertainty is tracked instead by
    # `unknown_counts["changed_files_without_symbol_coverage"]` for diffs where
    # no symbol-level finding could be produced at all.
    confidence = AnalysisConfidence.HIGH
    evidence = [
        EvidenceLocation(
            path=impact.file_path,
            start_line=impact.start_line,
            end_line=impact.end_line,
            handle=handle,
            description=signal.detail,
        )
        for signal in impact.signals
    ]
    return SymbolCandidate(
        handle=handle,
        file_path=impact.file_path,
        symbol_name=impact.symbol_name,
        qualified_name=impact.qualified_name,
        symbol_kind=impact.symbol_kind,
        start_line=impact.start_line,
        end_line=impact.end_line,
        risk_level=impact.level.value,
        confidence=confidence,
        signals=[f"{signal.name}: {signal.detail}" for signal in impact.signals],
        evidence=evidence,
    )


def _build_diff_analysis(
    *,
    base_ref: str,
    base_resolved_sha: str,
    head_ref: str,
    report: ImpactReport,
    touched_chunks: list[CodeChunk],
    hunks: dict[str, list[tuple[int, int]]],
) -> DiffAnalysis:
    changed_files = [
        DiffFileChange(
            path=change.path,
            status=change.status,
            old_path=change.old_path,
            handle=file_handle(change.path),
            hunks=[
                DiffHunk(start_line=start, end_line=end)
                for start, end in hunks.get(change.path, [])
            ],
        )
        for change in report.changed_files
    ]
    changed_total = len(changed_files)
    changed_files = changed_files[:MAX_CHANGED_FILES]

    symbol_candidates = [
        _symbol_candidate(impact, chunk)
        for impact, chunk in _zip_symbols(report.affected_symbols, touched_chunks)
    ]
    symbols_total = len(symbol_candidates)
    symbol_candidates = symbol_candidates[:MAX_SYMBOL_CANDIDATES]

    affected_interfaces = [
        InterfaceCandidate(
            path=path,
            handle=file_handle(path),
            evidence=[EvidenceLocation(path=path, description="public interface symbol changed")],
        )
        for path in report.affected_interfaces
    ]
    interfaces_total = len(affected_interfaces)
    affected_interfaces = affected_interfaces[:MAX_INTERFACE_CANDIDATES]

    test_candidates = [
        TestCandidate(
            path=path,
            handle=file_handle(path),
            reason="affected_test_surface",
            evidence=[
                EvidenceLocation(path=path, description="test file reachable from changed files")
            ],
        )
        for path in report.affected_tests
    ]
    tests_total = len(test_candidates)
    test_candidates = test_candidates[:MAX_TEST_CANDIDATES]

    unsupported_files = [UnsupportedFile(path=path) for path in report.unmapped_files]
    unsupported_total = len(unsupported_files)
    unsupported_files = unsupported_files[:MAX_UNSUPPORTED_FILES]

    return DiffAnalysis(
        base_ref=base_ref,
        base_resolved_sha=base_resolved_sha,
        head_ref=head_ref,
        changed_files=changed_files,
        changed_files_total=changed_total,
        symbol_candidates=symbol_candidates,
        symbol_candidates_total=symbols_total,
        affected_interfaces=affected_interfaces,
        affected_interfaces_total=interfaces_total,
        test_candidates=test_candidates,
        test_candidates_total=tests_total,
        unsupported_files=unsupported_files,
        unsupported_files_total=unsupported_total,
        risk_level=report.risk.level,
        risk_reasons=report.risk.reasons,
    )


def _zip_symbols(
    impacts: list[SymbolImpact], chunks: list[CodeChunk]
) -> list[tuple[SymbolImpact, CodeChunk | None]]:
    """Pair each `SymbolImpact` with the chunk it was derived from.

    `analyze_diff_impact` builds `affected_symbols` by iterating
    `resolve_diff_symbols`'s already-sorted output in order and without
    filtering, so index-aligned zipping is exact, not a best-effort match.
    """
    if len(impacts) != len(chunks):
        return [(impact, None) for impact in impacts]
    return list(zip(impacts, chunks, strict=True))


def _files_without_symbol_coverage(
    report: ImpactReport,
    touched_chunks: list[CodeChunk],
    hunks: dict[str, list[tuple[int, int]]],
) -> int:
    """Count changed files whose hunks touch no indexed symbol span.

    Distinct from `unmapped_files` (not in the index at all): these files
    *are* indexed, but the diff only touched module-level code, imports,
    comments, or blank lines outside any symbol body, so no symbol-level
    finding could be produced -- the diff's structural impact for that file
    is genuinely unknown rather than known-clean.
    """
    files_with_hunks = {
        change.path
        for change in report.changed_files
        if change.status != "D" and hunks.get(change.path)
    }
    files_with_coverage = {chunk.file_path for chunk in touched_chunks}
    unmapped = set(report.unmapped_files)
    return len((files_with_hunks - files_with_coverage) - unmapped)


def _artifact_confidence(diff: DiffAnalysis) -> AnalysisConfidence:
    if diff.changed_files_total == 0:
        return AnalysisConfidence.HIGH
    if diff.unsupported_files_total >= diff.changed_files_total:
        return AnalysisConfidence.LOW
    if diff.unsupported_files_total > 0:
        return AnalysisConfidence.MEDIUM
    return AnalysisConfidence.HIGH


def _top_level_evidence(diff: DiffAnalysis) -> list[EvidenceLocation]:
    evidence = [
        EvidenceLocation(
            path=change.path,
            start_line=change.hunks[0].start_line if change.hunks else None,
            end_line=change.hunks[-1].end_line if change.hunks else None,
            handle=change.handle,
            description=f"changed file ({change.status})",
        )
        for change in diff.changed_files
    ]
    return evidence[:MAX_EVIDENCE_LOCATIONS]
