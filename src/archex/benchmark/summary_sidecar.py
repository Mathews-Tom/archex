"""Offline-built summary sidecars for the summary-first benchmark lane.

A sidecar is an explicit, offline-generated artifact: per file/symbol summaries
plus the provenance needed to detect staleness (source content hash + index
revision + generation time) and to fetch the exact original code (file path +
line range + fetch handle). Summaries are deterministic, rule-based digests of
already-parsed symbol metadata — no hosted LLM calls, no network. They are built
only on explicit offline opt-in and are NEVER auto-generated during a normal
query. Inspired by Meta-RAG's summary-first retrieval lane (arXiv:2508.02611).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from archex.scout import chunk_handle

if TYPE_CHECKING:
    from archex.models import CodeChunk

# Max characters kept in a single summary digest; keeps sidecars compact and the
# summary-first selection cheap while preserving the signal-bearing prefix.
_SUMMARY_MAX_CHARS = 280


class SummaryGranularity(StrEnum):
    """Granularity a summary sidecar entry describes."""

    FILE = "file"
    SYMBOL = "symbol"
    BLOCK = "block"


class SummaryEntry(BaseModel):
    """A single offline summary plus the provenance needed to trust and fetch it."""

    source_file_path: str
    source_content_hash: str
    index_revision: str
    generated_at: str
    granularity: SummaryGranularity
    summary: str
    symbol_name: str | None = None
    start_line: int
    end_line: int
    fetch_original_handle: str


class SummarySidecar(BaseModel):
    """An offline-built collection of summaries for one repository revision."""

    sidecar_version: int = 1
    repo: str
    commit: str
    index_revision: str
    generated_at: str
    granularity: SummaryGranularity
    entries: list[SummaryEntry] = []

    def save(self, path: str | Path) -> Path:
        """Write the sidecar to ``path`` as JSON; returns the resolved path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> SummarySidecar:
        """Load a sidecar from a JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _first_meaningful_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def summarize_chunk(chunk: CodeChunk) -> str:
    """Deterministic rule-based digest of a chunk's parsed symbol metadata.

    Composes breadcrumbs, signature (or symbol name), and the first docstring
    line, falling back to the first non-blank source line. No model, no network.
    """
    parts: list[str] = []
    if chunk.breadcrumbs:
        parts.append(chunk.breadcrumbs)
    if chunk.signature:
        parts.append(chunk.signature)
    elif chunk.symbol_name:
        parts.append(chunk.symbol_name)
    if chunk.docstring:
        parts.append(_first_meaningful_line(chunk.docstring))
    if not parts:
        parts.append(_first_meaningful_line(chunk.content))
    digest = " | ".join(part for part in parts if part)
    return digest[:_SUMMARY_MAX_CHARS]


def file_content_hash(repo_path: Path, file_path: str) -> str | None:
    """SHA-256 of a repo file's bytes, or None when the file is missing."""
    target = repo_path / file_path
    if not target.is_file():
        return None
    return hashlib.sha256(target.read_bytes()).hexdigest()


def build_summary_sidecar(
    chunks: list[CodeChunk],
    repo_path: Path,
    *,
    repo: str,
    commit: str,
    index_revision: str,
    granularity: SummaryGranularity = SummaryGranularity.SYMBOL,
    generated_at: str | None = None,
) -> SummarySidecar:
    """Build a summary sidecar from already-parsed chunks (explicit offline step).

    The source content hash is the whole-file hash so any edit to a file marks all
    of its summaries stale. Chunks whose file cannot be read are skipped.
    """
    now = generated_at or datetime.now(tz=UTC).isoformat()
    hash_cache: dict[str, str | None] = {}
    entries: list[SummaryEntry] = []
    for chunk in chunks:
        if chunk.file_path not in hash_cache:
            hash_cache[chunk.file_path] = file_content_hash(repo_path, chunk.file_path)
        source_hash = hash_cache[chunk.file_path]
        if source_hash is None:
            continue
        entries.append(
            SummaryEntry(
                source_file_path=chunk.file_path,
                source_content_hash=source_hash,
                index_revision=index_revision,
                generated_at=now,
                granularity=granularity,
                summary=summarize_chunk(chunk),
                symbol_name=chunk.symbol_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                fetch_original_handle=chunk_handle(chunk.id),
            )
        )
    return SummarySidecar(
        repo=repo,
        commit=commit,
        index_revision=index_revision,
        generated_at=now,
        granularity=granularity,
        entries=entries,
    )


def is_entry_stale(entry: SummaryEntry, repo_path: Path) -> bool:
    """True when the source file is gone or its content no longer matches the hash."""
    current = file_content_hash(repo_path, entry.source_file_path)
    return current is None or current != entry.source_content_hash
