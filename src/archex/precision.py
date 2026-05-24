"""Precision file and symbol APIs for indexed repositories."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from archex.models import (
    CodeChunk,
    Config,
    FileOutline,
    FileTree,
    FileTreeEntry,
    IndexConfig,
    PipelineTiming,
    RepoSource,
    SymbolKind,
    SymbolMatch,
    SymbolOutline,
    SymbolSource,
    Visibility,
)

if TYPE_CHECKING:
    from archex.index.store import IndexStore


class EnsureIndex(Protocol):
    """Callable boundary for opening an indexed repository store."""

    def __call__(
        self,
        source: RepoSource,
        config: Config | None = None,
        timing: PipelineTiming | None = None,
        index_config: IndexConfig | None = None,
    ) -> IndexStore: ...


def file_tree(
    source: RepoSource,
    max_depth: int,
    language: str | None,
    config: Config | None,
    timing: PipelineTiming | None,
    ensure_index: EnsureIndex,
) -> FileTree:
    """Return the annotated file structure of an indexed repository."""
    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        file_meta = store.get_file_metadata()
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if language:
        file_meta = [m for m in file_meta if m["language"] == language]

    lang_counts: dict[str, int] = {}
    root_entries: dict[str, FileTreeEntry] = {}

    for meta in file_meta:
        fp = str(meta["file_path"])
        lang = str(meta["language"])
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        parts = fp.split("/")
        current_level = root_entries
        for i, part in enumerate(parts[:-1]):
            if i >= max_depth:
                break
            if part not in current_level:
                dir_path = "/".join(parts[: i + 1])
                current_level[part] = FileTreeEntry(path=dir_path, is_directory=True)
            entry = current_level[part]
            child_map = {c.path.split("/")[-1]: c for c in entry.children}
            current_level = child_map
            if i + 1 < len(parts) - 1:
                next_part = parts[i + 1]
                if next_part not in current_level:
                    next_path = "/".join(parts[: i + 2])
                    new_child = FileTreeEntry(path=next_path, is_directory=True)
                    entry.children.append(new_child)
                    current_level[next_part] = new_child

        if len(parts) - 1 < max_depth:
            file_entry = FileTreeEntry(
                path=fp,
                language=lang,
                lines=int(meta["lines"]),
                symbol_count=int(meta["symbol_count"]),
                is_directory=False,
            )
            if len(parts) == 1:
                root_entries[parts[0]] = file_entry
            else:
                _add_file_to_tree(root_entries, parts, file_entry)

    entries = sorted(root_entries.values(), key=lambda e: (not e.is_directory, e.path))

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return FileTree(
        root=source.local_path or source.url or "",
        entries=entries,
        total_files=len(file_meta),
        languages=lang_counts,
    )


def file_outline(
    source: RepoSource,
    file_path: str,
    config: Config | None,
    timing: PipelineTiming | None,
    ensure_index: EnsureIndex,
) -> FileOutline:
    """Return the symbol hierarchy for a single file."""
    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        chunks = store.get_chunks_for_file(file_path)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if not chunks:
        return FileOutline(
            file_path=file_path,
            language="unknown",
            lines=0,
            symbols=[],
            token_count_raw=0,
        )

    language = chunks[0].language
    max_line = max(c.end_line for c in chunks)
    token_count_raw = sum(c.token_count for c in chunks)
    outlines = [_chunk_to_symbol_outline(c) for c in chunks]

    top_level: list[SymbolOutline] = []
    by_qname: dict[str, SymbolOutline] = {}

    for outline in outlines:
        qname = outline.name
        chunk = next((c for c in chunks if (c.symbol_id or c.id) == outline.symbol_id), None)
        if chunk and chunk.qualified_name:
            qname = chunk.qualified_name
        by_qname[qname] = outline

    for outline in outlines:
        chunk = next((c for c in chunks if (c.symbol_id or c.id) == outline.symbol_id), None)
        qname = chunk.qualified_name if chunk and chunk.qualified_name else outline.name
        parent_name = _get_parent_qname(qname)
        if parent_name and parent_name in by_qname:
            by_qname[parent_name].children.append(outline)
        else:
            top_level.append(outline)

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return FileOutline(
        file_path=file_path,
        language=language,
        lines=max_line,
        symbols=top_level,
        token_count_raw=token_count_raw,
    )


def search_symbols(
    source: RepoSource,
    query: str,
    kind: str | None,
    language: str | None,
    limit: int,
    config: Config | None,
    timing: PipelineTiming | None,
    ensure_index: EnsureIndex,
) -> list[SymbolMatch]:
    """Search symbols by name across the indexed repository."""
    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        sym_kind = SymbolKind(kind) if kind else None
        chunks = store.search_symbols(query, kind=sym_kind, limit=limit)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if language:
        chunks = [c for c in chunks if c.language == language]

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return [_chunk_to_symbol_match(c) for c in chunks[:limit]]


def get_symbol(
    source: RepoSource,
    symbol_id: str,
    config: Config | None,
    timing: PipelineTiming | None,
    ensure_index: EnsureIndex,
) -> SymbolSource | None:
    """Retrieve the full source code of a single symbol by its stable ID."""
    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        chunk = store.get_chunk_by_symbol_id(symbol_id)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    if chunk is None:
        return None
    return _chunk_to_symbol_source(chunk)


def get_symbols_batch(
    source: RepoSource,
    symbol_ids: list[str],
    config: Config | None,
    timing: PipelineTiming | None,
    ensure_index: EnsureIndex,
) -> list[SymbolSource | None]:
    """Batch retrieve N symbols by their stable IDs. Preserves input order."""
    if len(symbol_ids) > 50:
        raise ValueError(f"Maximum 50 symbol IDs per batch, got {len(symbol_ids)}")

    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        chunks = store.get_chunks_by_symbol_ids(symbol_ids)
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    by_sid: dict[str, CodeChunk] = {c.symbol_id: c for c in chunks if c.symbol_id}
    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return [_chunk_to_symbol_source(by_sid[sid]) if sid in by_sid else None for sid in symbol_ids]


def get_repo_total_tokens(
    source: RepoSource,
    config: Config | None,
    ensure_index: EnsureIndex,
) -> int:
    """Return the total token count across all indexed chunks for a repository."""
    store = ensure_index(source, config)
    try:
        return store.get_total_tokens()
    finally:
        store.close()


def get_file_token_count(
    source: RepoSource,
    file_path: str,
    config: Config | None,
    ensure_index: EnsureIndex,
) -> int:
    """Return the total token count for a single file in an indexed repository."""
    store = ensure_index(source, config)
    try:
        return store.get_file_tokens(file_path)
    finally:
        store.close()


def get_files_token_count(
    source: RepoSource,
    file_paths: list[str],
    config: Config | None,
    ensure_index: EnsureIndex,
) -> int:
    """Return the total token count across unique files in an indexed repository."""
    store = ensure_index(source, config)
    try:
        return store.get_files_tokens(file_paths)
    finally:
        store.close()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _chunk_to_symbol_source(chunk: CodeChunk) -> SymbolSource:
    return SymbolSource(
        symbol_id=chunk.symbol_id or chunk.id,
        name=chunk.symbol_name or "",
        kind=chunk.symbol_kind or SymbolKind.VARIABLE,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        signature=chunk.signature,
        visibility=Visibility(chunk.visibility) if chunk.visibility else Visibility.PUBLIC,
        docstring=chunk.docstring,
        source=chunk.content,
        imports_context=chunk.imports_context,
        token_count=chunk.token_count,
    )


def _chunk_to_symbol_outline(chunk: CodeChunk) -> SymbolOutline:
    return SymbolOutline(
        symbol_id=chunk.symbol_id or chunk.id,
        name=chunk.symbol_name or "",
        kind=chunk.symbol_kind or SymbolKind.VARIABLE,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        signature=chunk.signature,
        visibility=Visibility(chunk.visibility) if chunk.visibility else Visibility.PUBLIC,
        docstring=chunk.docstring,
    )


def _chunk_to_symbol_match(chunk: CodeChunk, score: float = 0.0) -> SymbolMatch:
    return SymbolMatch(
        symbol_id=chunk.symbol_id or chunk.id,
        name=chunk.symbol_name or "",
        kind=chunk.symbol_kind or SymbolKind.VARIABLE,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        signature=chunk.signature,
        visibility=Visibility(chunk.visibility) if chunk.visibility else Visibility.PUBLIC,
        relevance_score=score,
    )


def _add_file_to_tree(
    root_entries: dict[str, FileTreeEntry],
    parts: list[str],
    file_entry: FileTreeEntry,
) -> None:
    """Walk the tree and add a file entry under its parent directory."""
    if parts[0] not in root_entries:
        dir_path = parts[0]
        root_entries[parts[0]] = FileTreeEntry(path=dir_path, is_directory=True)

    current = root_entries[parts[0]]
    for part in parts[1:-1]:
        found = False
        for child in current.children:
            if child.path.split("/")[-1] == part:
                current = child
                found = True
                break
        if not found:
            return

    existing_paths = {c.path for c in current.children}
    if file_entry.path not in existing_paths:
        current.children.append(file_entry)


def _get_parent_qname(qualified_name: str) -> str | None:
    """Extract the parent's qualified name from a dotted or :: separated name."""
    if "::" in qualified_name:
        parts = qualified_name.rsplit("::", 1)
        return parts[0] if len(parts) > 1 else None
    if "." in qualified_name:
        parts = qualified_name.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else None
    return None
