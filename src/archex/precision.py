"""Precision file and symbol APIs for indexed repositories."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from archex.languages import get_language_tier
from archex.models import (
    ChunkRange,
    CodeChunk,
    Config,
    FileOutline,
    FileTree,
    FileTreeEntry,
    ImportStatement,
    IndexConfig,
    LanguageTier,
    PipelineTiming,
    RepoSource,
    SymbolKind,
    SymbolMatch,
    SymbolOutline,
    SymbolSource,
    Visibility,
)
from archex.scout import normalize_symbol_lookup_handle, parse_scout_handle

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
    is_structured = get_language_tier(language) is LanguageTier.STRUCTURED
    if is_structured:
        symbol_chunks: list[CodeChunk] = []
        outline_ranges = [
            ChunkRange(start_line=chunk.start_line, end_line=chunk.end_line) for chunk in chunks
        ]
    else:
        symbol_chunks = chunks
        outline_ranges: list[ChunkRange] = []
    outlines = [_chunk_to_symbol_outline(c) for c in symbol_chunks]

    top_level: list[SymbolOutline] = []
    by_qname: dict[str, SymbolOutline] = {}

    for outline in outlines:
        qname = outline.name
        chunk = next((c for c in symbol_chunks if (c.symbol_id or c.id) == outline.symbol_id), None)
        if chunk and chunk.qualified_name:
            qname = chunk.qualified_name
        by_qname[qname] = outline

    for outline in outlines:
        chunk = next((c for c in symbol_chunks if (c.symbol_id or c.id) == outline.symbol_id), None)
        qname = chunk.qualified_name if chunk and chunk.qualified_name else outline.name
        parent_name = _get_parent_qname(qname)
        if parent_name and parent_name in by_qname:
            by_qname[parent_name].children.append(outline)
        else:
            top_level.append(outline)

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    references = (
        _extract_file_references(source, file_path, language, config) if is_structured else []
    )
    return FileOutline(
        file_path=file_path,
        language=language,
        lines=max_line,
        symbols=top_level,
        token_count_raw=token_count_raw,
        outline_ranges=outline_ranges,
        references=references,
    )


def _extract_file_references(
    source: RepoSource,
    file_path: str,
    language: str,
    config: Config | None,  # noqa: ARG001
) -> list[ImportStatement]:
    if source.local_path is None:
        return []

    repo_path = Path(source.local_path)
    absolute_path = repo_path / file_path
    if not absolute_path.is_file():
        return []

    from archex.parse import TreeSitterEngine
    from archex.parse.adapters import default_adapter_registry

    adapter_cls = default_adapter_registry.get(language)
    if adapter_cls is None:
        return []
    adapter = adapter_cls()

    source_bytes = absolute_path.read_bytes()
    tree = TreeSitterEngine().parse_bytes(source_bytes, language)
    references = adapter.parse_imports(tree, source_bytes, file_path)
    if not references:
        return []

    file_map = _local_file_map(repo_path)
    for reference in references:
        reference.resolved_path = adapter.resolve_import(reference, file_map)
    return references


def _local_file_map(repo_path: Path) -> dict[str, str]:
    file_map: dict[str, str] = {}
    for path in repo_path.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(repo_path).parts:
            continue
        relative_path = path.relative_to(repo_path).as_posix()
        file_map[relative_path] = relative_path
    return file_map


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
    """Retrieve the full source code of a single symbol by its stable ID or scout handle."""
    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        chunk = _chunk_for_symbol_lookup(store, symbol_id)
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
    """Batch retrieve symbols by stable IDs or scout handles. Preserves input order."""
    if len(symbol_ids) > 50:
        raise ValueError(f"Maximum 50 symbol IDs per batch, got {len(symbol_ids)}")

    t0 = time.perf_counter()
    store = ensure_index(source, config, timing)
    try:
        t_op = time.perf_counter()
        chunks = [_chunk_for_symbol_lookup(store, symbol_id) for symbol_id in symbol_ids]
        if timing is not None:
            timing.search_ms = _elapsed_ms(t_op)
    finally:
        store.close()

    if timing is not None:
        timing.total_ms = _elapsed_ms(t0)
    return [_chunk_to_symbol_source(chunk) if chunk is not None else None for chunk in chunks]


def _chunk_for_symbol_lookup(store: IndexStore, symbol_id: str) -> CodeChunk | None:
    lookup_id = normalize_symbol_lookup_handle(symbol_id)
    handle = parse_scout_handle(symbol_id)
    if handle is not None and handle.kind == "chunk":
        return store.get_chunk(handle.value)
    if handle is not None and handle.kind == "file":
        return None
    return store.get_chunk_by_symbol_id(lookup_id)


def get_repo_total_tokens(
    source: RepoSource,
    config: Config | None,
    ensure_index: EnsureIndex,
) -> int | None:
    """Return the true total file-token count for a repository.

    Returns None for a legacy index whose per-file token totals are unpopulated.
    """
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
) -> int | None:
    """Return the true token total for a single file (None if unpopulated)."""
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
) -> int | None:
    """Return the true token total across unique files (None if any unpopulated)."""
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
