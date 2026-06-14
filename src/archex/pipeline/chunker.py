"""Token-aware code chunker: split ParsedFile symbols into bounded CodeChunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import tiktoken

from archex.models import (
    ChunkerName,
    CodeChunk,
    ImportStatement,
    IndexConfig,
    ParsedFile,
    Symbol,
    SymbolKind,
    make_symbol_id,
)

_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SNAKE_SPLIT = re.compile(r"_+")
_MODULE_EXTENSIONS = frozenset(
    {".py", ".js", ".ts", ".tsx", ".jsx", ".rb", ".java", ".kt", ".go", ".rs", ".cs", ".swift"}
)


def expand_identifiers(text: str) -> str:
    """Expand camelCase and snake_case identifiers into space-separated tokens for FTS5."""
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)
    fragments: list[str] = []
    for ident in identifiers:
        parts = _CAMEL_SPLIT.split(ident)
        for part in parts:
            fragments.extend(_SNAKE_SPLIT.split(part))
    unique = {f.lower() for f in fragments if len(f) > 1}
    return text + "\n" + " ".join(sorted(unique)) if unique else text


def _file_path_to_module(file_path: str) -> str:
    """Convert a file path to a dotted module-like string.

    ``src/archex/pipeline/chunker.py`` → ``archex.pipeline.chunker``
    Strips common prefixes (``src/``, ``lib/``) and file extensions.
    """
    import os

    path = file_path.replace("\\", "/")

    for prefix in ("src/", "lib/", "app/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break

    root, ext = os.path.splitext(path)
    if ext in _MODULE_EXTENSIONS:
        path = root

    for suffix in ("/__init__", "/index"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]

    return path.replace("/", ".")


def build_breadcrumbs(
    file_path: str,
    symbol: Symbol | None,
    all_symbols: list[Symbol] | None = None,
) -> str:
    """Build a compact structural breadcrumb string for a chunk.

    Format: ``module: archex.pipeline.chunker > class: Greeter > method: greet``
    For file-level chunks (no symbol): ``module: archex.pipeline.chunker``
    """
    parts: list[str] = []
    module_path = _file_path_to_module(file_path)
    parts.append(f"module: {module_path}")

    if symbol is None:
        return " > ".join(parts)

    qname = symbol.qualified_name
    if not qname:
        parts.append(f"{symbol.kind}: {symbol.name}")
        return " > ".join(parts)

    segments = qname.split(".")
    if len(segments) == 1:
        parts.append(f"{symbol.kind}: {segments[0]}")
    else:
        parent_kinds = _resolve_parent_kinds(segments[:-1], file_path, all_symbols)
        for seg, kind in zip(segments[:-1], parent_kinds, strict=False):
            parts.append(f"{kind}: {seg}")
        parts.append(f"{symbol.kind}: {segments[-1]}")

    return " > ".join(parts)


def _resolve_parent_kinds(
    parent_segments: list[str],
    file_path: str,
    all_symbols: list[Symbol] | None,
) -> list[str]:
    """Resolve the SymbolKind for each parent segment in a qualified name chain."""
    if not all_symbols:
        return ["class"] * len(parent_segments)

    sym_kinds: dict[str, str] = {}
    for sym in all_symbols:
        if sym.file_path == file_path and sym.qualified_name:
            sym_kinds[sym.qualified_name] = str(sym.kind)

    result: list[str] = []
    for i, _seg in enumerate(parent_segments):
        partial_qname = ".".join(parent_segments[: i + 1])
        kind = sym_kinds.get(partial_qname, "class")
        result.append(kind)
    return result


@runtime_checkable
class Chunker(Protocol):
    """Protocol for code chunkers, allowing custom implementations."""

    def chunk_file(self, parsed_file: ParsedFile, source: bytes) -> list[CodeChunk]: ...

    def chunk_files(
        self, parsed_files: list[ParsedFile], sources: dict[str, bytes]
    ) -> list[CodeChunk]: ...


def _count_tokens(encoder: tiktoken.Encoding, text: str) -> int:
    return len(encoder.encode(text, disallowed_special=()))


def _format_import(imp: ImportStatement) -> str:
    if imp.symbols:
        symbols_str = ", ".join(imp.symbols)
        if imp.alias:
            return f"from {imp.module} import {symbols_str} as {imp.alias}"
        return f"from {imp.module} import {symbols_str}"
    if imp.alias:
        return f"import {imp.module} as {imp.alias}"
    return f"import {imp.module}"


def _import_relevant(imp: ImportStatement, content: str) -> bool:
    """Return True if the import is used in content."""
    if imp.alias and imp.alias in content:
        return True
    if imp.symbols:
        return any(sym in content for sym in imp.symbols)
    # bare import — check if the last component appears in content
    base = imp.module.split(".")[-1]
    return base in content


def _split_lines_at_boundary(
    lines: list[bytes], max_tokens: int, encoder: tiktoken.Encoding
) -> list[list[bytes]]:
    """Split a list of lines into groups, each under max_tokens."""
    chunks: list[list[bytes]] = []
    current: list[bytes] = []
    current_tokens = 0

    for line in lines:
        line_text = line.decode("utf-8", errors="replace")
        line_tokens = _count_tokens(encoder, line_text)

        # If adding this line would exceed the max, flush
        if current_tokens + line_tokens > max_tokens and current:
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append(current)

    return chunks


def _is_blank_lines(lines: list[bytes]) -> bool:
    return not lines or all(line.strip() == b"" for line in lines)


def _extract_source_lines(all_lines: list[bytes], start_line: int, end_line: int) -> list[bytes]:
    """Extract 1-indexed [start_line, end_line] from pre-split line list."""
    lo = max(0, start_line - 1)
    hi = min(len(all_lines), end_line)
    return all_lines[lo:hi]


def _make_chunk_id(file_path: str, symbol_name: str | None, start_line: int) -> str:
    name = symbol_name if symbol_name is not None else "_module"
    return f"{file_path}:{name}:{start_line}"


def _make_symbol_id(
    file_path: str,
    qualified_name: str | None,
    kind: SymbolKind | None,
) -> str:
    return make_symbol_id(file_path, qualified_name, kind)


def _disambiguate_symbol_ids(chunks: list[CodeChunk]) -> None:
    seen: dict[str, list[CodeChunk]] = {}
    for chunk in chunks:
        if chunk.symbol_id:
            seen.setdefault(chunk.symbol_id, []).append(chunk)
    for sid, group in seen.items():
        if len(group) > 1:
            group.sort(key=lambda c: c.start_line)
            for i, chunk in enumerate(group):
                if i > 0:
                    chunk.symbol_id = f"{sid}@{i + 1}"


def _lines_to_text(lines: list[bytes]) -> str:
    return "\n".join(line.decode("utf-8", errors="replace") for line in lines)


@dataclass
class _ChunkCandidate:
    lines: list[bytes]
    start_line: int
    symbol: Symbol | None
    content: str
    token_count: int


@dataclass(frozen=True)
class _CastUnit:
    start_line: int
    end_line: int
    symbol: Symbol | None
    token_count: int


def _cast_merge_budget(max_tokens: int) -> int:
    return max(1, int(max_tokens * 0.7))


def _append_candidates(
    candidates: list[_ChunkCandidate],
    *,
    lines: list[bytes],
    start_line: int,
    symbol: Symbol | None,
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> None:
    if _is_blank_lines(lines):
        return
    content = _lines_to_text(lines)
    token_count = _count_tokens(encoder, content)
    if token_count <= max_tokens:
        candidates.append(
            _ChunkCandidate(
                lines=lines,
                start_line=start_line,
                symbol=symbol,
                content=content,
                token_count=token_count,
            )
        )
        return

    offset = start_line
    for group in _split_lines_at_boundary(lines, max_tokens, encoder):
        group_content = _lines_to_text(group)
        candidates.append(
            _ChunkCandidate(
                lines=group,
                start_line=offset,
                symbol=symbol,
                content=group_content,
                token_count=_count_tokens(encoder, group_content),
            )
        )
        offset += len(group)


def _build_chunk(
    *,
    file_path: str,
    language: str,
    candidate: _ChunkCandidate,
    imports: list[ImportStatement],
    encoder: tiktoken.Encoding,
    all_symbols: list[Symbol] | None = None,
) -> CodeChunk:
    content = candidate.content

    relevant_imports = [imp for imp in imports if _import_relevant(imp, content)]
    imports_context = "\n".join(_format_import(imp) for imp in relevant_imports)
    combined = (imports_context + "\n" + content) if imports_context else content
    token_count = _count_tokens(encoder, combined) if imports_context else candidate.token_count
    breadcrumbs = build_breadcrumbs(file_path, candidate.symbol, all_symbols)

    return CodeChunk(
        id=_make_chunk_id(
            file_path,
            candidate.symbol.name if candidate.symbol else None,
            candidate.start_line,
        ),
        symbol_id=_make_symbol_id(
            file_path,
            candidate.symbol.qualified_name if candidate.symbol else None,
            candidate.symbol.kind if candidate.symbol else None,
        ),
        qualified_name=candidate.symbol.qualified_name if candidate.symbol else None,
        visibility=str(candidate.symbol.visibility) if candidate.symbol else "public",
        signature=candidate.symbol.signature if candidate.symbol else None,
        docstring=candidate.symbol.docstring if candidate.symbol else None,
        content=content,
        file_path=file_path,
        start_line=candidate.start_line,
        end_line=candidate.start_line + len(candidate.lines) - 1,
        symbol_name=candidate.symbol.name if candidate.symbol else None,
        symbol_kind=candidate.symbol.kind if candidate.symbol else None,
        language=language,
        imports_context=imports_context,
        token_count=token_count,
        breadcrumbs=breadcrumbs,
    )


class ASTChunker:
    def __init__(self, config: IndexConfig | None = None) -> None:
        self._config = config or IndexConfig()
        self._encoder = tiktoken.get_encoding(self._config.token_encoding)

    def chunk_file(self, parsed_file: ParsedFile, source: bytes) -> list[CodeChunk]:
        config = self._config
        encoder = self._encoder
        file_path = parsed_file.path
        language = parsed_file.language
        imports = parsed_file.imports

        all_source_lines = source.split(b"\n")
        total_lines = len(all_source_lines)

        symbols = sorted(parsed_file.symbols, key=lambda s: s.start_line)
        candidates: list[_ChunkCandidate] = []
        has_structural_ranges = False

        if symbols:
            covered: list[tuple[int, int]] = [(s.start_line, s.end_line) for s in symbols]
            for sym in symbols:
                _append_candidates(
                    candidates,
                    lines=_extract_source_lines(all_source_lines, sym.start_line, sym.end_line),
                    start_line=sym.start_line,
                    symbol=sym,
                    max_tokens=config.chunk_max_tokens,
                    encoder=encoder,
                )
        else:
            ranges = sorted(parsed_file.chunk_ranges, key=lambda r: r.start_line)
            has_structural_ranges = bool(ranges)
            covered = [(r.start_line, r.end_line) for r in ranges]
            for chunk_range in ranges:
                _append_candidates(
                    candidates,
                    lines=_extract_source_lines(
                        all_source_lines, chunk_range.start_line, chunk_range.end_line
                    ),
                    start_line=chunk_range.start_line,
                    symbol=None,
                    max_tokens=config.chunk_max_tokens,
                    encoder=encoder,
                )

        uncovered_ranges = _find_uncovered_ranges(covered, total_lines)
        for range_start, range_end in uncovered_ranges:
            _append_candidates(
                candidates,
                lines=_extract_source_lines(all_source_lines, range_start, range_end),
                start_line=range_start,
                symbol=None,
                max_tokens=config.chunk_max_tokens,
                encoder=encoder,
            )

        if has_structural_ranges:
            merged = sorted(candidates, key=lambda item: item.start_line)
        else:
            merged = _merge_small_chunks(candidates, config.chunk_min_tokens, encoder)

        result: list[CodeChunk] = []
        for candidate in merged:
            if _is_blank_lines(candidate.lines):
                continue
            result.append(
                _build_chunk(
                    file_path=file_path,
                    language=language,
                    candidate=candidate,
                    imports=imports,
                    encoder=encoder,
                    all_symbols=symbols,
                )
            )
        result.sort(key=lambda c: c.start_line)
        _disambiguate_symbol_ids(result)
        return result

    def chunk_files(
        self, parsed_files: list[ParsedFile], sources: dict[str, bytes]
    ) -> list[CodeChunk]:
        all_chunks: list[CodeChunk] = []
        for pf in parsed_files:
            src = sources.get(pf.path, b"")
            all_chunks.extend(self.chunk_file(pf, src))
        all_chunks.sort(key=lambda c: (c.file_path, c.start_line))
        return all_chunks


class CastChunker(ASTChunker):
    """Recursive AST split-merge chunker selected by IndexConfig(chunker="cast")."""

    def chunk_file(self, parsed_file: ParsedFile, source: bytes) -> list[CodeChunk]:
        config = self._config
        encoder = self._encoder
        file_path = parsed_file.path
        language = parsed_file.language
        imports = parsed_file.imports
        all_source_lines = source.split(b"\n")
        total_lines = len(all_source_lines)
        symbols = sorted(parsed_file.symbols, key=lambda s: (s.start_line, s.end_line))

        if symbols:
            units = _cast_units_for_scope(
                start_line=1,
                end_line=total_lines,
                children=_top_level_symbols(symbols),
                all_symbols=symbols,
                all_source_lines=all_source_lines,
                max_tokens=config.chunk_max_tokens,
                encoder=encoder,
            )
        else:
            units = _cast_units_for_ranges(
                parsed_file,
                all_source_lines,
                config.chunk_max_tokens,
                encoder,
            )

        result: list[CodeChunk] = []
        for unit in units:
            lines = _extract_source_lines(all_source_lines, unit.start_line, unit.end_line)
            if _is_blank_lines(lines):
                continue
            content = _lines_to_text(lines)
            result.append(
                _build_chunk(
                    file_path=file_path,
                    language=language,
                    candidate=_ChunkCandidate(
                        lines=lines,
                        start_line=unit.start_line,
                        symbol=unit.symbol,
                        content=content,
                        token_count=_count_tokens(encoder, content),
                    ),
                    imports=imports,
                    encoder=encoder,
                    all_symbols=symbols,
                )
            )
        result.sort(key=lambda c: c.start_line)
        _disambiguate_symbol_ids(result)
        return result


_CHUNKER_REVISIONS: dict[ChunkerName, str] = {
    "default": "v1",
    "cast": "v2",
}


def chunker_revision(chunker: ChunkerName) -> str:
    return _CHUNKER_REVISIONS[chunker]


def create_chunker(config: IndexConfig | None = None) -> Chunker:
    effective = config or IndexConfig()
    if effective.chunker == "cast":
        return CastChunker(effective)
    return ASTChunker(effective)


def _top_level_symbols(symbols: list[Symbol]) -> list[Symbol]:
    top_level: list[Symbol] = []
    for symbol in symbols:
        if _direct_parent_symbol(symbol, symbols) is None:
            top_level.append(symbol)
    return _non_overlapping_symbols(top_level)


def _direct_child_symbols(parent: Symbol, symbols: list[Symbol]) -> list[Symbol]:
    children = [symbol for symbol in symbols if _direct_parent_symbol(symbol, symbols) is parent]
    return _non_overlapping_symbols(children)


def _direct_parent_symbol(symbol: Symbol, symbols: list[Symbol]) -> Symbol | None:
    by_name = {candidate.qualified_name: candidate for candidate in symbols}
    if symbol.parent and symbol.parent in by_name:
        parent = by_name[symbol.parent]
        if (
            parent is not symbol
            and parent.start_line <= symbol.start_line
            and symbol.end_line <= parent.end_line
        ):
            return parent

    containers = [
        candidate
        for candidate in symbols
        if candidate is not symbol
        and candidate.start_line <= symbol.start_line
        and symbol.end_line <= candidate.end_line
        and (candidate.start_line, candidate.end_line) != (symbol.start_line, symbol.end_line)
    ]
    if not containers:
        return None
    return min(containers, key=lambda item: (item.end_line - item.start_line, item.start_line))


def _non_overlapping_symbols(symbols: list[Symbol]) -> list[Symbol]:
    result: list[Symbol] = []
    last_end = 0
    for symbol in sorted(symbols, key=lambda item: (item.start_line, item.end_line)):
        if symbol.start_line <= last_end:
            continue
        result.append(symbol)
        last_end = symbol.end_line
    return result


def _cast_units_for_ranges(
    parsed_file: ParsedFile,
    all_source_lines: list[bytes],
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[_CastUnit]:
    ranges = sorted(parsed_file.chunk_ranges, key=lambda item: item.start_line)
    if not ranges:
        return _split_cast_range(
            1,
            len(all_source_lines),
            None,
            all_source_lines,
            max_tokens,
            encoder,
        )

    units: list[_CastUnit] = []
    for chunk_range in ranges:
        units.extend(
            _split_cast_range(
                chunk_range.start_line,
                chunk_range.end_line,
                None,
                all_source_lines,
                max_tokens,
                encoder,
            )
        )
    return _merge_cast_units(units, all_source_lines, max_tokens, encoder)


def _cast_units_for_scope(
    *,
    start_line: int,
    end_line: int,
    children: list[Symbol],
    all_symbols: list[Symbol],
    all_source_lines: list[bytes],
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[_CastUnit]:
    units: list[_CastUnit] = []
    current = start_line
    for child in children:
        child_start = max(child.start_line, start_line)
        child_end = min(child.end_line, end_line)
        if current < child_start:
            units.extend(
                _split_cast_range(
                    current,
                    child_start - 1,
                    None,
                    all_source_lines,
                    max_tokens,
                    encoder,
                )
            )
        units.extend(
            _cast_units_for_symbol(
                child,
                all_symbols,
                all_source_lines,
                max_tokens,
                encoder,
            )
        )
        current = max(current, child_end + 1)
    if current <= end_line:
        units.extend(
            _split_cast_range(current, end_line, None, all_source_lines, max_tokens, encoder)
        )
    return _merge_cast_units(units, all_source_lines, max_tokens, encoder)


def _cast_units_for_symbol(
    symbol: Symbol,
    all_symbols: list[Symbol],
    all_source_lines: list[bytes],
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[_CastUnit]:
    token_count = _count_range_tokens(
        symbol.start_line,
        symbol.end_line,
        all_source_lines,
        encoder,
    )
    if token_count <= max_tokens:
        return [_CastUnit(symbol.start_line, symbol.end_line, symbol, token_count)]

    children = [
        child
        for child in _direct_child_symbols(symbol, all_symbols)
        if symbol.start_line <= child.start_line and child.end_line <= symbol.end_line
    ]
    if not children:
        return _split_cast_range(
            symbol.start_line,
            symbol.end_line,
            None,
            all_source_lines,
            max_tokens,
            encoder,
        )
    return _cast_units_for_scope(
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        children=children,
        all_symbols=all_symbols,
        all_source_lines=all_source_lines,
        max_tokens=max_tokens,
        encoder=encoder,
    )


def _split_cast_range(
    start_line: int,
    end_line: int,
    symbol: Symbol | None,
    all_source_lines: list[bytes],
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[_CastUnit]:
    if end_line < start_line:
        return []
    lines = _extract_source_lines(all_source_lines, start_line, end_line)
    if _is_blank_lines(lines):
        return []
    content = _lines_to_text(lines)
    token_count = _count_tokens(encoder, content)
    if token_count <= max_tokens:
        return [_CastUnit(start_line, end_line, symbol, token_count)]

    units: list[_CastUnit] = []
    offset = start_line
    for group in _split_lines_at_boundary(lines, max_tokens, encoder):
        if _is_blank_lines(group):
            offset += len(group)
            continue
        group_content = _lines_to_text(group)
        units.append(
            _CastUnit(
                start_line=offset,
                end_line=offset + len(group) - 1,
                symbol=None,
                token_count=_count_tokens(encoder, group_content),
            )
        )
        offset += len(group)
    return units


def _merge_cast_units(
    units: list[_CastUnit],
    all_source_lines: list[bytes],
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[_CastUnit]:
    if not units:
        return []

    merge_budget = _cast_merge_budget(max_tokens)
    result: list[_CastUnit] = []
    group_start = units[0].start_line
    group_end = units[0].end_line
    group_symbol = units[0].symbol
    group_token_count = units[0].token_count
    group_size = 1

    for unit in units[1:]:
        merged_tokens = _count_range_tokens(group_start, unit.end_line, all_source_lines, encoder)
        can_merge = (
            group_token_count < merge_budget
            and unit.token_count < merge_budget
            and merged_tokens <= merge_budget
        )
        if can_merge:
            group_end = unit.end_line
            group_token_count = merged_tokens
            group_size += 1
            continue
        result.append(
            _CastUnit(
                group_start,
                group_end,
                group_symbol if group_size == 1 else None,
                group_token_count,
            )
        )
        group_start = unit.start_line
        group_end = unit.end_line
        group_symbol = unit.symbol
        group_token_count = unit.token_count
        group_size = 1

    result.append(
        _CastUnit(
            group_start,
            group_end,
            group_symbol if group_size == 1 else None,
            group_token_count,
        )
    )
    return result


def _count_range_tokens(
    start_line: int,
    end_line: int,
    all_source_lines: list[bytes],
    encoder: tiktoken.Encoding,
) -> int:
    lines = _extract_source_lines(all_source_lines, start_line, end_line)
    return _count_tokens(encoder, _lines_to_text(lines))


def _find_uncovered_ranges(
    covered: list[tuple[int, int]], total_lines: int
) -> list[tuple[int, int]]:
    """Return 1-indexed line ranges not covered by any symbol."""
    if not covered:
        if total_lines > 0:
            return [(1, total_lines)]
        return []

    uncovered: list[tuple[int, int]] = []

    prev_end = 0
    for start, end in covered:
        if start > prev_end + 1:
            uncovered.append((prev_end + 1, start - 1))
        prev_end = max(prev_end, end)

    if prev_end < total_lines:
        uncovered.append((prev_end + 1, total_lines))

    return uncovered


def _merge_candidates(
    left: _ChunkCandidate,
    right: _ChunkCandidate,
    encoder: tiktoken.Encoding,
) -> _ChunkCandidate:
    lines = left.lines + right.lines
    content = left.content + "\n" + right.content
    return _ChunkCandidate(
        lines=lines,
        start_line=left.start_line,
        symbol=None,
        content=content,
        token_count=_count_tokens(encoder, content),
    )


def _merge_small_chunks(
    candidates: list[_ChunkCandidate],
    min_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[_ChunkCandidate]:
    """Merge adjacent small (below min_tokens) file-level chunks."""
    if not candidates:
        return []

    result: list[_ChunkCandidate] = []
    i = 0
    while i < len(candidates):
        candidate = candidates[i]

        if candidate.token_count < min_tokens and candidate.symbol is None:
            if i + 1 < len(candidates) and candidates[i + 1].symbol is None:
                candidates[i + 1] = _merge_candidates(candidate, candidates[i + 1], encoder)
                i += 1
                continue
            if result and result[-1].symbol is None:
                result[-1] = _merge_candidates(result[-1], candidate, encoder)
                i += 1
                continue

        result.append(candidate)
        i += 1

    return result
