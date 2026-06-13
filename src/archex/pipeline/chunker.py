"""Token-aware code chunker: split ParsedFile symbols into bounded CodeChunks."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import tiktoken

from archex.models import (
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


def _append_candidates(
    candidates: list[tuple[list[bytes], int, Symbol | None]],
    *,
    lines: list[bytes],
    start_line: int,
    symbol: Symbol | None,
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> None:
    if _is_blank_lines(lines):
        return

    token_count = _count_tokens(encoder, _lines_to_text(lines))
    if token_count <= max_tokens:
        candidates.append((lines, start_line, symbol))
        return

    offset = start_line
    for group in _split_lines_at_boundary(lines, max_tokens, encoder):
        candidates.append((group, offset, symbol))
        offset += len(group)


def _build_chunk(
    *,
    file_path: str,
    language: str,
    lines: list[bytes],
    start_line: int,
    symbol: Symbol | None,
    imports: list[ImportStatement],
    encoder: tiktoken.Encoding,
    all_symbols: list[Symbol] | None = None,
) -> CodeChunk:
    content = _lines_to_text(lines)

    # Filter imports relevant to this chunk
    relevant_imports = [imp for imp in imports if _import_relevant(imp, content)]
    imports_context = "\n".join(_format_import(imp) for imp in relevant_imports)

    combined = (imports_context + "\n" + content) if imports_context else content
    token_count = _count_tokens(encoder, combined)

    breadcrumbs = build_breadcrumbs(file_path, symbol, all_symbols)

    return CodeChunk(
        id=_make_chunk_id(file_path, symbol.name if symbol else None, start_line),
        symbol_id=_make_symbol_id(
            file_path,
            symbol.qualified_name if symbol else None,
            symbol.kind if symbol else None,
        ),
        qualified_name=symbol.qualified_name if symbol else None,
        visibility=str(symbol.visibility) if symbol else "public",
        signature=symbol.signature if symbol else None,
        docstring=symbol.docstring if symbol else None,
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        symbol_name=symbol.name if symbol else None,
        symbol_kind=symbol.kind if symbol else None,
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

        # Sort symbols by start_line; methods nested under classes are already separate.
        symbols = sorted(parsed_file.symbols, key=lambda s: s.start_line)

        # Collect raw chunk candidates: (lines_list, start_line, symbol_or_None).
        candidates: list[tuple[list[bytes], int, Symbol | None]] = []
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

        # File-level code: lines not covered by symbols or chunk-only AST ranges.
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

        # Merge small adjacent file-level chunks. Chunk-only AST ranges must remain
        # line-stable; merging them can combine non-contiguous source ranges.
        if has_structural_ranges:
            merged = sorted(candidates, key=lambda item: item[1])
        else:
            merged = _merge_small_chunks(candidates, config.chunk_min_tokens, encoder)

        # Build CodeChunk objects
        result: list[CodeChunk] = []
        for lines_group, start_line, sym in merged:
            if _is_blank_lines(lines_group):
                continue
            chunk = _build_chunk(
                file_path=file_path,
                language=language,
                lines=lines_group,
                start_line=start_line,
                symbol=sym,
                imports=imports,
                encoder=encoder,
                all_symbols=symbols,
            )
            result.append(chunk)

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


def _merge_small_chunks(
    candidates: list[tuple[list[bytes], int, Symbol | None]],
    min_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[tuple[list[bytes], int, Symbol | None]]:
    """Merge adjacent small (below min_tokens) file-level chunks."""
    if not candidates:
        return []

    result: list[tuple[list[bytes], int, Symbol | None]] = []
    i = 0
    while i < len(candidates):
        lines, start, sym = candidates[i]
        tokens = _count_tokens(encoder, _lines_to_text(lines))

        # Only merge file-level (sym is None) small chunks with adjacent file-level chunks
        if tokens < min_tokens and sym is None:
            # Try to merge with next file-level chunk
            if i + 1 < len(candidates) and candidates[i + 1][2] is None:
                next_lines: list[bytes] = candidates[i + 1][0]
                merged_fwd: list[bytes] = lines + next_lines
                candidates[i + 1] = (merged_fwd, start, None)
                i += 1
                continue
            # Try to merge into previous result if it's also file-level
            if result and result[-1][2] is None:
                prev_entry = result[-1]
                prev_lines: list[bytes] = prev_entry[0]
                prev_start: int = prev_entry[1]
                merged_back: list[bytes] = prev_lines + lines
                result[-1] = (merged_back, prev_start, None)
                i += 1
                continue

        result.append((lines, start, sym))
        i += 1

    return result
