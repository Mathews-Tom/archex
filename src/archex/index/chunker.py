"""Token-aware code chunker: split ParsedFile symbols into bounded CodeChunks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import tiktoken

from archex.models import CodeChunk, ImportStatement, IndexConfig, ParsedFile, Symbol


@runtime_checkable
class Chunker(Protocol):
    """Protocol for code chunkers, allowing custom implementations."""

    def chunk_file(self, parsed_file: ParsedFile, source: bytes) -> list[CodeChunk]: ...

    def chunk_files(
        self, parsed_files: list[ParsedFile], sources: dict[str, bytes]
    ) -> list[CodeChunk]: ...


def _count_tokens(encoder: tiktoken.Encoding, text: str) -> int:
    return len(encoder.encode(text))


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


def _extract_source_lines(all_lines: list[bytes], start_line: int, end_line: int) -> list[bytes]:
    """Extract 1-indexed [start_line, end_line] from pre-split line list."""
    lo = max(0, start_line - 1)
    hi = min(len(all_lines), end_line)
    return all_lines[lo:hi]


def _make_chunk_id(file_path: str, symbol_name: str | None, start_line: int) -> str:
    name = symbol_name if symbol_name is not None else "_module"
    return f"{file_path}:{name}:{start_line}"


def _lines_to_text(lines: list[bytes]) -> str:
    return "\n".join(line.decode("utf-8", errors="replace") for line in lines)


def _build_chunk(
    *,
    file_path: str,
    language: str,
    lines: list[bytes],
    start_line: int,
    symbol: Symbol | None,
    imports: list[ImportStatement],
    encoder: tiktoken.Encoding,
) -> CodeChunk:
    content = _lines_to_text(lines)

    # Filter imports relevant to this chunk
    relevant_imports = [imp for imp in imports if _import_relevant(imp, content)]
    imports_context = "\n".join(_format_import(imp) for imp in relevant_imports)

    combined = (imports_context + "\n" + content) if imports_context else content
    token_count = _count_tokens(encoder, combined)

    return CodeChunk(
        id=_make_chunk_id(file_path, symbol.name if symbol else None, start_line),
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        symbol_name=symbol.name if symbol else None,
        symbol_kind=symbol.kind if symbol else None,
        language=language,
        imports_context=imports_context,
        token_count=token_count,
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

        # Sort symbols by start_line; methods nested under classes are already separate
        symbols = sorted(parsed_file.symbols, key=lambda s: s.start_line)

        # Track which lines are covered by symbols so we can emit file-level chunks
        covered: list[tuple[int, int]] = [(s.start_line, s.end_line) for s in symbols]

        # Collect raw chunk candidates: (lines_list, start_line, symbol_or_None)
        candidates: list[tuple[list[bytes], int, Symbol | None]] = []

        for sym in symbols:
            sym_lines = _extract_source_lines(all_source_lines, sym.start_line, sym.end_line)
            if not sym_lines:
                continue

            tokens = _count_tokens(encoder, _lines_to_text(sym_lines))

            if tokens > config.chunk_max_tokens:
                sub_groups = _split_lines_at_boundary(sym_lines, config.chunk_max_tokens, encoder)
                offset = sym.start_line
                for group in sub_groups:
                    candidates.append((group, offset, sym))
                    offset += len(group)
            else:
                candidates.append((sym_lines, sym.start_line, sym))

        # File-level code: lines not covered by any symbol
        uncovered_ranges = _find_uncovered_ranges(covered, total_lines)
        for range_start, range_end in uncovered_ranges:
            fl_lines = _extract_source_lines(all_source_lines, range_start, range_end)
            if not fl_lines or all(line.strip() == b"" for line in fl_lines):
                continue
            tokens = _count_tokens(encoder, _lines_to_text(fl_lines))
            if tokens > config.chunk_max_tokens:
                sub_groups = _split_lines_at_boundary(fl_lines, config.chunk_max_tokens, encoder)
                offset = range_start
                for group in sub_groups:
                    candidates.append((group, offset, None))
                    offset += len(group)
            else:
                candidates.append((fl_lines, range_start, None))

        # Merge small adjacent file-level chunks
        merged = _merge_small_chunks(candidates, config.chunk_min_tokens, encoder)

        # Build CodeChunk objects
        result: list[CodeChunk] = []
        for lines_group, start_line, sym in merged:
            if not lines_group or all(line.strip() == b"" for line in lines_group):
                continue
            chunk = _build_chunk(
                file_path=file_path,
                language=language,
                lines=lines_group,
                start_line=start_line,
                symbol=sym,
                imports=imports,
                encoder=encoder,
            )
            result.append(chunk)

        result.sort(key=lambda c: c.start_line)
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

    sorted_covered = sorted(covered)
    uncovered: list[tuple[int, int]] = []

    prev_end = 0
    for start, end in sorted_covered:
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
