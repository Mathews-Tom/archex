"""Annotate an agent's own search results with the indexed code units they hit.

Host-neutral core of the grep-result annotation hook. A host adapter (the
omp/Pi extension module, or a user at a shell via `archex annotate`) passes
the tool name, the tool's input, and the tool's model-facing result text.
This module parses the hits out of that text, maps every ``(path, line)`` hit
to the smallest indexed code unit containing it, and renders one fact line per
distinct unit:

    [archex] src/pkg/service.py::Service.authenticate method L120-168 · importers 2

Contract:

- **Augment, never replace.** The output is only the lines to append; the
  caller keeps the original result byte-for-byte.
- **Deterministic and local.** The same index and the same hits give the same
  text: no timestamps, no model calls, no network.
- **Bounded.** One line per distinct unit, about `MAX_LINE_TOKENS` tokens per
  line and `MAX_CALL_TOKENS` per call (archex's cl100k tokenizer); units past
  the cap collapse into ``+N more units``. A unit whose whole span is already
  visible in the result gets no line.
- **Fresh index only.** Anything but a `fresh` index (the lifecycle check the
  status command reports) yields no annotation.
- **Fail open.** An unrecognised format, an unindexed path, or any error
  yields no annotation; the reason is recorded, never raised.

Degree is the file's importer count from the index's import edges. The index
persists no call edges, so no caller count is rendered.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from archex.index.store import IndexStore
from archex.receipt import index_revision_from_store
from archex.reporting import count_tokens
from archex.status import inspect_index_freshness

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from archex.index.store import ChunkSpan

AnnotateTool = Literal["grep", "glob", "find", "bash"]
ANNOTATE_TOOLS: tuple[AnnotateTool, ...] = ("grep", "glob", "find", "bash")

MAX_LINE_TOKENS = 30
MAX_CALL_TOKENS = 300

LINE_PREFIX = "[archex]"
RECEIPT_PREFIX = "[archex receipt]"

# --- omp tool output shapes (derived from local omp session transcripts) ---
#
# grep, one file:   "[path#1A2B]" then " 5:context" / "*6:match" lines
# grep, many files: "# dir/" / "## file.py#1A2B" headers (depth = number of
#                   '#', a trailing '/' marks a directory) then " 514|" /
#                   "*515|" lines; a depth-1 header may name a file directly
# grep, bare:       line entries with no header; the file is the input `path`
# glob:             the same header tree, with bare file names under it; names
#                   before the first header sit in the search root
# Hashline anchors ("#1A2B") are four upper-case hex digits after the name.
_TREE_HEADER = re.compile(r"^(?P<depth>#{1,16}) (?P<name>\S.*)$")
_SINGLE_HEADER = re.compile(r"^\[(?P<name>[^\]\s]+)\]$")
_GREP_LINE = re.compile(r"^(?P<mark>[ *])(?P<line>\d+)[:|]")
_HASHLINE_TAG = re.compile(r"#[0-9A-F]{4}$")
_GREP_NO_MATCHES = "No matches found"
_GLOB_NO_MATCHES = "No files found matching pattern"
_PATH_GLOB_CHARS = re.compile(r"[*?\[{]")

# --- bash search output ---
#
# plain:    `path:line:text` matches, `path-line-text` context (rg -n,
#           grep -rn, git grep -n)
# grouped:  a condensed view seen in omp bash results (which component
#           produced it is not pinned): a "grep: 6 matches in 2 files" line,
#           then per file a "path:" line and indented "  315: text" entries
# empty:    the omp bash tool prints "(no output)" when a search matched nothing
_PATH_LINE_MATCH = re.compile(r"^(?P<path>[^:\n]+):(?P<line>\d+):")
_PATH_LINE_CONTEXT = re.compile(r"^(?P<path>.+?)-(?P<line>\d+)-")
_GROUPED_SUMMARY = re.compile(r"^\S+: \d+ match(?:es)? in \d+ files?$")
_GROUPED_FILE = re.compile(r"^(?P<path>\S[^:]*):$")
_GROUPED_ENTRY = re.compile(r"^\s+(?P<line>\d+)(?P<sep>[:-]) ")
_BASH_NO_OUTPUT = "(no output)"
_SEARCH_PROGRAMS = frozenset({"rg", "grep", "egrep", "fgrep"})
_SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", "&", ";;", "(", ")", "|&"})


@dataclass
class _FileHits:
    path: str
    matches: list[int] = field(default_factory=list[int])
    visible: set[int] = field(default_factory=set[int])


@dataclass(frozen=True)
class ParsedSearch:
    """Hits recovered from one tool result.

    ``kind == "lines"`` carries matched line numbers per file (grep and bash
    search); ``kind == "files"`` carries only file paths (glob and find).
    """

    format: str
    kind: Literal["lines", "files"]
    files: tuple[_FileHits, ...]
    base_dir: str = ""


@dataclass(frozen=True)
class Annotation:
    """Outcome of one annotation request; `text` is empty when nothing is added."""

    text: str
    eligible: bool
    annotated: bool
    units_hit: int
    units_rendered: int
    tokens: int
    freshness: str
    reason: str | None
    format: str | None
    index_revision: str = ""
    capped: bool = False
    out_of_range_hits: int = 0

    def as_record(self) -> dict[str, object]:
        return {
            "annotation": self.text,
            "eligible": self.eligible,
            "annotated": self.annotated,
            "units_hit": self.units_hit,
            "units_rendered": self.units_rendered,
            "tokens": self.tokens,
            "freshness": self.freshness,
            "reason": self.reason,
            "format": self.format,
            "index_revision": self.index_revision,
            "capped": self.capped,
            "out_of_range_hits": self.out_of_range_hits,
        }


def _decline(
    reason: str,
    *,
    eligible: bool = True,
    freshness: str = "unchecked",
    fmt: str | None = None,
    units_hit: int = 0,
    out_of_range_hits: int = 0,
) -> Annotation:
    return Annotation(
        text="",
        eligible=eligible,
        annotated=False,
        units_hit=units_hit,
        units_rendered=0,
        tokens=0,
        freshness=freshness,
        reason=reason,
        format=fmt,
        out_of_range_hits=out_of_range_hits,
    )


# --- Parsing ---------------------------------------------------------------


def bash_search_base(command: str) -> str | None:
    """Directory a bash search command runs in, or `None` if it is not a search.

    A search is a command whose first program is `rg`, `grep`/`egrep`/`fgrep`,
    or `git grep`, optionally after one leading ``cd <dir> &&``. The returned
    string is that directory ("" when there is none), against which printed
    relative paths resolve.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    base = ""
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
        base = tokens[1]
        tokens = tokens[3:]
    first: list[str] = []
    for token in tokens:
        if token in _SHELL_OPERATORS:
            break
        first.append(token)
    if not first:
        return None
    program = os.path.basename(first[0])
    if program in _SEARCH_PROGRAMS:
        return base
    if program == "git" and len(first) > 1 and first[1] == "grep":
        return base
    return None


def _single_input_path(tool_input: Mapping[str, object]) -> str | None:
    raw = tool_input.get("path")
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate or ";" in candidate or _PATH_GLOB_CHARS.search(candidate):
        return None
    return candidate


def _strip_tag(name: str) -> str:
    return _HASHLINE_TAG.sub("", name.strip())


def _parse_omp_tree(
    lines: list[str], *, entries: Literal["lines", "files"]
) -> list[_FileHits] | None:
    """Parse the omp grep/glob header tree; `None` if a line breaks the shape."""
    stack: list[str] = []
    files: dict[str, _FileHits] = {}
    current: _FileHits | None = None
    for raw in lines:
        header = _TREE_HEADER.match(raw)
        if header is not None:
            depth = len(header.group("depth"))
            if depth > len(stack) + 1:
                return None
            name = header.group("name").strip()
            if name.endswith("/"):
                stack = [*stack[: depth - 1], name]
                current = None
                continue
            path = "".join(stack[: depth - 1]) + _strip_tag(name)
            stack = stack[: depth - 1]
            current = files.setdefault(path, _FileHits(path))
            continue
        if entries == "lines":
            entry = _GREP_LINE.match(raw)
            if entry is None or current is None:
                continue
            line_no = int(entry.group("line"))
            current.visible.add(line_no)
            if entry.group("mark") == "*":
                current.matches.append(line_no)
            continue
        name = raw.strip()
        if not name or name.startswith("[") or name.endswith("/"):
            continue
        path = "".join(stack) + name
        files.setdefault(path, _FileHits(path))
    return list(files.values())


def _parse_grep_entries(lines: list[str], path: str) -> _FileHits:
    hits = _FileHits(path)
    for raw in lines:
        entry = _GREP_LINE.match(raw)
        if entry is None:
            continue
        line_no = int(entry.group("line"))
        hits.visible.add(line_no)
        if entry.group("mark") == "*":
            hits.matches.append(line_no)
    return hits


def _parse_path_line(lines: list[str]) -> list[_FileHits]:
    files: dict[str, _FileHits] = {}
    for raw in lines:
        match = _PATH_LINE_MATCH.match(raw)
        if match is None:
            continue
        hits = files.setdefault(match.group("path"), _FileHits(match.group("path")))
        line_no = int(match.group("line"))
        hits.matches.append(line_no)
        hits.visible.add(line_no)
    for raw in lines:
        context = _PATH_LINE_CONTEXT.match(raw)
        if context is not None and context.group("path") in files:
            files[context.group("path")].visible.add(int(context.group("line")))
    return list(files.values())


def _parse_grouped(lines: list[str]) -> list[_FileHits]:
    files: dict[str, _FileHits] = {}
    current: _FileHits | None = None
    for raw in lines[1:]:
        header = _GROUPED_FILE.match(raw)
        if header is not None:
            path = header.group("path")
            current = files.setdefault(path, _FileHits(path))
            continue
        entry = _GROUPED_ENTRY.match(raw)
        if entry is None or current is None:
            continue
        line_no = int(entry.group("line"))
        current.visible.add(line_no)
        if entry.group("sep") == ":":
            current.matches.append(line_no)
    return [hits for hits in files.values() if hits.matches]


def parse_search_result(
    tool: str, tool_input: Mapping[str, object], text: str
) -> ParsedSearch | str:
    """Recover hits from a tool result, or return the reason it can't be read."""
    lines = text.split("\n")
    first = next((line for line in lines if line.strip()), "")
    if tool == "grep":
        if first.strip() == _GREP_NO_MATCHES:
            return ParsedSearch("omp-grep", "lines", ())
        if _TREE_HEADER.match(first):
            tree = _parse_omp_tree(lines, entries="lines")
            if tree is None:
                return "unrecognized_format"
            return ParsedSearch("omp-grep-tree", "lines", tuple(tree))
        single = _SINGLE_HEADER.match(first)
        if single is not None:
            path = _strip_tag(single.group("name"))
            return ParsedSearch("omp-grep-file", "lines", (_parse_grep_entries(lines, path),))
        if _GREP_LINE.match(first):
            path = _single_input_path(tool_input)
            if path is None:
                return "unrecognized_format"
            return ParsedSearch("omp-grep-bare", "lines", (_parse_grep_entries(lines, path),))
        return "unrecognized_format"
    if tool in ("glob", "find"):
        if first.strip() == _GLOB_NO_MATCHES:
            return ParsedSearch("omp-glob", "files", ())
        if any(_TREE_HEADER.match(line) for line in lines):
            tree = _parse_omp_tree(lines, entries="files")
            if tree is None:
                return "unrecognized_format"
            return ParsedSearch("omp-glob-tree", "files", tuple(tree))
        names = [line.strip() for line in lines if line.strip()]
        if names and all(" " not in name and not name.startswith("[") for name in names):
            paths = tuple(_FileHits(name) for name in names if not name.endswith("/"))
            return ParsedSearch("path-list", "files", paths)
        return "unrecognized_format"
    if tool == "bash":
        command = tool_input.get("command")
        if not isinstance(command, str):
            return "not_search_command"
        base = bash_search_base(command)
        if base is None:
            return "not_search_command"
        if first.strip() == _BASH_NO_OUTPUT:
            return ParsedSearch("path-line", "lines", (), base_dir=base)
        if _GROUPED_SUMMARY.match(first.strip()):
            grouped = _parse_grouped(lines[lines.index(first) :])
            return ParsedSearch("grouped", "lines", tuple(grouped), base_dir=base)
        files = _parse_path_line(lines)
        if not files:
            return "unrecognized_format" if first.strip() else "no_hits"
        return ParsedSearch("path-line", "lines", tuple(files), base_dir=base)
    return "unsupported_tool"


# --- Resolution --------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    key: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int


@dataclass
class _FileUnits:
    units: list[_Unit]
    line_count: int
    top_level: int


def _unit_key(span: ChunkSpan) -> str:
    base = span.symbol_id or f"{span.file_path}::{span.qualified_name}#{span.symbol_kind}"
    return re.sub(r"@\d+$", "", base)


def _line_count(path: Path) -> int | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data.count(b"\n") + (0 if not data or data.endswith(b"\n") else 1)


def _build_file_units(spans: Iterable[ChunkSpan], repo_root: Path) -> dict[str, _FileUnits]:
    """Group each indexed file's chunks into code units.

    A unit is one symbol: its chunks share a symbol id up to the ``@N`` suffix
    the chunker adds when it splits a long symbol, so the unit's span is the
    union of its parts. Chunks without a symbol kind are module-level text
    and form no unit. Files missing on disk are dropped.
    """
    grouped: dict[str, dict[str, list[ChunkSpan]]] = {}
    for span in spans:
        by_key = grouped.setdefault(span.file_path, {})
        if span.symbol_kind is not None:
            by_key.setdefault(_unit_key(span), []).append(span)
    result: dict[str, _FileUnits] = {}
    for path, by_key in grouped.items():
        line_count = _line_count(repo_root / path)
        if line_count is None:
            continue
        units = [
            _Unit(
                key=key,
                qualified_name=parts[0].qualified_name or parts[0].symbol_name or "?",
                kind=str(parts[0].symbol_kind),
                start_line=min(part.start_line for part in parts),
                end_line=max(part.end_line for part in parts),
            )
            for key, parts in by_key.items()
        ]
        units.sort(key=lambda unit: (unit.start_line, -unit.end_line, unit.key))
        top_level = 0
        covered_to = 0
        for unit in units:
            if unit.start_line > covered_to:
                top_level += 1
                covered_to = unit.end_line
        result[path] = _FileUnits(units=units, line_count=line_count, top_level=top_level)
    return result


def _containing_unit(units: list[_Unit], line: int) -> _Unit | None:
    best: _Unit | None = None
    for unit in units:
        if unit.start_line <= line <= unit.end_line and (
            best is None
            or (unit.end_line - unit.start_line, -unit.start_line)
            < (best.end_line - best.start_line, -best.start_line)
        ):
            best = unit
    return best


def _repo_relative(printed: str, base_dir: Path, repo_root: Path) -> str | None:
    resolved = Path(os.path.realpath(base_dir / printed))
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return None


# --- Rendering ---------------------------------------------------------------


def _shorten(line_parts: tuple[str, str, str], tokens_of: dict[str, int]) -> str:
    """Render one line, trimming the path then the name if it exceeds the line cap."""
    path, name, rest = line_parts

    def render(p: str, n: str) -> str:
        head = f"{LINE_PREFIX} {p}::{n}" if n else f"{LINE_PREFIX} {p}"
        return f"{head} {rest}"

    line = render(path, name)
    if _tokens(line, tokens_of) <= MAX_LINE_TOKENS:
        return line
    segments = path.split("/")
    if len(segments) > 2:
        path = "…/" + "/".join(segments[-2:])
        line = render(path, name)
        if _tokens(line, tokens_of) <= MAX_LINE_TOKENS:
            return line
    dotted = name.split(".")
    if len(dotted) > 2:
        line = render(path, "…." + ".".join(dotted[-2:]))
    return line


def _tokens(text: str, cache: dict[str, int]) -> int:
    if text not in cache:
        cache[text] = count_tokens(text)
    return cache[text]


def _render(
    lines: list[str], units_hit: int, revision: str, cache: dict[str, int]
) -> tuple[str, int, int, bool]:
    header = f"{RECEIPT_PREFIX} index_revision={revision[:12]} units={units_hit}"
    kept: list[str] = [header]
    used = _tokens(header + "\n", cache)
    for index, line in enumerate(lines):
        remaining = len(lines) - index - 1
        reserve = _tokens(f"{LINE_PREFIX} +{remaining} more units\n", cache) if remaining else 0
        cost = _tokens(line + "\n", cache)
        if used + cost + reserve > MAX_CALL_TOKENS:
            more = f"{LINE_PREFIX} +{len(lines) - index} more units"
            kept.append(more)
            text = "\n".join(kept)
            return text, index, count_tokens(text), True
        kept.append(line)
        used += cost
    text = "\n".join(kept)
    return text, len(lines), count_tokens(text), False


# --- Entry point ---------------------------------------------------------------


def annotate(
    tool: str,
    tool_input: Mapping[str, object],
    text: str,
    cwd: str | Path,
) -> Annotation:
    """Annotation for one tool result; never raises for expected declines."""
    if tool not in ANNOTATE_TOOLS:
        return _decline("unsupported_tool", eligible=False)
    parsed = parse_search_result(tool, tool_input, text)
    if isinstance(parsed, str):
        return _decline(parsed, eligible=parsed != "not_search_command")
    if not parsed.files:
        return _decline("no_hits", fmt=parsed.format)

    freshness = inspect_index_freshness(cwd)
    if freshness.state != "fresh":
        return _decline("index_not_fresh", freshness=freshness.state, fmt=parsed.format)
    return annotate_parsed(
        parsed,
        cwd=cwd,
        repo_root=freshness.repo_root,
        index_path=freshness.index_path,
        freshness="fresh",
    )


def annotate_parsed(
    parsed: ParsedSearch,
    *,
    cwd: str | Path,
    repo_root: Path,
    index_path: Path,
    freshness: str,
) -> Annotation:
    """Resolve parsed hits against the index at ``index_path`` and render them.

    Performs no freshness check: `annotate` gates on it; offline replays that
    measure against a repository's current index call this directly.
    """
    root = Path(os.path.realpath(repo_root))
    base_dir = Path(os.path.realpath(Path(cwd).expanduser()))
    if parsed.base_dir:
        base_dir = Path(os.path.realpath(base_dir / Path(parsed.base_dir).expanduser()))

    resolved: list[tuple[str, _FileHits]] = []
    for hits in parsed.files:
        rel = _repo_relative(hits.path, base_dir, root)
        if rel is not None:
            resolved.append((rel, hits))
    if not resolved:
        return _decline("no_indexed_paths", freshness=freshness, fmt=parsed.format)

    paths = sorted({rel for rel, _ in resolved})
    store = IndexStore(index_path)
    try:
        file_units = _build_file_units(store.get_chunk_spans_for_files(paths), root)
        importers = store.count_importers(paths)
        revision = index_revision_from_store(store)
    finally:
        store.close()

    rendered: list[str] = []
    seen: set[str] = set()
    units_hit = 0
    out_of_range = 0
    cache: dict[str, int] = {}
    for rel, hits in resolved:
        units = file_units.get(rel)
        if units is None or not units.units:
            continue
        degree = f"importers {importers.get(rel, 0)}"
        if parsed.kind == "files":
            if rel in seen:
                continue
            seen.add(rel)
            units_hit += 1
            rendered.append(_shorten((rel, "", f"· units {units.top_level} · {degree}"), cache))
            continue
        for line in hits.matches:
            if line < 1 or line > units.line_count:
                out_of_range += 1
                continue
            unit = _containing_unit(units.units, line)
            key = f"{rel}::<module>" if unit is None else unit.key
            if key in seen:
                continue
            seen.add(key)
            units_hit += 1
            if unit is None:
                rest = f"module-level · units {units.top_level} · {degree}"
                rendered.append(_shorten((rel, "", rest), cache))
                continue
            span = range(unit.start_line, unit.end_line + 1)
            if all(line_no in hits.visible for line_no in span):
                continue
            rest = f"{unit.kind} L{unit.start_line}-{unit.end_line} · {degree}"
            rendered.append(_shorten((rel, unit.qualified_name, rest), cache))

    if units_hit == 0:
        reason = "hits_out_of_range" if out_of_range else "no_code_units"
        return _decline(
            reason, freshness=freshness, fmt=parsed.format, out_of_range_hits=out_of_range
        )
    if not rendered:
        return _decline(
            "all_units_visible",
            freshness=freshness,
            fmt=parsed.format,
            units_hit=units_hit,
            out_of_range_hits=out_of_range,
        )
    block, kept, tokens, capped = _render(rendered, units_hit, revision, cache)
    return Annotation(
        text=block,
        eligible=True,
        annotated=True,
        units_hit=units_hit,
        units_rendered=kept,
        tokens=tokens,
        freshness=freshness,
        reason=None,
        format=parsed.format,
        index_revision=revision,
        capped=capped,
        out_of_range_hits=out_of_range,
    )
