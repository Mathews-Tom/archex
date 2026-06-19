"""Deterministic, low-risk post-retrieval content compression primitives.

These operate purely on already-assembled returned-region text. They never call
hosted services, local models, or semantic summarizers; every mode is a
deterministic text transform that preserves stable anchors and points back to
the original via an explicit ``fetch original`` marker. Required, direct, and
high-confidence regions pass through unchanged.

The primitives here are content-only. Token counts and content hashes for the
:class:`~archex.models.CompressionMetadata` record are computed by the caller,
where the file path and line range are known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from archex.models import CompressionLossRisk, CompressionMode
from archex.reporting import count_tokens

# A run of at least this many non-blank lines is worth replacing with a marker.
_ELISION_MIN_RUN = 2

# Large-literal summarization keeps this many head/tail lines and only fires on
# blocks of at least this many lines that look mostly like data entries.
_LITERAL_MIN_LINES = 12
_LITERAL_HEAD = 3
_LITERAL_TAIL = 3
_LITERAL_ITEM_FRACTION = 0.5

# JSON/log crushing keeps head/tail plus every anomaly line, on blocks of at
# least this many lines.
_JSONLOG_MIN_LINES = 12
_JSONLOG_HEAD = 3
_JSONLOG_TAIL = 3

_MARKER_TOKEN = "[archex"

_HASH_COMMENT_LANGUAGES = frozenset(
    {"python", "ruby", "shell", "bash", "sh", "yaml", "yml", "toml", "r", "perl"}
)
_DATA_LANGUAGES = frozenset(
    {"json", "jsonl", "ndjson", "yaml", "yml", "toml", "log", "text", "txt", "csv", "tsv"}
)

_IMPORT_PREFIXES = (
    "import ",
    "from ",
    "#include",
    "using ",
    "package ",
    "use ",
    "require ",
)
_DECL_PREFIXES = (
    "def ",
    "async def ",
    "class ",
    "function ",
    "func ",
    "fn ",
    "interface ",
    "type ",
    "struct ",
    "impl ",
    "trait ",
    "enum ",
    "module ",
    "namespace ",
    "export ",
    "public ",
    "private ",
    "protected ",
    "internal ",
    "abstract ",
    "const ",
    "let ",
    "var ",
    "static ",
    "final ",
    "override ",
)
_COMMENT_PREFIXES = ("#", "//", "/*", "*")
_BLOCK_OPENERS = (":", "{", "(", "[")
_BLOCK_CLOSERS = ("}", ")", "]", "};", "});", "),")
# Block-comment delimiters that look like decoration but must not be dropped, to
# avoid leaving orphaned ``* text`` interior lines without their open/close.
_BLOCK_COMMENT_DELIMITERS = frozenset({"/*", "*/", "/**", "**/"})

# A comment line made only of separator/decoration characters (banner rules).
_DECORATION_COMMENT_RE = re.compile(r"^[#/*\-=_~. ]+$")
# Anomaly tokens that must survive json/log crushing.
_ANOMALY_RE = re.compile(
    r"\b(error|err|warn|warning|exception|traceback|fail|failed|failure|fatal|"
    r"critical|panic|denied|timeout|refused|unauthorized)\b",
    re.IGNORECASE,
)
# Cheap code-signal fallback when language is unknown.
_CODE_SIGNAL_RE = re.compile(r"(^|\n)\s*(def |class |function |func |fn |import |#include)")
# key: value or key = value data lines for literal detection.
_KEYVAL_RE = re.compile(r"""^\s*["']?[\w.\-]+["']?\s*[:=]\s*\S""")
# Timestamp / level evidence for log-line detection.
_LOG_LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})|(\b(DEBUG|INFO|WARN|WARNING|ERROR|TRACE|FATAL)\b)"
)

_MODE_PRIORITY = (
    CompressionMode.STRUCTURAL_CODE_ELISION,
    CompressionMode.LARGE_LITERAL_SUMMARIZATION,
    CompressionMode.JSON_LOG_SMART_CRUSHING,
    CompressionMode.COMMENT_AND_WHITESPACE_SLIMMING,
)
_MODE_RISK = {
    CompressionMode.PASSTHROUGH_REQUIRED: CompressionLossRisk.NONE,
    CompressionMode.STRUCTURAL_CODE_ELISION: CompressionLossRisk.LOW,
    CompressionMode.COMMENT_AND_WHITESPACE_SLIMMING: CompressionLossRisk.LOW,
    CompressionMode.LARGE_LITERAL_SUMMARIZATION: CompressionLossRisk.MEDIUM,
    CompressionMode.JSON_LOG_SMART_CRUSHING: CompressionLossRisk.MEDIUM,
}


@dataclass(frozen=True)
class RegionCompression:
    """Result of compressing one returned region.

    ``content`` is the text to display in place of the original. For
    ``passthrough_required`` it equals the input unchanged.
    """

    mode: CompressionMode
    content: str
    loss_risk: CompressionLossRisk


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _comment_prefix(language: str | None) -> str:
    if language and language.lower() in _HASH_COMMENT_LANGUAGES:
        return "#"
    return "//"


def _is_comment(stripped: str) -> bool:
    return stripped.startswith(_COMMENT_PREFIXES)


def _looks_like_code(language: str | None, content: str) -> bool:
    lang = (language or "").lower()
    if lang in _DATA_LANGUAGES:
        return False
    if lang:
        # Any known, non-empty language that is not a data format is treated as
        # code so compression fails safe toward code protection: literal/JSON-log
        # modes never fire on code, and protect_code stays effective for
        # shell/sql/lua and any other source language.
        return True
    return bool(_CODE_SIGNAL_RE.search(content))


def _looks_like_json(content: str) -> bool:
    stripped = content.strip()
    return stripped.startswith(("{", "[")) and ('":' in stripped or ": " in stripped)


def _looks_like_log(lines: list[str]) -> bool:
    hits = sum(1 for line in lines if _LOG_LINE_RE.search(line))
    return hits >= max(3, len(lines) // 4)


def _is_structural_line(stripped: str) -> bool:
    """Whether a line should survive structural code elision."""
    if not stripped:
        return False
    if stripped.startswith("@"):  # decorator / annotation
        return True
    if _is_comment(stripped):
        return True
    if stripped.startswith(_IMPORT_PREFIXES):
        return True
    if stripped.startswith(_DECL_PREFIXES):
        return True
    if stripped.endswith(_BLOCK_OPENERS):  # signature / block opener
        return True
    return stripped in _BLOCK_CLOSERS


def _elision_marker(indent: str, prefix: str, line_count: int, handle: str) -> str:
    detail = f"elided {line_count} line(s); fetch original: {handle}"
    return f"{indent}{prefix} {_MARKER_TOKEN} {detail}]"


def _omission_marker(indent: str, unit: str, count: int, handle: str, *, suffix: str = "") -> str:
    detail = f"{count} {unit} omitted{suffix}"
    return f"{indent}... {_MARKER_TOKEN} {detail}; fetch original: {handle}] ..."


def _elide_structural(content: str, *, language: str | None, handle: str) -> str | None:
    """Keep imports/declarations/signatures/anchors, elide low-risk bodies."""
    prefix = _comment_prefix(language)
    lines = content.split("\n")
    out: list[str] = []
    body_run: list[str] = []

    def flush() -> None:
        if not body_run:
            return
        nonblank = [line for line in body_run if line.strip()]
        if not nonblank:
            out.append("")  # collapse a pure-blank run to a single blank line
        elif len(nonblank) >= _ELISION_MIN_RUN:
            out.append(_elision_marker(_leading_ws(nonblank[0]), prefix, len(body_run), handle))
        else:
            out.extend(body_run)
        body_run.clear()

    for line in lines:
        if _is_structural_line(line.strip()):
            flush()
            out.append(line)
        else:
            body_run.append(line)
    flush()

    new = "\n".join(out)
    return new if new != content else None


def _slim_comments_whitespace(content: str) -> str | None:
    """Collapse repeated blank lines, trim trailing whitespace, drop banners."""
    lines = content.split("\n")
    out: list[str] = []
    previous_blank = False
    for line in lines:
        trimmed = line.rstrip()
        stripped = trimmed.strip()
        if (
            stripped
            and _is_comment(stripped)
            and stripped not in _BLOCK_COMMENT_DELIMITERS
            and _DECORATION_COMMENT_RE.match(stripped)
        ):
            continue  # banner / separator comment with no information
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            out.append("")
        else:
            previous_blank = False
            out.append(trimmed)
    new = "\n".join(out)
    return new if new != content else None


def _summarize_large_literal(content: str, *, handle: str) -> str | None:
    """Keep first/last items and the omitted count for a long data literal.

    Defers to :func:`_crush_json_log` for JSON objects/arrays and log streams so
    each mode owns a distinct content shape.
    """
    lines = content.split("\n")
    if len(lines) < _LITERAL_MIN_LINES:
        return None
    if _looks_like_json(content) or _looks_like_log(lines):
        return None
    item_like = sum(
        1 for line in lines if line.rstrip().endswith((",", "[", "{")) or _KEYVAL_RE.match(line)
    )
    if item_like < len(lines) * _LITERAL_ITEM_FRACTION:
        return None
    head = lines[:_LITERAL_HEAD]
    tail = lines[-_LITERAL_TAIL:]
    omitted = len(lines) - len(head) - len(tail)
    if omitted < _ELISION_MIN_RUN:
        return None
    indent = _leading_ws(lines[_LITERAL_HEAD])
    marker = _omission_marker(
        indent, "item(s)", omitted, handle, suffix=f" (kept first {len(head)} / last {len(tail)})"
    )
    new = "\n".join([*head, marker, *tail])
    return new if new != content else None


def _crush_json_log(content: str, *, handle: str) -> str | None:
    """Preserve anomalies, first/last examples, and omitted counts.

    Fires only when the content actually looks like a JSON object/array or a log
    stream. Plain sequence/table literals are handled by
    :func:`_summarize_large_literal` instead.
    """
    lines = content.split("\n")
    if len(lines) < _JSONLOG_MIN_LINES:
        return None
    if not _looks_like_json(content) and not _looks_like_log(lines):
        return None
    keep = set(range(min(_JSONLOG_HEAD, len(lines))))
    keep |= set(range(max(0, len(lines) - _JSONLOG_TAIL), len(lines)))
    for index, line in enumerate(lines):
        if _ANOMALY_RE.search(line):
            keep.add(index)
    out: list[str] = []
    omitted_total = 0
    index = 0
    while index < len(lines):
        if index in keep:
            out.append(lines[index])
            index += 1
            continue
        run_start = index
        while index < len(lines) and index not in keep:
            index += 1
        run = index - run_start
        omitted_total += run
        out.append(_omission_marker(_leading_ws(lines[run_start]), "line(s)", run, handle))
    if omitted_total < _ELISION_MIN_RUN:
        return None
    new = "\n".join(out)
    return new if new != content else None


def compress_region(
    content: str,
    *,
    language: str | None,
    fetch_original_handle: str,
    required: bool,
    protect_code: bool = False,
) -> RegionCompression | None:
    """Compress one returned region deterministically, or return ``None``.

    Required/direct/high-confidence regions always pass through unchanged. Other
    regions are run through the applicable deterministic modes; the smallest
    strictly-beneficial result wins (ties broken by a fixed mode priority). When
    no mode yields fewer tokens than the original — including the
    larger-than-original case — the function returns ``None`` and the caller
    leaves the region uncompressed.

    ``protect_code`` disables structural code elision (for fix/debug/review
    intents) while still allowing low-risk whitespace/comment slimming.
    """
    if required:
        return RegionCompression(
            CompressionMode.PASSTHROUGH_REQUIRED, content, CompressionLossRisk.NONE
        )

    original_tokens = count_tokens(content)
    if original_tokens == 0:
        return None

    candidates: list[tuple[CompressionMode, str]] = []
    is_code = _looks_like_code(language, content)
    if is_code and not protect_code:
        elided = _elide_structural(content, language=language, handle=fetch_original_handle)
        if elided is not None:
            candidates.append((CompressionMode.STRUCTURAL_CODE_ELISION, elided))
    # Literal and JSON/log modes target data payloads, not source code. Keeping
    # them off code prevents data-shaped code (assignment-heavy bodies) from
    # being summarized and keeps protect_code an effective guard for editors.
    if not is_code:
        literal = _summarize_large_literal(content, handle=fetch_original_handle)
        if literal is not None:
            candidates.append((CompressionMode.LARGE_LITERAL_SUMMARIZATION, literal))
        crushed = _crush_json_log(content, handle=fetch_original_handle)
        if crushed is not None:
            candidates.append((CompressionMode.JSON_LOG_SMART_CRUSHING, crushed))
    slimmed = _slim_comments_whitespace(content)
    if slimmed is not None:
        candidates.append((CompressionMode.COMMENT_AND_WHITESPACE_SLIMMING, slimmed))

    scored = [
        (count_tokens(new_content), _MODE_PRIORITY.index(mode), mode, new_content)
        for mode, new_content in candidates
    ]
    beneficial = [entry for entry in scored if entry[0] < original_tokens]
    if not beneficial:
        return None
    beneficial.sort(key=lambda entry: (entry[0], entry[1]))
    _, _, mode, new_content = beneficial[0]
    return RegionCompression(mode, new_content, _MODE_RISK[mode])
