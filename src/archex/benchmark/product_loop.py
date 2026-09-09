"""Paired product-as-shipped agent-loop benchmark (R20).

Two arms drive the *same* agent — Claude Code — over the same frozen task
population. Only the wired context product differs: `archex_product_loop` uses
Archex's shipped project wiring, `graft_product_loop` uses the pinned Graft
release's. The comparison is descriptive by construction; the load-bearing
output is the Archex-arm baseline that later workflow milestones re-run against.

Everything here follows the frozen protocol in
`benchmarks/preregistrations/R20-product-loop-agent-baseline.md`:

* the measured prompt is the task `question`, one blank line, then
  :data:`INSTRUCTION_BLOCK` verbatim — pinned by SHA-256 so its wording cannot be
  tuned after a pilot;
* the answer's file list may name at most :data:`ANSWER_PATH_CAP` paths, because
  the primary is a pure recall measure and an uncapped list is gameable by
  breadth alone;
* path normalization is specified step by step and applied identically to both
  arms, because matching is exact and Graft's native pointers carry
  `:L<start>-L<end>` suffixes that exact matching would otherwise reject;
* a cell that could not be measured is retained as an explicit failure and
  enters the primary mean as `0.0`; no cell is ever dropped;
* cost is Claude Code's list-price `total_cost_usd` under subscription auth, so
  it is modelled rather than billed, and the ceiling is enforced against it;
* artifacts carry no prompt text, no source, and no absolute machine paths.

The module fails closed: an artifact that contradicts the frozen identities, the
tool policy, or the cost ceiling is rejected rather than published.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from archex.benchmark.models import BenchmarkTask

PREREGISTRATION_PATH = "benchmarks/preregistrations/R20-product-loop-agent-baseline.md"
"""Tracked contract this module implements; its commit precedes every cell."""

AGENT_NAME = "claude"
"""Frozen agent executable."""

AGENT_VERSION = "2.1.266"
"""Frozen Claude Code version; a mismatch stops the run rather than substituting."""

AGENT_MODEL = "claude-haiku-4-5"
"""Frozen model, identical on both arms."""

BILLING_MODE = "oauth_subscription"
"""No dollars are charged; `total_cost_usd` is list-price arithmetic, not a bill."""

REPETITIONS = 3
"""Repetitions per task and arm; they estimate agent nondeterminism."""

ANSWER_PATH_CAP = 6
"""Maximum paths a `FILES:` block may name — twice the largest denominator."""

CELL_DEADLINE_SECONDS = 300
"""Per-cell wall-clock deadline; the agent exposes no turn or time cap."""

COST_CEILING_USD = 25.0
"""Frozen modelled-cost ceiling for the whole run."""

BOOTSTRAP_RESAMPLES = 10_000
"""Cluster bootstrap resamples, frozen with the seed below."""

BOOTSTRAP_SEED = 20260909
"""Cluster bootstrap seed, inherited from R19 so the scales are comparable."""

ANSWER_MARKER = "FILES:"
"""The line that opens the answer's file list."""

INSTRUCTION_BLOCK = (
    "Identify the files in this repository that are required to answer the "
    "question above. Use the tools available to you.\n"
    "\n"
    "End your reply with a line containing exactly `FILES:` and nothing else, "
    "followed by one repository-relative path per line, most relevant first. "
    "Name at most 6 paths, and name only files that are required to answer the "
    "question.\n"
)
"""Frozen instruction block appended to every cell's prompt, byte for byte."""

INSTRUCTION_BLOCK_SHA256 = "39acce3ce7bff054ee09b52414c9fede8807da77ccc683b211bd16129ee1007d"
"""SHA-256 of :data:`INSTRUCTION_BLOCK`, also recorded in the pre-registration."""

BASE_TOOL_ORDER: tuple[str, ...] = ("Read", "Grep", "Glob")
"""Base tool allowlist in the order the pre-registered command line writes it."""

BASE_TOOLS: frozenset[str] = frozenset(BASE_TOOL_ORDER)
"""Base tool allowlist as a set, identical on both arms."""

DENIED_TOOLS: tuple[str, ...] = (
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Bash",
    "Task",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "WebSearch",
    "WebFetch",
    "Skill",
    "Workflow",
    "DesignSync",
    "EnterWorktree",
    "ExitWorktree",
    "ListAgents",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "CronCreate",
    "CronDelete",
    "CronList",
    "Monitor",
    "PushNotification",
    "RemoteTrigger",
    "ShareOnboardingGuide",
    "ToolSearch",
)
"""Explicit denylist; the frozen tool set is asserted per cell from the receipt.

The last five are plugin-provided tools that survive ``--setting-sources
project`` on an operator machine with plugins installed. Naming them is what
reduces the advertised non-MCP set to exactly :data:`BASE_TOOLS`.
"""

SETTING_SOURCES = "project"
"""Only the cell checkout's own settings are loaded.

Subscription auth breaks whenever ``HOME`` or ``CLAUDE_CONFIG_DIR`` is
redirected, so the agent runs with the operator's real home. This flag is what
excludes the operator's user settings and ``CLAUDE.md`` instead; measured, it
takes the model from quoting the operator's memory file to reporting none, and
cuts the system prompt from 18 397 to 6 969 cache-creation tokens.
"""

_PATH_LEAK_MARKERS = ("/Users/", "/home/", "/private/", "/tmp/")

_LINE_SUFFIX = re.compile(r":L\d+-L\d+$|:\d+$")
_MARKDOWN_LINK = re.compile(r"^\[(?P<label>[^\]]*)\]\((?P<target>[^)]*)\)$")
_COST_IN_TEXT = re.compile(r'"total_cost_usd"\s*:\s*([0-9]+(?:\.[0-9]+)?)')

JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"


def as_json_object(value: JsonValue) -> dict[str, JsonValue] | None:
    """Narrow a decoded JSON value to an object, or None."""
    return value if isinstance(value, dict) else None


def as_json_array(value: JsonValue) -> list[JsonValue] | None:
    """Narrow a decoded JSON value to an array, or None."""
    return value if isinstance(value, list) else None


def as_json_text(value: JsonValue) -> str | None:
    """Narrow a decoded JSON value to a string, or None."""
    return value if isinstance(value, str) else None


def decode_json(document: str) -> JsonValue:
    """Decode a JSON document as an explicitly typed value, not `Any`."""
    raw: Any = json.loads(document)
    return cast("JsonValue", raw)


class ProductLoopError(Exception):
    """A product-loop artifact or directory contradicts the frozen protocol."""


class ProductLoopArm(StrEnum):
    """Which product's shipped wiring the agent ran with."""

    ARCHEX = "archex_product_loop"
    GRAFT = "graft_product_loop"

    @property
    def mcp_server(self) -> str:
        """Name of the single MCP server this arm registers."""
        return _ARM_MCP_SERVER[self]

    @property
    def mcp_tools(self) -> frozenset[str]:
        """Exact MCP tool names this arm's server advertises at session start."""
        return _ARM_MCP_TOOLS[self]

    @property
    def expected_tools(self) -> frozenset[str]:
        """Frozen `system`/`init` tool set for this arm."""
        return BASE_TOOLS | self.mcp_tools


_ARM_MCP_SERVER: dict[ProductLoopArm, str] = {
    ProductLoopArm.ARCHEX: "archex",
    ProductLoopArm.GRAFT: "graft",
}

_ARM_MCP_TOOLS: dict[ProductLoopArm, frozenset[str]] = {
    # Archex ships 20 MCP tools but advertises only its retrieval core until a
    # successful retrieval call opens its disclosure gate. That shipped
    # behaviour is measured, not bypassed.
    ProductLoopArm.ARCHEX: frozenset({"mcp__archex__context", "mcp__archex__query_repo"}),
    ProductLoopArm.GRAFT: frozenset(
        {
            "mcp__graft__graft_check_freshness",
            "mcp__graft__graft_file_api",
            "mcp__graft__graft_find_all",
            "mcp__graft__graft_find_code",
            "mcp__graft__graft_repo_map",
            "mcp__graft__graft_trace_calls",
        }
    ),
}


class ProductLoopCellStatus(StrEnum):
    """Whether a planned cell produced a measurement or a recorded failure."""

    OK = "ok"
    FAILED = "failed"


class ProductLoopFailureReason(StrEnum):
    """Why a planned cell became a recorded failure.

    Every one of these enters the primary completeness mean as ``0.0``. None of
    them is ever dropped from the denominator.
    """

    DEADLINE_EXCEEDED = "deadline_exceeded"
    RATE_LIMITED = "rate_limited"
    TOOL_PARITY_VIOLATION = "tool_parity_violation"
    MCP_DISCONNECTED = "mcp_disconnected"
    AGENT_ERROR = "agent_error"
    TRANSCRIPT_UNPARSABLE = "transcript_unparsable"


class ProductLoopAnswerFlag(StrEnum):
    """How the answer's file list was read."""

    SCORED = "scored"
    ANSWER_UNPARSED = "answer_unparsed"
    ANSWER_OVER_BROAD = "answer_over_broad"


class HookRecord(BaseModel):
    """One invocation of a product's shipped hook, observed at the recorder.

    Claude Code surfaces hook events for `SessionStart` only, so neither
    Archex's `PreToolUse` hook nor Graft's `UserPromptSubmit`/`PostToolUse`
    hooks appear in the transcript. The recorder wraps each hook command the
    product installed, records this row, and passes the original command's
    stdout and exit code through unchanged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str
    matcher: str | None = None
    tool_name: str | None = None
    stdin_bytes: int = Field(ge=0)
    stdout_bytes: int = Field(ge=0)
    exit_code: int
    latency_ms: float = Field(ge=0.0)
    augmented: bool
    """Whether the hook returned any `additionalContext` payload."""


class ProductLoopCellArtifact(BaseModel):
    """One planned cell: a scored measurement or a recorded failure.

    Field set is the pre-registration's privacy contract. There is no prompt
    text, no repository source, no tool input or output, and no absolute machine
    path; the prompt is fully determined by the task `question` already in the
    repository plus :data:`INSTRUCTION_BLOCK_SHA256`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: int = Field(default=1, ge=1, le=1)
    preregistration: str = PREREGISTRATION_PATH
    instruction_block_sha256: str
    task_id: str
    repo: str
    commit: str
    arm: ProductLoopArm
    repetition: int = Field(ge=1)
    status: ProductLoopCellStatus
    failure_reason: ProductLoopFailureReason | None = None
    failure_detail: str | None = None

    agent_name: str
    agent_version: str
    model_requested: str
    models_observed: list[str] = Field(default_factory=list)
    billing_mode: str

    mcp_server: str
    mcp_status: str | None = None
    mcp_dropped: bool = False
    tools_advertised: list[str] = Field(default_factory=list)

    expected_file_count: int = Field(ge=1)
    matched_file_count: int = Field(ge=0)
    completeness: float = Field(ge=0.0, le=1.0)
    answer_flag: ProductLoopAnswerFlag
    answer_paths: list[str] = Field(default_factory=list)
    answer_path_count: int = Field(ge=0)
    answer_precision: float = Field(ge=0.0, le=1.0)

    tool_calls: int = Field(ge=0)
    tool_call_mix: dict[str, int] = Field(default_factory=dict)
    product_tool_calls: int = Field(ge=0)
    no_product_use: bool

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)
    modelled_cost_usd: float = Field(ge=0.0)

    setup_seconds: float = Field(ge=0.0)
    wall_seconds: float = Field(ge=0.0)
    num_turns: int = Field(ge=0)

    freshness_state: str | None = None
    stale_index_event: bool = False
    hook_records: list[HookRecord] = Field(default_factory=list[HookRecord])
    hook_invocations: int = Field(default=0, ge=0)
    graft_cards_greppable: bool | None = None
    mcp_command_original: str | None = None
    mcp_command_used: str | None = None
    quota_utilization: float | None = None
    """Subscription five-hour window utilization at the end of the cell, when reported."""
    ambient_tool_fingerprint: str
    """SHA-256 of the advertised non-MCP tool list, which must not vary across cells."""
    provider_endpoint_overridden: bool = False
    """True when the cell ran against a stub endpoint; such cells are never publishable."""

    @model_validator(mode="after")
    def _validate_cell(self) -> Self:
        if self.instruction_block_sha256 != INSTRUCTION_BLOCK_SHA256:
            msg = (
                "instruction_block_sha256 does not match the frozen prompt "
                f"({self.instruction_block_sha256!r})"
            )
            raise ValueError(msg)
        if self.agent_name != AGENT_NAME or self.agent_version != AGENT_VERSION:
            msg = f"agent identity is not the frozen {AGENT_NAME} {AGENT_VERSION}"
            raise ValueError(msg)
        if self.model_requested != AGENT_MODEL:
            msg = f"model_requested is not the frozen {AGENT_MODEL}"
            raise ValueError(msg)
        if self.billing_mode != BILLING_MODE:
            msg = f"billing_mode is not the frozen {BILLING_MODE}"
            raise ValueError(msg)
        if self.repetition > REPETITIONS:
            msg = f"repetition {self.repetition} exceeds the frozen {REPETITIONS}"
            raise ValueError(msg)
        if self.mcp_server != self.arm.mcp_server:
            msg = f"{self.arm.value} must register the {self.arm.mcp_server!r} MCP server"
            raise ValueError(msg)
        self._validate_status()
        self._validate_scoring()
        return self

    def _validate_status(self) -> None:
        if self.status is ProductLoopCellStatus.FAILED:
            if self.failure_reason is None:
                msg = "a failed cell must record a failure reason"
                raise ValueError(msg)
            if self.completeness != 0.0 or self.matched_file_count != 0:
                msg = "a failed cell enters the mean as 0.0 and matches no files"
                raise ValueError(msg)
            return
        if self.failure_reason is not None:
            msg = "an ok cell must not record a failure reason"
            raise ValueError(msg)
        if sorted(self.tools_advertised) != sorted(self.arm.expected_tools):
            msg = (
                f"{self.arm.value} advertised {sorted(self.tools_advertised)}, "
                f"expected {sorted(self.arm.expected_tools)}"
            )
            raise ValueError(msg)
        if self.ambient_tool_fingerprint != ambient_tool_fingerprint(self.tools_advertised):
            msg = "ambient_tool_fingerprint does not match the advertised tool list"
            raise ValueError(msg)
        if self.mcp_status != "connected" or self.mcp_dropped:
            msg = "an ok cell requires a connected MCP server that never dropped"
            raise ValueError(msg)

    def _validate_scoring(self) -> None:
        if self.matched_file_count > self.expected_file_count:
            msg = "matched_file_count cannot exceed expected_file_count"
            raise ValueError(msg)
        if self.answer_path_count != len(self.answer_paths):
            msg = "answer_path_count must equal the number of recorded answer paths"
            raise ValueError(msg)
        if self.answer_flag is not ProductLoopAnswerFlag.SCORED and self.completeness != 0.0:
            msg = f"{self.answer_flag.value} cells score 0.0"
            raise ValueError(msg)
        expected = self.matched_file_count / self.expected_file_count
        if abs(self.completeness - expected) > 1e-9:
            msg = (
                f"completeness {self.completeness} does not equal "
                f"{self.matched_file_count}/{self.expected_file_count}"
            )
            raise ValueError(msg)
        if self.no_product_use != (self.product_tool_calls == 0):
            msg = "no_product_use must equal product_tool_calls == 0"
            raise ValueError(msg)
        if self.product_tool_calls > self.tool_calls:
            msg = "product_tool_calls cannot exceed tool_calls"
            raise ValueError(msg)
        if self.hook_invocations != len(self.hook_records):
            msg = "hook_invocations must equal the number of recorded hook rows"
            raise ValueError(msg)


def ambient_tool_fingerprint(tools_advertised: Sequence[str]) -> str:
    """Fingerprint the non-MCP tools the client advertised.

    Subscription auth forces the agent to run under the operator's real home, so
    the built-in and plugin surface is not fully excludable. Fingerprinting what
    each cell actually saw is what makes the residual auditable: the validator
    requires one value across the whole run, so an install that changes
    mid-campaign is a visible failure rather than a silent confound.
    """
    non_mcp = sorted(name for name in tools_advertised if not name.startswith("mcp__"))
    return hashlib.sha256("\n".join(non_mcp).encode("utf-8")).hexdigest()


def build_prompt(question: str) -> str:
    """Return the frozen measured prompt for a task's question.

    Exactly the `question`, one blank line, then :data:`INSTRUCTION_BLOCK`.
    """
    return f"{question.strip()}\n\n{INSTRUCTION_BLOCK}"


def instruction_block_digest(block: str = INSTRUCTION_BLOCK) -> str:
    """SHA-256 of an instruction block, over its exact bytes."""
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def preregistered_instruction_block(preregistration: Path) -> str:
    """Extract the instruction block the pre-registration froze.

    The pre-registration's Appendix A fenced block is the authoritative bytes.
    Reading it back is what makes drift between document and code detectable
    instead of a silent protocol change.
    """
    text = preregistration.read_text(encoding="utf-8")
    marker = "## Appendix A"
    start = text.find(marker)
    if start < 0:
        msg = f"{preregistration} has no Appendix A instruction block"
        raise ProductLoopError(msg)
    fence = text.find("```text\n", start)
    if fence < 0:
        msg = f"{preregistration} Appendix A has no fenced text block"
        raise ProductLoopError(msg)
    body_start = fence + len("```text\n")
    end = text.find("```", body_start)
    if end < 0:
        msg = f"{preregistration} Appendix A fence is unterminated"
        raise ProductLoopError(msg)
    return text[body_start:end]


def assert_frozen_prompt(preregistration: Path) -> None:
    """Fail closed unless code, document, and recorded hash all agree."""
    digest = instruction_block_digest()
    if digest != INSTRUCTION_BLOCK_SHA256:
        msg = f"INSTRUCTION_BLOCK hashes to {digest}, not the frozen {INSTRUCTION_BLOCK_SHA256}"
        raise ProductLoopError(msg)
    document_block = preregistered_instruction_block(preregistration)
    if document_block != INSTRUCTION_BLOCK:
        msg = "INSTRUCTION_BLOCK does not match the pre-registration's Appendix A block"
        raise ProductLoopError(msg)


def normalize_answer_path(token: str, *, repo_root: Path) -> str | None:
    """Normalize one answer token to a repository-relative POSIX path.

    Applied identically on both arms, in the pre-registered order: strip
    markdown link syntax and surrounding backticks; strip a trailing
    `:L<start>-L<end>` or `:<line>` suffix, because Graft's native pointers
    carry one; convert `\\` to `/`; strip a leading `./`; rebase an absolute
    path under the checkout root; discard anything that does not resolve inside
    the checkout.
    """
    candidate = token.strip().strip("\"'").rstrip(".,;")
    link = _MARKDOWN_LINK.match(candidate)
    if link is not None:
        candidate = link.group("target").strip() or link.group("label").strip()
    candidate = candidate.strip("`").strip()
    if not candidate:
        return None
    candidate = _LINE_SUFFIX.sub("", candidate)
    candidate = candidate.replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or candidate.endswith("/"):
        return None

    root = repo_root.resolve()
    raw = Path(candidate)
    absolute = raw if raw.is_absolute() else root / raw
    try:
        resolved = absolute.resolve()
        relative = resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    text = relative.as_posix()
    return text or None


def extract_answer_paths(
    final_text: str, *, repo_root: Path
) -> tuple[list[str], ProductLoopAnswerFlag]:
    """Read the answer's file list from the final assistant text.

    Returns the normalized paths and how the list was read. The list is taken
    from the *last* line whose stripped content equals `FILES:`.

    The cardinality cap applies to *paths*, which is what the pre-registration
    freezes, so tokens are normalized and resolved first and only then counted.
    Counting raw tokens instead is not equivalent and is not arm-neutral: Graft's
    shipped `Stop` hook appends a token-savings footer after the answer, whose
    words would otherwise be counted as answer entries and push every Graft cell
    over the cap. A block that still names more than :data:`ANSWER_PATH_CAP` real
    repository files is `answer_over_broad` and scores zero, because the primary
    is a pure recall measure and an uncapped list would let breadth alone reach
    `1.0`.
    """
    lines = final_text.splitlines()
    marker_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == ANSWER_MARKER:
            marker_index = index
    if marker_index is None:
        return [], ProductLoopAnswerFlag.ANSWER_UNPARSED

    paths: list[str] = []
    for line in lines[marker_index + 1 :]:
        if not line.strip():
            continue
        for token in re.split(r"[,\s]+", line.strip()):
            if not token:
                continue
            normalized = normalize_answer_path(token, repo_root=repo_root)
            if normalized is not None and normalized not in paths:
                paths.append(normalized)
    if len(paths) > ANSWER_PATH_CAP:
        return [], ProductLoopAnswerFlag.ANSWER_OVER_BROAD
    return paths, ProductLoopAnswerFlag.SCORED


def score_completeness(answer_paths: Sequence[str], expected_files: Sequence[str]) -> int:
    """Count the task's labeled required files named in the answer.

    Matching is exact on the normalized path — no prefix, basename, or fuzzy
    matching, on either arm.
    """
    named = set(answer_paths)
    return sum(1 for expected in set(expected_files) if expected in named)


class TranscriptSummary(BaseModel):
    """What one cell's `stream-json` transcript establishes about the run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tools_advertised: list[str]
    mcp_status: str | None
    mcp_dropped: bool
    models_observed: list[str]
    tool_call_mix: dict[str, int]
    tool_calls: int
    product_tool_calls: int
    final_text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    modelled_cost_usd: float
    num_turns: int
    is_error: bool
    error_text: str | None
    saw_result: bool
    rate_limited: bool
    quota_utilization: float | None = None
    """Fraction of the subscription's five-hour window in use, when reported."""


def summarize_transcript(lines: Iterable[str], *, arm: ProductLoopArm) -> TranscriptSummary:
    """Reduce a Claude Code `stream-json` transcript to the recorded fields.

    Unknown event types are ignored, but a malformed JSON line is a hard error:
    a transcript the harness cannot read must become a recorded failure rather
    than a silently short-counted cell.
    """
    tools_advertised: list[str] = []
    mcp_status: str | None = None
    mcp_dropped = False
    models: list[str] = []
    mix: dict[str, int] = {}
    final_text = ""
    usage: dict[str, int] = {}
    cost = 0.0
    turns = 0
    is_error = False
    error_text: str | None = None
    saw_result = False
    rate_limited = False
    quota_utilization: float | None = None
    server = arm.mcp_server

    for raw in lines:
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            decoded = decode_json(line)
        except json.JSONDecodeError as exc:
            msg = f"transcript line is not JSON: {exc}"
            raise ProductLoopError(msg) from exc
        event = as_json_object(decoded)
        if event is None:
            msg = "transcript line is not a JSON object"
            raise ProductLoopError(msg)

        kind = as_json_text(event.get("type"))
        if kind == "system":
            subtype = as_json_text(event.get("subtype"))
            if subtype == "init":
                names = as_json_array(event.get("tools")) or []
                tools_advertised = [
                    text for text in (as_json_text(name) for name in names) if text is not None
                ]
                mcp_status = _mcp_status(event.get("mcp_servers"), server=server)
            elif subtype == "mcp_status":
                later = _mcp_status(event.get("mcp_servers"), server=server)
                if later is not None and later != "connected":
                    mcp_dropped = True
        elif kind == "rate_limit_event":
            # Claude Code emits this routinely as quota telemetry, with
            # `status: "allowed"`. Treating every occurrence as a failure would
            # turn an entirely healthy run into 114 recorded failures.
            info = as_json_object(event.get("rate_limit_info"))
            status = as_json_text(info.get("status")) if info is not None else None
            if status is not None and status != "allowed":
                rate_limited = True
            quota_utilization = _five_hour_utilization(info) or quota_utilization
        elif kind == "assistant":
            message = as_json_object(event.get("message"))
            if message is None:
                continue
            model = as_json_text(message.get("model"))
            if model is not None and model not in models:
                models.append(model)
            for raw_block in as_json_array(message.get("content")) or []:
                block = as_json_object(raw_block)
                if block is None:
                    continue
                block_type = as_json_text(block.get("type"))
                if block_type == "tool_use":
                    name = as_json_text(block.get("name"))
                    if name:
                        mix[name] = mix.get(name, 0) + 1
                elif block_type == "text":
                    final_text = as_json_text(block.get("text")) or ""
        elif kind == "user":
            mcp_dropped = mcp_dropped or _reports_mcp_error(event, server=server)
        elif kind == "result" or "total_cost_usd" in event:
            saw_result = True
            cost = _as_number(event.get("total_cost_usd"))
            turns = int(_as_number(event.get("num_turns")))
            is_error = event.get("is_error") is True
            usage = _usage_counts(event.get("usage"))
            result_text = as_json_text(event.get("result"))
            if is_error:
                error_text = result_text
            if result_text is not None and "rate limit" in result_text.lower():
                rate_limited = True

    product_calls = sum(count for name, count in mix.items() if name.startswith(f"mcp__{server}__"))
    return TranscriptSummary(
        tools_advertised=tools_advertised,
        mcp_status=mcp_status,
        mcp_dropped=mcp_dropped,
        models_observed=models,
        tool_call_mix=mix,
        tool_calls=sum(mix.values()),
        product_tool_calls=product_calls,
        final_text=final_text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        modelled_cost_usd=cost,
        num_turns=turns,
        is_error=is_error,
        error_text=error_text,
        saw_result=saw_result,
        rate_limited=rate_limited,
        quota_utilization=quota_utilization,
    )


def _five_hour_utilization(info: dict[str, JsonValue] | None) -> float | None:
    """Read the five-hour window utilization from a rate-limit telemetry event."""
    if info is None:
        return None
    windows = as_json_object(info.get("unifiedWindows"))
    five_hour = as_json_object(windows.get("five_hour")) if windows is not None else None
    if five_hour is None:
        return None
    value = five_hour.get("utilization")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_number(value: JsonValue) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mcp_status(servers: JsonValue, *, server: str) -> str | None:
    for raw_entry in as_json_array(servers) or []:
        entry = as_json_object(raw_entry)
        if entry is not None and as_json_text(entry.get("name")) == server:
            return as_json_text(entry.get("status"))
    return None


def _reports_mcp_error(event: dict[str, JsonValue], *, server: str) -> bool:
    """Whether a tool result shows the arm's MCP server failing after start.

    The `init` event only proves the server was connected at t=0. A server that
    dies afterwards leaves the agent silently falling back to file tools, and
    the cell would otherwise still be attributed to the product.
    """
    message = as_json_object(event.get("message"))
    if message is None:
        return False
    for raw_block in as_json_array(message.get("content")) or []:
        block = as_json_object(raw_block)
        if block is None or as_json_text(block.get("type")) != "tool_result":
            continue
        if block.get("is_error") is not True:
            continue
        text = json.dumps(block.get("content", ""))
        if f"mcp__{server}__" in text or f"MCP server {server}" in text:
            return True
    return False


def _usage_counts(usage: JsonValue) -> dict[str, int]:
    counts: dict[str, int] = {}
    fields = as_json_object(usage)
    if fields is None:
        return counts
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        value = fields.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        counts[key] = int(value)
    return counts


def read_hook_records(cell_dir: Path) -> list[HookRecord]:
    """Read the instrumented hook boundary's rows, ignoring recorder anomalies."""
    log = cell_dir / "hooks.jsonl"
    if not log.is_file():
        return []
    records: list[HookRecord] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = as_json_object(decode_json(line))
        except json.JSONDecodeError:
            # Killing the process group can interrupt the recorder mid-write.
            continue
        if row is None or "recorder_error" in row:
            continue
        records.append(
            HookRecord(
                event=as_json_text(row.get("event")) or "",
                matcher=as_json_text(row.get("matcher")),
                tool_name=as_json_text(row.get("tool_name")),
                stdin_bytes=as_json_int(row.get("stdin_bytes")),
                stdout_bytes=as_json_int(row.get("stdout_bytes")),
                exit_code=as_json_int(row.get("exit_code")),
                latency_ms=as_json_float(row.get("latency_ms")),
                augmented=row.get("augmented") is True,
            )
        )
    return records


def as_json_int(value: JsonValue) -> int:
    """Read a recorded integer field, defaulting to zero rather than guessing."""
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def as_json_float(value: JsonValue) -> float:
    """Read a recorded float field, defaulting to zero rather than guessing."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def classify_cell_failure(
    *, timed_out: bool, summary: TranscriptSummary | None, arm: ProductLoopArm
) -> ProductLoopFailureReason | None:
    """Classify a cell that cannot be scored, in the pre-registered order."""
    if timed_out:
        return ProductLoopFailureReason.DEADLINE_EXCEEDED
    if summary is None:
        return ProductLoopFailureReason.TRANSCRIPT_UNPARSABLE
    if summary.rate_limited:
        return ProductLoopFailureReason.RATE_LIMITED
    if not summary.saw_result or summary.is_error:
        return ProductLoopFailureReason.AGENT_ERROR
    if summary.mcp_status != "connected" or summary.mcp_dropped:
        return ProductLoopFailureReason.MCP_DISCONNECTED
    if sorted(summary.tools_advertised) != sorted(arm.expected_tools):
        return ProductLoopFailureReason.TOOL_PARITY_VIOLATION
    return None


def salvage_modelled_cost(transcript: str) -> float:
    """Recover a cell's recorded cost from a transcript the parser rejected.

    Killing the process group at the deadline truncates the stream, so the
    terminal result object may be unreadable as JSON while the cost it reports
    is still present in the bytes. The frozen deadline rule says a terminated
    cell's cost counts against the ceiling, so a best-effort scan is better than
    silently charging the run zero for tokens it actually spent.
    """
    matches = _COST_IN_TEXT.findall(transcript)
    if not matches:
        return 0.0
    try:
        return max(float(value) for value in matches)
    except ValueError:  # pragma: no cover - findall only yields numeric text
        return 0.0


def build_cell_artifact(
    *,
    task: BenchmarkTask,
    arm: ProductLoopArm,
    repetition: int,
    summary: TranscriptSummary | None,
    reason: ProductLoopFailureReason | None,
    detail: str | None,
    salvaged_cost_usd: float = 0.0,
    repo_path: Path,
    setup_seconds: float,
    wall_seconds: float,
    hook_records: list[HookRecord],
    freshness_state: str | None,
    stale_index_event: bool,
    graft_cards_greppable: bool | None,
    mcp_command_original: str | None,
    mcp_command_used: str | None,
    provider_endpoint_overridden: bool,
) -> ProductLoopCellArtifact:
    expected_count = len(task.expected_files)
    common: dict[str, Any] = {
        "instruction_block_sha256": INSTRUCTION_BLOCK_SHA256,
        "task_id": task.task_id,
        "repo": task.repo,
        "commit": task.commit,
        "arm": arm,
        "repetition": repetition,
        "agent_name": AGENT_NAME,
        "agent_version": AGENT_VERSION,
        "model_requested": AGENT_MODEL,
        "billing_mode": BILLING_MODE,
        "mcp_server": arm.mcp_server,
        "expected_file_count": expected_count,
        "setup_seconds": setup_seconds,
        "wall_seconds": wall_seconds,
        "freshness_state": freshness_state,
        "stale_index_event": stale_index_event,
        "hook_records": hook_records,
        "hook_invocations": len(hook_records),
        "graft_cards_greppable": graft_cards_greppable,
        "mcp_command_original": mcp_command_original,
        "mcp_command_used": mcp_command_used,
        "provider_endpoint_overridden": provider_endpoint_overridden,
        "ambient_tool_fingerprint": ambient_tool_fingerprint([]),
    }
    if summary is not None:
        common |= {
            "models_observed": summary.models_observed,
            "mcp_status": summary.mcp_status,
            "mcp_dropped": summary.mcp_dropped,
            "tools_advertised": sorted(summary.tools_advertised),
            "ambient_tool_fingerprint": ambient_tool_fingerprint(summary.tools_advertised),
            "tool_calls": summary.tool_calls,
            "tool_call_mix": summary.tool_call_mix,
            "product_tool_calls": summary.product_tool_calls,
            "no_product_use": summary.product_tool_calls == 0,
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "cache_read_tokens": summary.cache_read_tokens,
            "cache_creation_tokens": summary.cache_creation_tokens,
            "modelled_cost_usd": summary.modelled_cost_usd,
            "num_turns": summary.num_turns,
            "quota_utilization": summary.quota_utilization,
        }

    if reason is not None:
        # A cell with no usable transcript still has to become a retained
        # failure. Truncating the last stream-json line is the normal result of
        # killing the process group at the deadline, so without these zero
        # defaults the first timed-out cell would abort the whole run.
        zeroed: dict[str, Any] = {
            "tool_calls": 0,
            "product_tool_calls": 0,
            "no_product_use": True,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "modelled_cost_usd": salvaged_cost_usd,
            "num_turns": 0,
        }
        return ProductLoopCellArtifact.model_validate(
            zeroed
            | common
            | {
                "status": ProductLoopCellStatus.FAILED,
                "failure_reason": reason,
                "failure_detail": detail,
                "matched_file_count": 0,
                "completeness": 0.0,
                "answer_flag": ProductLoopAnswerFlag.ANSWER_UNPARSED,
                "answer_paths": [],
                "answer_path_count": 0,
                "answer_precision": 0.0,
            }
        )

    assert summary is not None  # noqa: S101 - _failure_reason returns None only with a summary
    paths, flag = extract_answer_paths(summary.final_text, repo_root=repo_path)
    matched = score_completeness(paths, task.expected_files) if paths else 0
    completeness = 0.0 if flag is not ProductLoopAnswerFlag.SCORED else matched / expected_count
    precision = (matched / len(paths)) if paths else 0.0
    return ProductLoopCellArtifact.model_validate(
        common
        | {
            "status": ProductLoopCellStatus.OK,
            "matched_file_count": 0 if flag is not ProductLoopAnswerFlag.SCORED else matched,
            "completeness": completeness,
            "answer_flag": flag,
            "answer_paths": paths,
            "answer_path_count": len(paths),
            "answer_precision": precision,
        }
    )


def sanitize_document(document: str, *, replacements: Sequence[tuple[Path, str]]) -> str:
    """Replace machine paths with placeholders and refuse residual leaks.

    Sanitization is not best-effort: a residual absolute path aborts the run
    rather than being published, because a leaked path is a privacy-contract
    violation and the artifact is the published surface.
    """
    sanitized = document
    for path, placeholder in replacements:
        for variant in {str(path), str(path.resolve())}:
            sanitized = sanitized.replace(variant, placeholder)
    for marker in _PATH_LEAK_MARKERS:
        if marker in sanitized:
            msg = f"artifact leaks an absolute path containing {marker!r}"
            raise ProductLoopError(msg)
    return sanitized


def cell_filename(task_id: str, repetition: int) -> str:
    """Stable per-cell artifact filename."""
    return f"{task_id}__rep{repetition}.json"


def load_product_loop_artifact(path: Path) -> ProductLoopCellArtifact:
    """Load and validate one cell artifact, failing closed on any mismatch."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"{path} is not readable product-loop JSON: {exc}"
        raise ProductLoopError(msg) from exc
    try:
        return ProductLoopCellArtifact.model_validate(payload)
    except ValidationError as exc:
        msg = f"{path} is not a valid product-loop artifact: {exc}"
        raise ProductLoopError(msg) from exc


class ProductLoopCoverage(BaseModel):
    """What a frozen artifact directory actually contains."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: int = Field(ge=0)
    planned_cells: int = Field(ge=0)
    ok_cells: int = Field(ge=0)
    failed_cells: int = Field(ge=0)
    total_modelled_cost_usd: float = Field(ge=0.0)
    arms: list[ProductLoopArm]
    task_ids: list[str]

    @property
    def complete(self) -> bool:
        """Whether every planned cell exists as a result or a recorded failure."""
        return self.cells == self.planned_cells


def validate_product_loop_directory(
    directory: Path,
    *,
    task_ids: Sequence[str],
    repetitions: int = REPETITIONS,
    cost_ceiling_usd: float = COST_CEILING_USD,
    require_complete: bool = True,
) -> ProductLoopCoverage:
    """Validate a whole product-loop artifact directory.

    Proves exact population, arm, and repetition coverage; identical agent,
    model, and billing identity across every cell; per-arm tool parity on every
    scored cell; and that recorded cost stays within the pre-registered ceiling.
    """
    if not task_ids:
        msg = "product-loop validation requires a task population"
        raise ProductLoopError(msg)
    if not directory.is_dir():
        msg = f"{directory} is not a product-loop artifact directory"
        raise ProductLoopError(msg)

    arms = list(ProductLoopArm)
    planned = {
        (arm, task_id, repetition)
        for arm in arms
        for task_id in task_ids
        for repetition in range(1, repetitions + 1)
    }
    seen: dict[tuple[ProductLoopArm, str, int], Path] = {}
    fingerprints: set[str] = set()
    costs: list[float] = []
    total_cost = 0.0
    ok_cells = 0
    failed_cells = 0

    for arm in arms:
        arm_dir = directory / arm.value
        if not arm_dir.is_dir():
            msg = f"{directory} is missing the {arm.value} arm directory"
            raise ProductLoopError(msg)
        for path in sorted(arm_dir.glob("*.json")):
            artifact = load_product_loop_artifact(path)
            if artifact.arm is not arm:
                msg = f"{path} records arm {artifact.arm.value} under {arm.value}"
                raise ProductLoopError(msg)
            if path.name != cell_filename(artifact.task_id, artifact.repetition):
                msg = f"{path} filename does not match its task and repetition"
                raise ProductLoopError(msg)
            key = (arm, artifact.task_id, artifact.repetition)
            if key in seen:
                msg = f"{path} duplicates the cell already recorded at {seen[key]}"
                raise ProductLoopError(msg)
            if key not in planned:
                msg = f"{path} is outside the frozen population"
                raise ProductLoopError(msg)
            if artifact.provider_endpoint_overridden:
                # A no-spend stub run must never be publishable as evidence.
                msg = f"{path} was produced against an overridden provider endpoint"
                raise ProductLoopError(msg)
            seen[key] = path
            total_cost += artifact.modelled_cost_usd
            costs.append(artifact.modelled_cost_usd)
            if artifact.status is ProductLoopCellStatus.OK:
                fingerprints.add(artifact.ambient_tool_fingerprint)
            if artifact.status is ProductLoopCellStatus.OK:
                ok_cells += 1
            else:
                failed_cells += 1

    if len(fingerprints) > 1:
        # The client surface changed mid-run, so the cells are not comparable.
        msg = f"scored cells report {len(fingerprints)} different ambient tool surfaces"
        raise ProductLoopError(msg)

    missing = planned - set(seen)
    if missing and require_complete:
        example = sorted(f"{arm.value}/{task}#{rep}" for arm, task, rep in missing)[:5]
        msg = f"{len(missing)} planned cells are missing, for example {example}"
        raise ProductLoopError(msg)
    # The runner aborts *before* starting a new cell, so the cell that crossed
    # the ceiling is allowed to finish and be retained. Enforcing a hard ceiling
    # here would reject exactly the partial evidence the protocol says to keep,
    # so the allowance is the most expensive single cell actually recorded.
    headroom = max(costs) if costs else 0.0
    if total_cost > cost_ceiling_usd + headroom + 1e-9:
        msg = (
            f"recorded modelled cost {total_cost:.4f} exceeds the pre-registered "
            f"ceiling {cost_ceiling_usd:.2f} by more than one cell"
        )
        raise ProductLoopError(msg)

    return ProductLoopCoverage(
        cells=len(seen),
        planned_cells=len(planned),
        ok_cells=ok_cells,
        failed_cells=failed_cells,
        total_modelled_cost_usd=total_cost,
        arms=arms,
        task_ids=sorted(task_ids),
    )
