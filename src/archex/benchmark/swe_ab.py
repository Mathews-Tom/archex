"""SWE-task A/B of archex surfaces under omp (draft R3x protocol).

One cell is (task, configuration, arm, repetition). A configuration is a model
plus a thinking effort; the same omp build, task prompt, base tool set, and time
cap run in every arm, and only the archex surface differs:

* ``A0`` — control: archex not installed;
* ``H``  — the grep-result annotation hook, archex not on the agent's ``PATH``;
* ``HC`` — the hook plus the archex CLI and the frozen CLI guide;
* ``C``  — the CLI and guide without the hook (pilot only).

This module holds everything a cell needs to be recorded and judged the same
way every time: the channel rule table (also used by the SWE-agent trajectory
headroom script), the omp session parser, compounded-cost arithmetic, the
configuration table, the cell schema, and the directory validator. It performs
no model call and no network I/O; the only files it reads are the ones it is
given.

Campaign cells run on Muna-hosted open models (``https://inference.muna.ai/v1``,
bearer key in ``MUNA_ACCESS_KEY``) through omp's ``openai-completions`` provider;
``usage.cost_usd`` is omp's cost model of the tokens at the prices in
``benchmarks/swe_ab/muna-models.yml``. The validator mirrors R20's discipline:
every declared cell must be present, recorded failures are kept, identities and
fingerprints must agree across cells, any cell whose provider endpoint is not
Muna's (the local stub used for no-spend rehearsals) is refused for publication,
and so is any cell that did not authenticate with exactly the campaign key,
ran on another provider, or ended in a rate-limit, capacity, or credit block (a
block is not a task outcome; the suite re-runs it or stops). Emulated and native
cells never mix within a stage.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

PREREGISTRATION_PATH = "benchmarks/preregistrations/R3x-swe-archex-ab.md"
"""The draft pre-registration this module implements."""

CLI_GUIDE_PATH = "benchmarks/swe_ab/cli-guide.md"
"""Frozen CLI guide appended to the system prompt in HC and C."""

OMP_VERSION = "18.4.4"
"""omp build pinned for the whole campaign."""

CAMPAIGN_PROVIDER = "muna"
"""The only provider a campaign cell may run on (an omp custom provider, ``openai-completions``)."""

MUNA_BASE_URL = "https://inference.muna.ai/v1"
"""The endpoint every campaign cell must have resolved; any other base URL is a rehearsal."""

CREDENTIAL_ENV_NAMES: tuple[str, ...] = ("MUNA_ACCESS_KEY",)
"""The only credential variable an agent container receives; cells record names, not values."""

PROVIDER_CONFIG_PATH = "benchmarks/swe_ab/muna-models.yml"
"""Frozen omp ``models.yml``: the Muna provider, its models, and their prices (USD per Mtok)."""

OMP_CONFIG_PATH = "benchmarks/swe_ab/omp-campaign.yml"
"""Frozen omp settings overlay (``--config``): the retry budget for Muna capacity waits."""

MAX_TIME = "60m"
"""omp session cap, identical across arms."""

BASE_TOOLS: tuple[str, ...] = ("read", "bash", "edit", "write", "grep", "glob", "todo")
"""The only tools advertised in every arm."""

ISOLATION_FLAGS: tuple[str, ...] = ("--no-lsp", "--no-skills", "--no-rules", "--no-extensions")
"""Every competing retrieval surface and ambient extension is off in every arm."""


class Configuration(NamedTuple):
    """One model at one thinking effort; the unit the campaign stratifies by."""

    label: str
    """Name used in plans, cells, directories, and analysis strata."""
    selector: str
    """The omp ``--model`` selector (``provider/id``); also the model family."""
    thinking: str
    """The omp ``--thinking`` effort sent for this configuration."""


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("qwen-3.8-27b@low", "muna/@qwen/qwen-3.8-27b", "low"),
    Configuration("qwen-3.8-27b@high", "muna/@qwen/qwen-3.8-27b", "high"),
    Configuration("gemma-4-26b-a4b-it@high", "muna/@google/gemma-4-26b-a4b-it", "high"),
)
"""Campaign configurations. Muna honours effort for Qwen but not for Gemma (RUNBOOK §4)."""

MODELS: tuple[str, ...] = tuple(configuration.label for configuration in CONFIGURATIONS)
"""Configuration labels; plans and cells carry the label as their ``model``."""

QUOTA_BLOCKED_DIR = "quota-blocked"
"""Directory of attempts filed for a rate-limit, capacity, or credit block."""

PROFILE = "swebench"
"""Isolated omp profile provisioned for the campaign."""

ANNOTATION_MARKER = "[archex receipt] index_revision="
"""First line of every appended annotation block."""

COMPRESSOR_MARKERS: tuple[str, ...] = ("[laconic ", "[shaken ~")
"""Tool-output rewriters that must not load in any arm."""

SYSTEM_PROMPT_FORBIDDEN: tuple[tuple[str, str], ...] = (
    ("agents_md", "AGENTS.md"),
    ("claude_md", "CLAUDE.md"),
    ("memories", "<memories>"),
    ("laconic", "[laconic"),
)
"""Ambient context that must not reach the system prompt in any arm."""


class SweAbError(ValueError):
    """A cell, plan, or result directory violates the protocol."""


def configuration(label: str) -> Configuration:
    """The campaign configuration named ``label``; an unknown label is a protocol error."""
    for entry in CONFIGURATIONS:
        if entry.label == label:
            return entry
    raise SweAbError(f"unknown configuration {label!r}; the campaign runs {list(MODELS)}")


class SweAbArm(StrEnum):
    A0 = "A0"
    H = "H"
    HC = "HC"
    C = "C"

    @property
    def hook(self) -> bool:
        return self in (SweAbArm.H, SweAbArm.HC)

    @property
    def cli(self) -> bool:
        return self in (SweAbArm.HC, SweAbArm.C)

    @property
    def archex_installed(self) -> bool:
        return self is not SweAbArm.A0


# --- channel rule table ---------------------------------------------------------
#
# Two corpora, one module. SWE-agent trajectories (the published Scale runs the
# token-headroom script reads) record each step as a shell-like action string;
# omp sessions record typed tool calls. The regexes below are shared where a
# rule means the same thing in both; the omp search rule is the spec's own list.

SWE_AGENT_CHANNELS: tuple[str, ...] = (
    "search",
    "read_file",
    "edit",
    "test_build",
    "submit_vcs",
    "other",
)

_SEARCH = re.compile(
    r"^\s*(?:grep|rg|egrep|fgrep|find|ls|tree|git\s+grep|git\s+ls-files|locate|which)\b"
)
_VIEW = re.compile(
    r"^\s*(?:str_replace_editor\s+view|cat|nl|head|tail|less|more|sed\s+-n\s+\S+)\s+(\S+)"
)
_VIEW_ANY = re.compile(r"^\s*(?:str_replace_editor\s+view|cat|nl|head|tail|less|more|sed\s+-n)\b")
_EDIT = re.compile(
    r"^\s*(?:str_replace_editor\s+(?:create|str_replace|insert|undo_edit)|patch|apply_patch)\b"
)
_TEST = re.compile(
    r"\b(?:pytest|go\s+test|gotestsum|ginkgo|npm\s+(?:test|run)|yarn\s+(?:test|run)|pnpm\s+run"
    r"|mocha|jest|vitest|tox|nosetests|python\s+-m\s+unittest|make\s+\S*test|make\s+build"
    r"|cargo\s+test|mvn\s+test|gradle\s+test)\b"
)
_VCS = re.compile(r"^\s*(?:submit|git\s+(?:diff|status|log|stash|checkout|add|apply|reset))\b")
_DIFF_FILE = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)

CHANNELS: tuple[str, ...] = (
    "archex-annotation",
    "archex-CLI",
    "search",
    "test",
    "read",
    "edit",
    "write",
    "other",
)
"""omp channel set (spec §7); frozen with the pre-registration."""

_OMP_BASH_SEARCH = re.compile(r"^\s*(?:rg|grep|egrep|fgrep|ag|ack|find|git\s+grep|ls\s+-\w*R)\b")
_ARCHEX_CLI = re.compile(r"^\s*archex(?:\s|$)")
_LEADING_CD = re.compile(r"^\s*cd\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s*&&\s*")


def swe_agent_channel_of(action: str) -> str:
    """Attribute one SWE-agent trajectory step to exactly one channel."""
    text = (action or "").strip()
    if not text:
        return "other"
    if _TEST.search(text):
        return "test_build"
    if _EDIT.match(text):
        return "edit"
    if _SEARCH.match(text) or "| grep" in text or "grep -" in text:
        return "search"
    if _VIEW_ANY.match(text):
        return "read_file"
    if _VCS.match(text):
        return "submit_vcs"
    return "other"


def viewed_path(action: str) -> str | None:
    """File a view-shaped shell action reads, repository-relative when rooted at /app."""
    match = _VIEW.match((action or "").strip())
    if match is None:
        return None
    path = match.group(1).strip("'\"")
    for prefix in ("/app/", "app/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    return path


def diff_files(patch: str) -> list[str]:
    """Files a unified git diff touches, in order of appearance."""
    return list(dict.fromkeys(_DIFF_FILE.findall(patch or "")))


def touches(path: str, targets: Iterable[str]) -> bool:
    """Whether ``path`` names one of ``targets`` up to a leading directory prefix."""
    return any(
        path == target or path.endswith("/" + target) or target.endswith("/" + path)
        for target in targets
    )


def bash_command_body(command: str) -> str:
    """The command with leading ``cd <dir> &&`` segments removed."""
    text = command
    while True:
        stripped = _LEADING_CD.sub("", text, count=1)
        if stripped == text:
            return text.strip()
        text = stripped


def omp_channel(tool: str, arguments: Mapping[str, object]) -> str:
    """Attribute one omp tool call to exactly one channel (spec §7)."""
    if tool in ("grep", "glob"):
        return "search"
    if tool in ("read", "edit", "write"):
        return tool
    if tool != "bash":
        return "other"
    raw = arguments.get("command")
    if not isinstance(raw, str):
        return "other"
    body = bash_command_body(raw)
    if _ARCHEX_CLI.match(body):
        return "archex-CLI"
    if _TEST.search(body):
        return "test"
    if _OMP_BASH_SEARCH.match(body) or "| grep" in body:
        return "search"
    if _VIEW_ANY.match(body):
        return "read"
    return "other"


def archex_subcommand(command: str) -> str | None:
    """The archex subcommand a bash command runs, or `None` if it runs none."""
    body = bash_command_body(command)
    if not _ARCHEX_CLI.match(body):
        return None
    try:
        tokens = shlex.split(body)
    except ValueError:
        tokens = body.split()
    return tokens[1] if len(tokens) > 1 else ""


def compound(
    observations: Iterable[tuple[int, str, int]], request_count: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Once and compounded token totals per channel.

    An observation produced by request ``i`` of ``n`` is re-sent with every
    later request, so it is charged ``n - 1 - i`` times; ``observations`` holds
    ``(request_index, channel, tokens)``.
    """
    once: dict[str, int] = dict.fromkeys(CHANNELS, 0)
    compounded: dict[str, int] = dict.fromkeys(CHANNELS, 0)
    for request_index, channel, tokens in observations:
        if channel not in once:
            raise SweAbError(f"unknown channel {channel!r}")
        once[channel] += tokens
        compounded[channel] += tokens * max(0, request_count - 1 - request_index)
    return once, compounded


# --- omp session parsing --------------------------------------------------------


@dataclass(frozen=True)
class RequestUsage:
    index: int
    provider: str
    model: str
    input: int
    output: int
    cache_read: int
    cache_write: int
    cost_usd: float
    stop_reason: str
    non_message_tokens: int | None


@dataclass
class ToolExchange:
    call_id: str
    tool: str
    arguments: dict[str, Any]
    request_index: int
    result_blocks: list[str] = field(default_factory=list[str])
    is_error: bool = False
    answered: bool = False

    @property
    def result_text(self) -> str:
        return "".join(self.result_blocks)


@dataclass
class OmpSession:
    requests: list[RequestUsage] = field(default_factory=list[RequestUsage])
    exchanges: list[ToolExchange] = field(default_factory=list[ToolExchange])
    error_message: str | None = None


def _int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def parse_omp_session(path: Path) -> OmpSession:
    """Read per-request usage and every tool call with its result from an omp session."""
    session = OmpSession()
    by_id: dict[str, ToolExchange] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry_obj: object = json.loads(line)
        if not isinstance(entry_obj, dict):
            continue
        entry = cast("dict[str, Any]", entry_obj)
        if entry.get("type") != "message" or not isinstance(entry.get("message"), dict):
            continue
        message = cast("dict[str, Any]", entry["message"])
        role = message.get("role")
        if role == "assistant":
            usage = cast("dict[str, Any]", message.get("usage") or {})
            cost = cast("dict[str, Any]", usage.get("cost") or {})
            snapshot = cast("dict[str, Any]", message.get("contextSnapshot") or {})
            index = len(session.requests)
            non_message = snapshot.get("nonMessageTokens")
            session.requests.append(
                RequestUsage(
                    index=index,
                    provider=str(message.get("provider", "")),
                    model=str(message.get("model", "")),
                    input=_int(usage.get("input")),
                    output=_int(usage.get("output")),
                    cache_read=_int(usage.get("cacheRead")),
                    cache_write=_int(usage.get("cacheWrite")),
                    cost_usd=float(cost.get("total") or 0.0),
                    stop_reason=str(message.get("stopReason", "")),
                    non_message_tokens=non_message if isinstance(non_message, int) else None,
                )
            )
            # Only the final assistant turn decides whether the run ended in an error: a turn
            # omp retried (a rate limit it waited out) is not a failure of the cell.
            session.error_message = (
                str(message.get("errorMessage") or "provider error")
                if message.get("stopReason") == "error"
                else None
            )
            for block in _dict_items(message.get("content")):
                if block.get("type") == "toolCall":
                    call = block
                    arguments = call.get("arguments")
                    exchange = ToolExchange(
                        call_id=str(call.get("id")),
                        tool=str(call.get("name")),
                        arguments=cast("dict[str, Any]", arguments)
                        if isinstance(arguments, dict)
                        else {},
                        request_index=index,
                    )
                    by_id[exchange.call_id] = exchange
                    session.exchanges.append(exchange)
        elif role == "toolResult":
            exchange = by_id.get(str(message.get("toolCallId")))
            if exchange is None:
                continue
            exchange.answered = True
            exchange.is_error = message.get("isError") is True
            exchange.result_blocks = [
                str(block.get("text", ""))
                for block in _dict_items(message.get("content"))
                if block.get("type") == "text"
            ]
    return session


def _dict_items(value: object) -> list[dict[str, Any]]:
    """The JSON-object members of ``value`` when it is a list, else nothing."""
    if not isinstance(value, list):
        return []
    return [
        cast("dict[str, Any]", item)
        for item in cast("list[object]", value)
        if isinstance(item, dict)
    ]


# --- rate limits, capacity, credits, and credential hygiene -----------------------------
#
# Muna answers a request it cannot serve with HTTP 429: ``model_loading`` while a model is
# swapped into shared GPU capacity, ``model_capacity_exhausted`` when no capacity is free
# without displacing an active model, and an ordinary rate limit. omp retries those inside
# the run (``retry.maxRetries`` attempts, each waiting at most ``retry.maxDelayMs``); when
# its budget is spent it ends the turn with the provider error. The harness classifies that
# error here: a rate-limit or capacity block is re-run by the suite after a cooldown, and
# credit exhaustion (never retried, never scored) stops the suite with an exit status that
# can be resumed once the account is topped up.

_QUOTA_ERROR = re.compile(
    r"usage[\s_-]?limit|rate[\s_-]?limit|quota|too many requests|\b429\b"
    r"|spend(?:ing)?[\s_-]?(?:limit|cap)|resource[\s_-]?exhausted"
    r"|model[\s_-]?loading|capacity[\s_-]?exhausted"
    r"|\b(?:subscription|plan|membership)\b[^\n]{0,80}\b(?:limit|cap)\b",
    re.IGNORECASE,
)

_CREDIT_ERROR = re.compile(
    r"\b402\b|insufficient[\s_-]?(?:credits?|balance|funds)|payment[\s_-]?required"
    r"|credits_required|out[\s_-]of[\s_-]credits",
    re.IGNORECASE,
)


def is_quota_error(message: str | None) -> bool:
    """Whether a provider error text is a rate-limit or capacity block (Muna HTTP 429)."""
    return bool(message) and _QUOTA_ERROR.search(message or "") is not None


def is_credit_error(message: str | None) -> bool:
    """Whether a provider error text says the account is out of credits (HTTP 402 and kin).

    The response shape Muna uses for exhausted credits is unverified; this pattern is the
    protocol's reading of the usual ones. Check it before `is_quota_error`.
    """
    return bool(message) and _CREDIT_ERROR.search(message or "") is not None


@dataclass(frozen=True)
class OmpRunEvents:
    """What omp's ``--mode json`` event stream says about retries inside one run."""

    retries: int = 0
    quota_retries: int = 0
    quota_wait_seconds: float = 0.0
    gave_up: str | None = None
    tool_calls_started: int = 0


def parse_omp_events(stdout: str) -> OmpRunEvents:
    """Count omp's own retries (``auto_retry_start``) and note a retry loop it gave up on."""
    retries = quota_retries = tool_calls = 0
    wait_ms = 0.0
    gave_up: str | None = None
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event_obj: object = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event_obj, dict):
            continue
        event = cast("dict[str, Any]", event_obj)
        kind = event.get("type")
        if kind == "auto_retry_start":
            retries += 1
            if is_quota_error(str(event.get("errorMessage") or "")):
                quota_retries += 1
                wait_ms += float(_int(event.get("delayMs")))
        elif kind == "auto_retry_end" and event.get("success") is False:
            gave_up = str(event.get("finalError") or "retry failed")
        elif kind == "tool_execution_start":
            tool_calls += 1
    return OmpRunEvents(
        retries=retries,
        quota_retries=quota_retries,
        quota_wait_seconds=round(wait_ms / 1000.0, 3),
        gave_up=gave_up,
        tool_calls_started=tool_calls,
    )


def credential_files_in_profile(profile_dir: Path) -> list[str]:
    """Credential stores under an omp profile directory (relative paths, never contents).

    The agent container's only credential is the ``MUNA_ACCESS_KEY`` variable, so a profile
    that is copied into it must carry none: no ``agent.db`` (omp's SQLite login vault), token
    file, or encrypted snapshot.
    """
    if not profile_dir.is_dir():
        return []
    return sorted(
        str(path.relative_to(profile_dir))
        for path in profile_dir.rglob("*")
        if path.is_file()
        and (path.name.startswith("agent.db") or path.suffix in (".token", ".enc"))
    )


def is_emulated(runtime: str, host_machine: str) -> bool:
    """Whether a cell's linux/amd64 image runs under x86 emulation on this host.

    Campaign images are always started with ``--platform linux/amd64``, so a docker cell on an
    arm64 host (Apple silicon: Rosetta or QEMU) is emulated; the local runtime is native.
    """
    return runtime == "docker" and host_machine.lower() not in ("x86_64", "amd64")


# --- annotation ledger ----------------------------------------------------------


class HookLedgerSummary(BaseModel):
    """Hook activity from the extension's own per-call ledger (ground truth)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: int = Field(ge=0)
    eligible: int = Field(ge=0)
    annotated: int = Field(ge=0)
    units: int = Field(ge=0)
    tokens: int = Field(ge=0)
    fail_open: dict[str, int] = Field(default_factory=dict)
    annotated_after_first_edit: int = Field(default=0, ge=0)
    not_fresh_after_first_edit: int = Field(default=0, ge=0)
    """Eligible calls declined because the agent's edits left the index behind the tree.

    Kill criterion 5 excludes exactly these from the hook-activity denominator; a stale decline
    before the first edit is not excluded (the index is fresh at setup, so it would be a defect).
    """


STALE_INDEX_REASON = "index_not_fresh"
"""The annotate core's decline reason when the index is not ``fresh`` (``archex.annotate``)."""


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row: object = json.loads(line)
            if isinstance(row, dict):
                rows.append(cast("dict[str, Any]", row))
    return rows


def summarize_ledger(
    rows: Sequence[Mapping[str, Any]], first_edit_call_ids: set[str]
) -> HookLedgerSummary:
    """Totals over one cell's ledger; ``first_edit_call_ids`` are calls after the first edit."""
    reasons: Counter[str] = Counter()
    for row in rows:
        if not row.get("annotated"):
            reasons[str(row.get("reason"))] += 1
    return HookLedgerSummary(
        results=len(rows),
        eligible=sum(1 for row in rows if row.get("eligible")),
        annotated=sum(1 for row in rows if row.get("annotated")),
        units=sum(_int(row.get("units")) for row in rows if row.get("annotated")),
        tokens=sum(_int(row.get("tokens")) for row in rows if row.get("annotated")),
        fail_open=dict(sorted(reasons.items())),
        annotated_after_first_edit=sum(
            1
            for row in rows
            if row.get("annotated") and str(row.get("toolCallId")) in first_edit_call_ids
        ),
        not_fresh_after_first_edit=sum(
            1
            for row in rows
            if row.get("eligible")
            and not row.get("annotated")
            and row.get("reason") == STALE_INDEX_REASON
            and str(row.get("toolCallId")) in first_edit_call_ids
        ),
    )


# --- attribution and localization -------------------------------------------------

_READ_SELECTOR = re.compile(r"^(?P<path>.+?)(?::(?:\d|raw|img|conflicts)[^/]*)?$")
_EDIT_TARGET = re.compile(r"^\[(?P<path>[^\]#\n]+)(?:#[0-9A-F]{4})?\]", re.MULTILINE)


def observations(
    session: OmpSession,
    annotation_tokens: Mapping[str, int],
    count: Callable[[str], int],
) -> list[tuple[int, str, int]]:
    """``(request_index, channel, tokens)`` for every answered tool result.

    ``annotation_tokens`` maps a tool-call id to the tokens its appended
    annotation added, from the hook's ledger; those tokens are charged to
    ``archex-annotation`` and removed from the call's own channel.
    """
    rows: list[tuple[int, str, int]] = []
    for exchange in session.exchanges:
        if not exchange.answered:
            continue
        total = count(exchange.result_text)
        annotation = min(total, annotation_tokens.get(exchange.call_id, 0))
        channel = omp_channel(exchange.tool, exchange.arguments)
        rows.append((exchange.request_index, channel, total - annotation))
        if annotation:
            rows.append((exchange.request_index, "archex-annotation", annotation))
    return rows


def _normalize(path: str, repo_prefixes: Sequence[str]) -> str:
    text = path.strip().strip("'\"")
    for prefix in repo_prefixes:
        if prefix and text.startswith(prefix.rstrip("/") + "/"):
            text = text[len(prefix.rstrip("/")) + 1 :]
            break
    return text.removeprefix("./")


def read_target(tool: str, arguments: Mapping[str, object]) -> str | None:
    """The file a read-shaped call opens, before normalization."""
    if tool == "read":
        raw = arguments.get("path")
        if isinstance(raw, str) and "://" not in raw:
            match = _READ_SELECTOR.match(raw.strip())
            return match.group("path") if match else raw
        return None
    if tool == "bash":
        raw = arguments.get("command")
        return viewed_path(bash_command_body(raw)) if isinstance(raw, str) else None
    return None


def edit_targets(tool: str, arguments: Mapping[str, object]) -> list[str]:
    """Files an edit-shaped call writes, before normalization."""
    if tool == "write":
        raw = arguments.get("path")
        return [raw] if isinstance(raw, str) else []
    if tool == "edit":
        raw = arguments.get("input")
        if isinstance(raw, str):
            return [match.group("path") for match in _EDIT_TARGET.finditer(raw)]
        path = arguments.get("path")
        return [path] if isinstance(path, str) else []
    return []


def localize(
    session: OmpSession,
    gold_files: Sequence[str],
    patch_files: Sequence[str],
    repo_prefixes: Sequence[str],
) -> Localization:
    """When the agent first read and first edited a gold file, and what it edited."""
    first_read: int | None = None
    first_edit: int | None = None
    read: set[str] = set()
    for exchange in session.exchanges:
        target = read_target(exchange.tool, exchange.arguments)
        if target is not None:
            path = _normalize(target, repo_prefixes)
            read.add(path)
            if first_read is None and touches(path, gold_files):
                first_read = exchange.request_index
        for raw in edit_targets(exchange.tool, exchange.arguments):
            if first_edit is None and touches(_normalize(raw, repo_prefixes), gold_files):
                first_edit = exchange.request_index
    return Localization(
        gold_files=sorted(gold_files),
        first_gold_read_request=first_read,
        first_gold_edit_request=first_edit,
        read_all_gold=bool(gold_files) and all(touches(g, read) for g in gold_files),
        edited_non_gold=any(not touches(p, gold_files) for p in patch_files),
    )


def out_of_patch_read_tokens_compounded(
    session: OmpSession,
    patch_files: Sequence[str],
    repo_prefixes: Sequence[str],
    count: Callable[[str], int],
) -> int:
    """Compounded tokens of ``read``-channel results outside the task's gold and test patches.

    The lever the headroom gate measures (kill criterion 3), defined as in
    ``scripts/swebench_pro_token_headroom.py``: a read of a file either accepted patch touches is
    not displaceable (the agent must see those lines to patch them); every other read is, and so
    is a read whose target cannot be resolved. Weighted like `compound`: a result produced by
    request ``i`` of ``n`` is charged ``n - 1 - i`` times.
    """
    request_count = len(session.requests)
    total = 0
    for exchange in session.exchanges:
        if not exchange.answered or omp_channel(exchange.tool, exchange.arguments) != "read":
            continue
        target = read_target(exchange.tool, exchange.arguments)
        if target is not None and touches(_normalize(target, repo_prefixes), patch_files):
            continue
        weight = max(0, request_count - 1 - exchange.request_index)
        total += count(exchange.result_text) * weight
    return total


# --- cell schema ------------------------------------------------------------------


class CellStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class FailureReason(StrEnum):
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    HARNESS_ERROR = "harness_error"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    SESSION_UNPARSABLE = "session_unparsable"
    QUOTA_BLOCK = "quota_block"
    CREDIT_EXHAUSTED = "credit_exhausted"


class Usage(BaseModel):
    """Provider-reported tokens summed over every request, and omp's cost figure.

    ``cost_usd`` is omp's cost model of the tokens at the prices in the frozen provider
    config (``benchmarks/swe_ab/muna-models.yml``); it feeds the cost ceiling and money
    translation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: int = Field(ge=0)
    output: int = Field(ge=0)
    cache_read: int = Field(ge=0)
    cache_write: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)

    @property
    def total_billed(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


class Isolation(BaseModel):
    """What the cell's own evidence shows about ambient context and tool surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tools_source: Literal["request_capture", "declared"]
    tools_advertised: list[str]
    undeclared_tool_calls: list[str] = Field(default_factory=list)
    system_prompt_checked: bool
    system_prompt_violations: list[str] = Field(default_factory=list)
    compressor_marker_seen: bool
    annotation_seen: bool
    non_message_tokens: int | None = None


class Localization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_files: list[str]
    first_gold_read_request: int | None = None
    first_gold_edit_request: int | None = None
    read_all_gold: bool = False
    edited_non_gold: bool = False


class QuotaEvidence(BaseModel):
    """Rate-limit, capacity, and credit-block activity around one cell (counts and phase only)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    omp_retries: int = Field(default=0, ge=0)
    """omp's own in-run retries whose error was a rate limit or quota block."""
    omp_retry_wait_seconds: float = Field(default=0.0, ge=0.0)
    block_phase: Literal["before_first_tool_call", "mid_run"] | None = None
    """Set iff a block ended the run: before any tool call, or after work had begun."""
    prior_blocked_attempts: int = Field(default=0, ge=0)
    """Attempts of this cell filed ahead of this artifact because a block ended them."""


class SweAbCell(BaseModel):
    """One recorded cell; counts, identities, and scores only — no prompt or source text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: Literal[2] = 2
    preregistration: str = PREREGISTRATION_PATH
    task_id: str
    repo: str
    model: str
    arm: SweAbArm
    repetition: int = Field(ge=1)
    status: CellStatus
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None
    retried: bool = False

    omp_version: str
    archex_version: str | None
    archex_wheel_sha256: str | None
    hook_module_sha256: str | None
    cli_guide_sha256: str | None
    image: str
    thinking: str
    tool_fingerprint: str
    provider: str | None
    provider_base_url: str | None
    """The ``baseUrl`` the cell's provider resolved to: Muna's, or a rehearsal stub's."""
    provider_config_sha256: str
    """SHA-256 of the provider config installed as the profile's ``models.yml``."""
    omp_config_sha256: str
    """SHA-256 of the omp settings overlay passed with ``--config``."""
    emulated: bool
    """The image ran under x86 emulation on an arm64 host; emulated and native cells never mix."""
    network: str
    """Docker network the agent container ran on (egress posture, identical across a stage)."""
    credential_env_names: list[str]
    """Names, never values, of the credential variables the agent container received."""

    usage: Usage
    requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    tool_call_mix: dict[str, int] = Field(default_factory=dict)
    channel_tokens_once: dict[str, int]
    channel_tokens_compounded: dict[str, int]
    out_of_patch_read_tokens_compounded: int = Field(ge=0)
    """The ``read`` share of ``channel_tokens_compounded`` outside the gold and test patches."""
    channel_tokenizer: str = "cl100k_base"
    hook_ledger: HookLedgerSummary | None
    archex_cli_calls: int = Field(ge=0)
    archex_cli_subcommands: dict[str, int] = Field(default_factory=dict)
    isolation: Isolation
    localization: Localization

    patch_sha256: str
    patch_bytes: int = Field(ge=0)
    patch_files: list[str] = Field(default_factory=list)
    resolved: bool | None
    score_source: Literal["pro_verifier", "not_scored", "failed_before_patch"]

    wall_seconds: float = Field(ge=0.0)
    setup_seconds: float = Field(ge=0.0)
    index_seconds: float | None = Field(default=None, ge=0.0)
    annotate_prewarm_seconds: float | None = Field(default=None, ge=0.0)
    quota: QuotaEvidence

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.omp_version != OMP_VERSION:
            raise ValueError(f"omp {self.omp_version} is not the pinned {OMP_VERSION}")
        if self.model not in MODELS:
            raise ValueError(f"{self.model!r} is not a campaign configuration")
        if self.thinking != configuration(self.model).thinking:
            raise ValueError(
                f"thinking {self.thinking!r} does not match configuration {self.model!r}"
            )
        blocked_reason = self.failure_reason in (
            FailureReason.QUOTA_BLOCK,
            FailureReason.CREDIT_EXHAUSTED,
        )
        if blocked_reason != (self.quota.block_phase is not None):
            raise ValueError(
                "a block phase is recorded iff the failure reason is quota_block or "
                "credit_exhausted"
            )
        if set(self.channel_tokens_once) != set(CHANNELS) or set(
            self.channel_tokens_compounded
        ) != set(CHANNELS):
            raise ValueError("channel totals must cover exactly the frozen channel set")
        if self.out_of_patch_read_tokens_compounded > self.channel_tokens_compounded["read"]:
            raise ValueError("out-of-patch read tokens exceed the read channel's compounded total")
        arm = self.arm
        if arm.archex_installed != (self.archex_version is not None):
            raise ValueError(f"arm {arm} archex installation does not match archex_version")
        if arm.hook != (self.hook_module_sha256 is not None) or arm.hook != (
            self.hook_ledger is not None
        ):
            raise ValueError(f"arm {arm} must carry a hook module and ledger iff it has the hook")
        if arm.cli != (self.cli_guide_sha256 is not None):
            raise ValueError(f"arm {arm} must carry the CLI guide iff it has the CLI")
        if not arm.hook and (
            self.channel_tokens_once["archex-annotation"] or self.isolation.annotation_seen
        ):
            raise ValueError(f"annotation evidence in arm {arm}, which has no hook")
        if self.status is CellStatus.FAILED:
            if self.failure_reason is None:
                raise ValueError("a failed cell must record its failure reason")
            if self.resolved:
                raise ValueError("a failed cell scores unresolved")
            return self
        if self.failure_reason is not None:
            raise ValueError("an ok cell carries no failure reason")
        isolation = self.isolation
        if sorted(isolation.tools_advertised) != sorted(BASE_TOOLS):
            raise ValueError("an ok cell must advertise exactly the base tool set")
        if self.tool_fingerprint != tool_fingerprint(isolation.tools_advertised):
            raise ValueError("tool fingerprint does not match the advertised tools")
        if (
            isolation.undeclared_tool_calls
            or isolation.system_prompt_violations
            or isolation.compressor_marker_seen
        ):
            raise ValueError("an ok cell must pass every isolation check")
        return self

    @property
    def key(self) -> CellKey:
        return CellKey(self.task_id, self.model, self.arm, self.repetition)


def tool_fingerprint(tools: Iterable[str]) -> str:
    return hashlib.sha256(",".join(sorted(tools)).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- plans and the directory validator -------------------------------------------


@dataclass(frozen=True, order=True)
class CellKey:
    task_id: str
    model: str
    arm: SweAbArm
    repetition: int

    @property
    def relative_path(self) -> Path:
        return (
            Path(model_slug(self.model))
            / self.arm.value
            / cell_filename(self.task_id, self.repetition)
        )


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def cell_filename(task_id: str, repetition: int) -> str:
    return f"{task_id}__rep{repetition}.json"


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    repo: str
    image: str | None = None
    task_dir: str | None = None
    local_repo: str | None = None
    prompt_file: str | None = None


class SweAbPlan(BaseModel):
    """The declared cells of one stage: tasks × models × arms × repetitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tasks: list[PlanTask] = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    repetitions: dict[SweAbArm, int] = Field(min_length=1)
    cost_ceiling_usd: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("plan task ids must be unique")
        if any(count < 1 for count in self.repetitions.values()):
            raise ValueError("every declared arm needs at least one repetition")
        unknown = [model for model in self.models if model not in MODELS]
        if unknown or len(self.models) != len(set(self.models)):
            raise ValueError(
                f"plan models must be distinct campaign configurations {list(MODELS)}; "
                f"got {self.models}"
            )
        return self

    def cells(self) -> list[CellKey]:
        return [
            CellKey(task.task_id, model, arm, repetition)
            for task in self.tasks
            for model in self.models
            for arm in SweAbArm
            if arm in self.repetitions
            for repetition in range(1, self.repetitions[arm] + 1)
        ]

    def scheduled_cells(self) -> list[CellKey]:
        """Cells in the order the suite runs them, one model family loaded at a time.

        Muna swaps models in and out of shared GPU capacity, so cells are ordered by
        (omp selector, task in plan order, configuration label, arm, repetition): every cell
        of a family, task by task, before the next family begins.
        """
        task_index = {task.task_id: index for index, task in enumerate(self.tasks)}
        return sorted(
            self.cells(),
            key=lambda key: (
                configuration(key.model).selector,
                task_index[key.task_id],
                key.model,
                key.arm.value,
                key.repetition,
            ),
        )


def load_plan(path: Path) -> SweAbPlan:
    try:
        return SweAbPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise SweAbError(f"invalid plan {path}: {exc}") from exc


def load_cell(path: Path) -> SweAbCell:
    try:
        return SweAbCell.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise SweAbError(f"invalid cell {path}: {exc}") from exc


class SweAbCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    declared: int
    present: int
    ok: int
    failed: int
    quota_blocked: int = 0
    """Attempts filed for a rate-limit, capacity, or credit block; never scored."""
    total_cost_usd: float


def quota_blocked_relative_path(key: CellKey, attempt: int) -> Path:
    """Where attempt ``attempt`` of ``key`` lives after a block sent it back to be re-run."""
    return Path(QUOTA_BLOCKED_DIR) / key.relative_path.with_suffix(f".attempt{attempt}.json")


def _quota_blocked_cells(directory: Path, declared: set[CellKey]) -> list[SweAbCell]:
    blocked: list[SweAbCell] = []
    root = directory / QUOTA_BLOCKED_DIR
    for path in sorted(root.rglob("*.json")) if root.is_dir() else []:
        cell = load_cell(path)
        if cell.failure_reason not in (FailureReason.QUOTA_BLOCK, FailureReason.CREDIT_EXHAUSTED):
            raise SweAbError(f"{path} is filed as a blocked attempt but is not one")
        if cell.key not in declared:
            raise SweAbError(f"{path} is not a declared cell")
        blocked.append(cell)
    return blocked


def validate_swe_ab_directory(
    directory: Path, plan: SweAbPlan, *, require_complete: bool = True
) -> SweAbCoverage:
    """Validate a result directory for publication; raise `SweAbError` on any violation."""
    if not directory.is_dir():
        raise SweAbError(f"result directory {directory} does not exist")
    declared = set(plan.cells())
    cells: dict[CellKey, SweAbCell] = {}
    for path in sorted(directory.rglob("*.json")):
        if path.relative_to(directory).parts[0] == QUOTA_BLOCKED_DIR:
            continue
        cell = load_cell(path)
        key = cell.key
        if path.relative_to(directory) != key.relative_path:
            raise SweAbError(f"{path} does not sit at its key's path {key.relative_path}")
        if key not in declared:
            raise SweAbError(f"{path} is not a declared cell of plan {plan.name!r}")
        if cell.provider_base_url != MUNA_BASE_URL:
            raise SweAbError(
                f"{path} resolved provider endpoint {cell.provider_base_url!r}, not "
                f"{MUNA_BASE_URL} (a local stub or another endpoint); such cells exercise the "
                "harness and are refused for publication"
            )
        if cell.failure_reason in (FailureReason.QUOTA_BLOCK, FailureReason.CREDIT_EXHAUSTED):
            raise SweAbError(
                f"{path} ended in a {cell.failure_reason} block, which is not a task outcome; "
                "resume the run so the cell is re-run once the block clears"
            )
        if sorted(cell.credential_env_names) != sorted(CREDENTIAL_ENV_NAMES):
            raise SweAbError(
                f"{path} did not authenticate with exactly {list(CREDENTIAL_ENV_NAMES)} "
                f"(credential variables: {sorted(cell.credential_env_names)})"
            )
        if cell.provider is not None and cell.provider != CAMPAIGN_PROVIDER:
            raise SweAbError(
                f"{path} ran {cell.model} through provider {cell.provider!r}, "
                f"not {CAMPAIGN_PROVIDER!r}"
            )
        cells[key] = cell
    if require_complete and (missing := sorted(declared - set(cells))):
        raise SweAbError(f"{len(missing)} declared cells are missing, first: {missing[0]}")
    blocked = _quota_blocked_cells(directory, declared)
    ok = [cell for cell in cells.values() if cell.status is CellStatus.OK]
    _require_single("omp_version", {cell.omp_version for cell in cells.values()})
    _require_single("provider config", {cell.provider_config_sha256 for cell in cells.values()})
    _require_single("omp config", {cell.omp_config_sha256 for cell in cells.values()})
    if len({cell.emulated for cell in cells.values()}) > 1:
        raise SweAbError(
            "emulated and native cells are mixed within the stage; wall times and timeouts are "
            "not comparable across them, so re-run the stage on one kind of host"
        )
    _require_single("network", {cell.network for cell in cells.values()})
    _require_single("tool_fingerprint", {cell.tool_fingerprint for cell in ok})
    for arm in SweAbArm:
        in_arm = [cell for cell in cells.values() if cell.arm is arm]
        _require_single(f"{arm} archex wheel", {c.archex_wheel_sha256 for c in in_arm})
        _require_single(f"{arm} hook module", {c.hook_module_sha256 for c in in_arm})
        _require_single(f"{arm} CLI guide", {c.cli_guide_sha256 for c in in_arm})
    for model in plan.models:
        for arm in SweAbArm:
            prefixes = {
                cell.isolation.non_message_tokens
                for cell in ok
                if cell.model == model and cell.arm is arm and cell.isolation.non_message_tokens
            }
            _require_single(f"{model} {arm} static prompt size", prefixes)
    total_cost = sum(cell.usage.cost_usd for cell in [*cells.values(), *blocked])
    largest = max((cell.usage.cost_usd for cell in [*cells.values(), *blocked]), default=0.0)
    if total_cost > plan.cost_ceiling_usd + largest:
        raise SweAbError(
            f"recorded cost ${total_cost:.2f} exceeds the ${plan.cost_ceiling_usd:.2f} ceiling"
        )
    return SweAbCoverage(
        declared=len(declared),
        present=len(cells),
        ok=len(ok),
        failed=len(cells) - len(ok),
        quota_blocked=len(blocked),
        total_cost_usd=round(total_cost, 4),
    )


def _require_single(label: str, values: set[Any]) -> None:
    if len(values) > 1:
        raise SweAbError(f"{label} differs across cells: {sorted(map(str, values))}")


# --- the frozen agent invocation ----------------------------------------------------


def omp_argv(
    omp_command: Sequence[str],
    *,
    arm: SweAbArm,
    model: str,
    prompt_path: str,
    session_dir: str,
    hook_module_path: str | None,
    cli_guide_path: str | None,
    omp_config_path: str,
) -> list[str]:
    """The one omp command line every cell runs (spec §5.3).

    ``model`` is a configuration label; its omp selector and thinking effort are derived
    from `CONFIGURATIONS`. Identical across arms except the explicit annotation extension
    (H, HC) and the appended CLI guide (HC, C). ``--no-title`` keeps omp from spending a
    model call on a session title; ``--config`` loads the frozen retry overlay.
    """
    config = configuration(model)
    if arm.hook != (hook_module_path is not None):
        raise SweAbError(f"arm {arm} {'needs' if arm.hook else 'must not load'} the hook module")
    if arm.cli != (cli_guide_path is not None):
        raise SweAbError(f"arm {arm} {'needs' if arm.cli else 'must not append'} the CLI guide")
    argv = [
        *omp_command,
        "-p",
        "--profile",
        PROFILE,
        "--config",
        omp_config_path,
        "--model",
        config.selector,
        "--thinking",
        config.thinking,
        "--tools",
        ",".join(BASE_TOOLS),
        *ISOLATION_FLAGS,
    ]
    if hook_module_path is not None:
        argv += ["-e", hook_module_path]
    if cli_guide_path is not None:
        argv += ["--append-system-prompt", cli_guide_path]
    argv += [
        "--approval-mode",
        "yolo",
        "--mode",
        "json",
        "--session-dir",
        session_dir,
        "--max-time",
        MAX_TIME,
        "--no-title",
        f"@{prompt_path}",
    ]
    return argv


def system_prompt_violations(system_prompt: str) -> list[str]:
    """Ambient context the rendered system prompt must not carry in any arm."""
    return [label for label, marker in SYSTEM_PROMPT_FORBIDDEN if marker in system_prompt]


_SEARCH_TOOL_WORDS = re.compile(
    r"`(?:grep|glob|find|rg|search|ast_grep|archex[^`]*)`|\b(?:grep|glob|ripgrep)\b", re.IGNORECASE
)


def search_routing_rules(system_prompt: str) -> list[str]:
    """Lines of the system prompt that route work to a search tool, for disclosure."""
    return [
        line.strip()
        for line in system_prompt.splitlines()
        if line.strip() and _SEARCH_TOOL_WORDS.search(line)
    ]


def _provider_entry(models_yml: Path, provider: str) -> dict[str, Any] | None:
    document: object = yaml.safe_load(models_yml.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    providers = cast("dict[str, Any]", document).get("providers")
    if not isinstance(providers, dict):
        return None
    entry = cast("dict[str, Any]", providers).get(provider)
    return cast("dict[str, Any]", entry) if isinstance(entry, dict) else None


def provider_base_url(models_yml: Path, provider: str) -> str | None:
    """The ``baseUrl`` ``models_yml`` declares for ``provider``; `None` if absent."""
    entry = _provider_entry(models_yml, provider) if models_yml.is_file() else None
    base_url = entry.get("baseUrl") if entry else None
    return str(base_url) if base_url else None


PRICED_FIELDS: tuple[str, ...] = ("input", "output", "cacheRead")
"""Price fields (USD per million tokens) every campaign model must carry above zero.

``cacheWrite`` is exempt: an ``openai-completions`` response reports no cache-write tokens.
"""


def unpriced_models(provider_config: Path) -> list[str]:
    """``"<model id>: <field>"`` for each price the provider config leaves at zero or absent.

    The cost ceiling is checked against omp's per-request ``usage.cost.total``, which is zero
    for an unpriced model; a docker run therefore refuses to start while this list is non-empty.
    """
    entry = _provider_entry(provider_config, CAMPAIGN_PROVIDER)
    missing: list[str] = []
    for model in _dict_items(entry.get("models") if entry else None):
        cost = model.get("cost")
        prices = cast("dict[str, Any]", cost) if isinstance(cost, dict) else {}
        for name in PRICED_FIELDS:
            value = prices.get(name)
            if not isinstance(value, int | float) or value <= 0:
                missing.append(f"{model.get('id')}: {name}")
    return missing
