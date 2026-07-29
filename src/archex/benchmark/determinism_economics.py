"""Provider-observed S7 prefix-cache economics measurement.

This benchmark-only module freezes Archex-selected context before contacting
OpenRouter. It never changes the product retrieval path or evaluates answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from archex.api import query
from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import clone_at_commit
from archex.benchmark.strategies import benchmark_repo_source
from archex.models import Config, IndexConfig

EVIDENCE_VERSION = 1
FIXTURE_VERSION = 1
MODEL = "anthropic/claude-opus-5"
PROVIDER = "Anthropic"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_RESAMPLES = 10_000
STUDY_INSTRUCTION = (
    "You are answering a frozen software-maintenance benchmark. "
    "Use only the selected source context."
)
CACHE_TTL_SECONDS = 301
TURN_DELAY_SECONDS = 60

TurnOrders: TypeAlias = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


class DeterminismEconomicsError(ValueError):
    """Raised when S7 fixture, receipt, or evidence invariants fail."""


class OrderingArm(StrEnum):
    """The pre-registered ordering arms."""

    DETERMINISTIC = "deterministic"
    PERTURBED = "perturbed"
    ANN_BASELINE = "ann_baseline"


class FrozenChunk(BaseModel):
    """One selected source chunk frozen before any provider request."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    dense_score: float


class FrozenSession(BaseModel):
    """A three-turn, source-pinned, resolved benchmark session."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    task_id: str
    repo: str
    commit: str
    resolved: bool
    turns: tuple[str, str, str]
    chunks: list[FrozenChunk] = Field(min_length=2)
    orders: dict[OrderingArm, TurnOrders]

    @model_validator(mode="after")
    def validate_orders(self) -> FrozenSession:
        chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if not self.resolved:
            raise ValueError("every S7 session must have resolved=true")
        if set(self.orders) != set(OrderingArm):
            raise ValueError("S7 fixture must contain exactly three ordering arms")
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ValueError("S7 fixture must record each immutable 40-character source SHA")
        for arm, turn_orders in self.orders.items():
            if len(turn_orders) != 3:
                raise ValueError(f"{arm} requires exactly three turn orders")
            for order in turn_orders:
                if len(order) != len(chunk_ids) or set(order) != set(chunk_ids):
                    raise ValueError(f"{arm} order must be a permutation of frozen chunks")
        return self


class SessionFixture(BaseModel):
    """Committed source and ordering fixture, excluding generated receipts."""

    model_config = ConfigDict(frozen=True)

    version: int = FIXTURE_VERSION
    source_revision: str
    retrieval_timestamp: str
    sessions: list[FrozenSession] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def validate_sessions(self) -> SessionFixture:
        ids = [session.session_id for session in self.sessions]
        if len(set(ids)) != 12:
            raise ValueError("S7 fixture requires exactly 12 unique repository clusters")
        repos = [session.repo for session in self.sessions]
        if len(set(repos)) != 12:
            raise ValueError("S7 fixture requires exactly 12 unique repositories")
        return self


class ProviderReceipt(BaseModel):
    """Auditable OpenRouter response for one frozen rendered prefix."""

    model_config = ConfigDict(frozen=True)

    arm: OrderingArm
    session_id: str
    turn_index: int = Field(ge=1, le=3)
    phase: str
    rendered_prefix_sha256: str
    request_timestamp: str
    response_timestamp: str
    model: str
    requested_provider: dict[str, Any]
    provider: str
    generation_id: str
    usage: dict[str, Any]
    total_cost: float
    upstream_inference_prompt_cost: float
    completion_cost: float
    prompt_tokens: int
    cache_write_tokens: int
    cached_tokens: int
    completion_tokens: int

    @model_validator(mode="after")
    def validate_receipt(self) -> ProviderReceipt:
        if self.model != MODEL or self.provider != PROVIDER:
            raise ValueError("receipt must resolve the pre-registered Anthropic model/provider")
        if self.requested_provider != {"only": ["anthropic"], "allow_fallbacks": False}:
            raise ValueError("receipt requested routing differs from preregistration")
        if not self.generation_id:
            raise ValueError("receipt lacks generation id")
        if self.prompt_tokens <= 0:
            raise ValueError("receipt must contain nonzero prompt_tokens")
        if self.upstream_inference_prompt_cost <= 0:
            raise ValueError("receipt must contain nonzero upstream prompt cost")
        return self


class DeterminismEconomicsArtifact(BaseModel):
    """Validated evidence emitted only after all provider receipts succeed."""

    model_config = ConfigDict(frozen=True)

    version: int = EVIDENCE_VERSION
    preregistration_commit: str
    fixture_sha256: str
    source_revision: str
    model: str
    sessions: list[FrozenSession]
    preflight_receipts: list[ProviderReceipt]
    measurement_receipts: list[ProviderReceipt]
    summary: dict[str, Any]
    generated_at: str

    @model_validator(mode="after")
    def validate_artifact(self) -> DeterminismEconomicsArtifact:
        fixture = SessionFixture(
            source_revision=self.source_revision,
            retrieval_timestamp="artifact-embedded",
            sessions=self.sessions,
        )
        if fixture_digest(fixture) != self.fixture_sha256:
            raise ValueError("fixture digest does not match embedded frozen sessions")
        validate_provider_receipts(self.preflight_receipts, fixture, preflight=True)
        validate_provider_receipts(self.measurement_receipts, fixture, preflight=False)
        if self.summary.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES:
            raise ValueError("artifact must use the pre-registered bootstrap count")
        return self


def utc_now() -> str:
    """Return an unambiguous UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fixture_digest(fixture: SessionFixture) -> str:
    """Hash stable fixture content independently of retrieval timestamp."""
    payload = fixture.model_dump(mode="json", exclude={"retrieval_timestamp"})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _seeded_permutation(chunk_ids: list[str], seed_text: str) -> tuple[str, ...]:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:16], "big")
    return tuple(np.random.Generator(np.random.PCG64(seed)).permutation(chunk_ids).tolist())


def perturbed_orders(session_id: str, chunk_ids: list[str]) -> TurnOrders:
    orders: list[tuple[str, ...]] = []
    for turn_index in range(1, 4):
        order = _seeded_permutation(chunk_ids, f"20260729|{session_id}|{turn_index}")
        if turn_index > 1 and order == orders[-1]:
            order = (order[-1], *order[:-1])
        orders.append(order)
    return cast("TurnOrders", tuple(orders))


def ann_baseline_orders(session_id: str, chunks: list[FrozenChunk]) -> TurnOrders:
    rounded: dict[float, list[str]] = defaultdict(list)
    for chunk in chunks:
        rounded[round(chunk.dense_score, 4)].append(chunk.chunk_id)
    orders: list[tuple[str, ...]] = []
    for turn_index in range(1, 4):
        output: list[str] = []
        for score in sorted(rounded, reverse=True):
            group = rounded[score]
            if len(group) == 1:
                output.extend(group)
            else:
                output.extend(
                    _seeded_permutation(group, f"20260730|{session_id}|{turn_index}|{score:.4f}")
                )
        orders.append(tuple(output))
    return cast("TurnOrders", tuple(orders))


def build_fixture(
    *,
    task_ids: list[str],
    repository_root: Path,
    source_revision: str,
) -> SessionFixture:
    """Retrieve each pre-registered source once and freeze its selected context."""
    tasks_by_id = {task.task_id: task for task in load_tasks(repository_root / "benchmarks/tasks")}
    missing = set(task_ids) - set(tasks_by_id)
    if missing:
        raise DeterminismEconomicsError(f"unknown S7 task ids: {sorted(missing)}")
    sessions: list[FrozenSession] = []
    for task_id in task_ids:
        task = tasks_by_id[task_id]
        repo_path, cleanup, resolved_commit = _prepare_task_repo(
            task, repository_root, source_revision
        )
        try:
            source = benchmark_repo_source(task, repo_path, strategy=Strategy.ARCHEX_QUERY)
            bundle = query(
                source,
                task.question,
                token_budget=task.token_budget,
                explicit_token_budget=True,
                config=Config(cache=False, languages=task.languages),
                index_config=IndexConfig(vector=False),
            )
        finally:
            if cleanup:
                import shutil

                shutil.rmtree(repo_path, ignore_errors=True)
        chunks = [
            FrozenChunk(
                chunk_id=ranked.chunk.id,
                file_path=ranked.chunk.file_path,
                start_line=ranked.chunk.start_line,
                end_line=ranked.chunk.end_line,
                content=ranked.chunk.content,
                dense_score=ranked.final_score,
            )
            for ranked in bundle.chunks
        ]
        if len(chunks) < 2:
            raise DeterminismEconomicsError(f"{task_id} retrieved fewer than two chunks")
        session_id = task_id.replace("_", "-")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        turns = _turns_for_task(task)
        sessions.append(
            FrozenSession(
                session_id=session_id,
                task_id=task.task_id,
                repo=task.repo,
                commit=resolved_commit,
                resolved=True,
                turns=turns,
                chunks=chunks,
                orders={
                    OrderingArm.DETERMINISTIC: cast("TurnOrders", (tuple(chunk_ids),) * 3),
                    OrderingArm.PERTURBED: perturbed_orders(session_id, chunk_ids),
                    OrderingArm.ANN_BASELINE: ann_baseline_orders(session_id, chunks),
                },
            )
        )
    return SessionFixture(
        source_revision=source_revision, retrieval_timestamp=utc_now(), sessions=sessions
    )


def _prepare_task_repo(
    task: BenchmarkTask, repository_root: Path, source_revision: str
) -> tuple[Path, bool, str]:
    if task.repo == ".":
        repo_path = Path(tempfile.mkdtemp(prefix="archex-s7-self-"))
        subprocess.run(
            ["git", "clone", "--no-checkout", str(repository_root), str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", source_revision],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return repo_path, True, source_revision
    repo_path, cleanup = clone_at_commit(task.repo, task.commit)
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved_commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_commit):
        raise DeterminismEconomicsError(
            f"could not resolve immutable SHA for {task.repo} at {task.commit}"
        )
    return repo_path, cleanup, resolved_commit


def _turns_for_task(task: BenchmarkTask) -> tuple[str, str, str]:
    followups = {
        "archex_query_pipeline": (
            "Which code path assembles the final context bundle for that query?",
            "Identify the stable ordering point used before final context packing.",
        ),
        "celery_task_dispatch": (
            "Which components prepare a task message before a worker receives it?",
            "Where does the worker choose the execution strategy for that task?",
        ),
        "click_decorators": (
            "Which decorator factories attach parameters to a command?",
            "Which command and parameter classes consume those attached parameters?",
        ),
        "django_middleware": (
            "Where is the middleware stack loaded for a request?",
            "Which path applies CommonMiddleware to the request and response?",
        ),
        "fastapi_dependency_injection": (
            "Which function constructs the dependency tree for an endpoint?",
            "Which function resolves nested dependencies before the handler runs?",
        ),
        "loc_flask_blueprint_register": (
            "Which setup state is created during that registration?",
            "Where does registration recurse into a child blueprint?",
        ),
        "gin_routing": (
            "Which file defines the routing tree nodes?",
            "Which path connects a route group to the routing tree?",
        ),
        "httpx_pooling": (
            "Which transport owns the connection pool?",
            "Which client configuration controls keep-alive behavior?",
        ),
        "mini_redis_async": (
            "Which server component processes incoming commands?",
            "Which connection component reads and writes protocol frames?",
        ),
        "pydantic_validators": (
            "Where are field_validator and model_validator declared?",
            "Which internal validators support that validation pipeline?",
        ),
        "pytest_fixtures": (
            "Which fixture manager discovers available fixture definitions?",
            "Which path resolves fixture arguments for a test item?",
        ),
        "react_hooks": (
            "Which reconciler module owns hook state during rendering?",
            "Which public hooks module dispatches useState calls?",
        ),
    }
    try:
        second, third = followups[task.task_id]
    except KeyError as exc:
        raise DeterminismEconomicsError(f"no frozen follow-up turns for {task.task_id}") from exc
    return task.question, second, third


def render_context(session: FrozenSession, arm: OrderingArm, turn_index: int) -> str:
    """Render the cache-controlled selected context for one frozen arm/turn."""
    chunk_map = {chunk.chunk_id: chunk for chunk in session.chunks}
    ordered_chunks = [chunk_map[chunk_id] for chunk_id in session.orders[arm][turn_index - 1]]
    rendered_chunks = "\n\n".join(
        f"## {chunk.file_path}:{chunk.start_line}-{chunk.end_line}\n{chunk.content}"
        for chunk in ordered_chunks
    )
    return f"<selected-context>\n{rendered_chunks}\n</selected-context>"


def render_prefix(session: FrozenSession, arm: OrderingArm, turn_index: int) -> str:
    """Render all content that precedes the cached context breakpoint."""
    return f"{STUDY_INSTRUCTION}\n\n{render_context(session, arm, turn_index)}"


def request_payload(
    session: FrozenSession, arm: OrderingArm, turn_index: int
) -> tuple[dict[str, Any], str]:
    """Build the exact provider request and its cacheable-prefix SHA-256."""
    context = render_context(session, arm, turn_index)
    prefix_sha256 = hashlib.sha256(f"{STUDY_INSTRUCTION}\n\n{context}".encode()).hexdigest()
    return {
        "model": MODEL,
        "provider": {"only": ["anthropic"], "allow_fallbacks": False},
        "max_tokens": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": STUDY_INSTRUCTION},
                    {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": session.turns[turn_index - 1]},
                ],
            }
        ],
    }, prefix_sha256


def call_openrouter(
    *, session: FrozenSession, arm: OrderingArm, turn_index: int, phase: str, api_key: str
) -> ProviderReceipt:
    """Send one pinned request and fail loudly on any routing or receipt mismatch."""
    payload, prefix_sha256 = request_payload(session, arm, turn_index)
    request_timestamp = utc_now()
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            raw = cast("dict[str, Any]", json.loads(response.read().decode()))
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise DeterminismEconomicsError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DeterminismEconomicsError(f"OpenRouter request failed: {exc}") from exc
    return provider_receipt_from_response(
        raw=raw,
        session=session,
        arm=arm,
        turn_index=turn_index,
        phase=phase,
        prefix_sha256=prefix_sha256,
        request_timestamp=request_timestamp,
    )


def provider_receipt_from_response(
    *,
    raw: dict[str, Any],
    session: FrozenSession,
    arm: OrderingArm,
    turn_index: int,
    phase: str,
    prefix_sha256: str,
    request_timestamp: str,
) -> ProviderReceipt:
    usage_value = raw.get("usage")
    if not isinstance(usage_value, dict):
        error = raw.get("error")
        raise DeterminismEconomicsError(
            f"OpenRouter response lacks usage; fields={sorted(raw)}; error={error!r}"
        )
    usage = cast("dict[str, Any]", usage_value)
    details_value = usage.get("prompt_tokens_details")
    costs_value = usage.get("cost_details")
    if not isinstance(details_value, dict) or not isinstance(costs_value, dict):
        raise DeterminismEconomicsError("OpenRouter response lacks cache or cost details")
    details = cast("dict[str, Any]", details_value)
    costs = cast("dict[str, Any]", costs_value)
    try:
        return ProviderReceipt(
            arm=arm,
            session_id=session.session_id,
            turn_index=turn_index,
            phase=phase,
            rendered_prefix_sha256=prefix_sha256,
            request_timestamp=request_timestamp,
            response_timestamp=utc_now(),
            model=raw["model"],
            requested_provider={"only": ["anthropic"], "allow_fallbacks": False},
            provider=raw["provider"],
            generation_id=raw["id"],
            usage=usage,
            total_cost=float(usage["cost"]),
            upstream_inference_prompt_cost=float(costs["upstream_inference_prompt_cost"]),
            completion_cost=float(costs["upstream_inference_completions_cost"]),
            prompt_tokens=int(usage["prompt_tokens"]),
            cache_write_tokens=int(details.get("cache_write_tokens", 0)),
            cached_tokens=int(details.get("cached_tokens", 0)),
            completion_tokens=int(usage["completion_tokens"]),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise DeterminismEconomicsError(
            f"OpenRouter response has invalid provider receipt: {exc}"
        ) from exc


def run_preflight(fixture: SessionFixture, api_key: str) -> list[ProviderReceipt]:
    """Prewarm/replay the deterministic prefix once and every comparator turn."""
    receipts: list[ProviderReceipt] = []
    for session in fixture.sessions:
        for arm in OrderingArm:
            turn_indices = (1,) if arm is OrderingArm.DETERMINISTIC else (1, 2, 3)
            for turn_index in turn_indices:
                prewarm = call_openrouter(
                    session=session,
                    arm=arm,
                    turn_index=turn_index,
                    phase="prewarm",
                    api_key=api_key,
                )
                if prewarm.cache_write_tokens <= 0:
                    raise DeterminismEconomicsError("prewarm receipt has zero cache_write_tokens")
                replay = call_openrouter(
                    session=session,
                    arm=arm,
                    turn_index=turn_index,
                    phase="replay",
                    api_key=api_key,
                )
                if replay.cached_tokens <= 0:
                    raise DeterminismEconomicsError("replay receipt has zero cached_tokens")
                receipts.extend((prewarm, replay))
    if len(receipts) != 168:
        raise DeterminismEconomicsError(f"expected 168 preflight receipts, got {len(receipts)}")
    return receipts


def run_measurement(
    *, fixture: SessionFixture, preregistration_commit: str, api_key: str
) -> DeterminismEconomicsArtifact:
    """Run isolated ordering arms and produce a fully validated evidence artifact."""
    preflight = run_preflight(fixture, api_key)
    time.sleep(CACHE_TTL_SECONDS)
    measured: list[ProviderReceipt] = []
    for arm_index, arm in enumerate(OrderingArm):
        for session in fixture.sessions:
            for turn_index in range(1, 4):
                receipt = call_openrouter(
                    session=session,
                    arm=arm,
                    turn_index=turn_index,
                    phase="measurement",
                    api_key=api_key,
                )
                _validate_measured_receipt(receipt, arm, turn_index)
                measured.append(receipt)
                if turn_index < 3:
                    time.sleep(TURN_DELAY_SECONDS)
        if arm_index < len(OrderingArm) - 1:
            time.sleep(CACHE_TTL_SECONDS)
    artifact = DeterminismEconomicsArtifact(
        preregistration_commit=preregistration_commit,
        fixture_sha256=fixture_digest(fixture),
        source_revision=fixture.source_revision,
        model=MODEL,
        sessions=fixture.sessions,
        preflight_receipts=preflight,
        measurement_receipts=measured,
        summary=_summarize(fixture, measured),
        generated_at=utc_now(),
    )
    return artifact


def _validate_measured_receipt(receipt: ProviderReceipt, arm: OrderingArm, turn_index: int) -> None:
    if turn_index == 1 and receipt.cache_write_tokens <= 0:
        raise DeterminismEconomicsError("initial measured request requires cache_write_tokens")
    if arm is OrderingArm.DETERMINISTIC and turn_index > 1 and receipt.cached_tokens <= 0:
        raise DeterminismEconomicsError("deterministic repeat requires cached_tokens")
    if (
        arm is OrderingArm.PERTURBED
        and turn_index > 1
        and (receipt.cache_write_tokens <= 0 or receipt.cached_tokens != 0)
    ):
        raise DeterminismEconomicsError("perturbed transition requires write and zero read")


def validate_provider_receipts(
    receipts: list[ProviderReceipt], fixture: SessionFixture, *, preflight: bool
) -> None:
    """Reject receipts that cannot be replayed from the frozen arm matrix."""
    expected: set[tuple[str, OrderingArm, int, str]] = set()
    expected_hashes: dict[tuple[str, OrderingArm, int], str] = {}
    for session in fixture.sessions:
        for arm in OrderingArm:
            turn_indices = (1,) if preflight and arm is OrderingArm.DETERMINISTIC else (1, 2, 3)
            phases = ("prewarm", "replay") if preflight else ("measurement",)
            for turn_index in turn_indices:
                key = (session.session_id, arm, turn_index)
                expected_hashes[key] = request_payload(session, arm, turn_index)[1]
                expected.update((*key, phase) for phase in phases)
    observed = {
        (receipt.session_id, receipt.arm, receipt.turn_index, receipt.phase) for receipt in receipts
    }
    if observed != expected or len(observed) != len(receipts):
        raise ValueError("receipts do not cover the frozen arm/turn/phase matrix exactly")
    for receipt in receipts:
        key = (receipt.session_id, receipt.arm, receipt.turn_index)
        if receipt.rendered_prefix_sha256 != expected_hashes[key]:
            raise ValueError("receipt rendered-prefix SHA-256 mismatches the frozen fixture")
        if receipt.phase == "prewarm" and receipt.cache_write_tokens <= 0:
            raise ValueError("prewarm receipt lacks cache write")
        if receipt.phase == "replay" and receipt.cached_tokens <= 0:
            raise ValueError("replay receipt lacks cache read")
        if not preflight:
            _validate_measured_receipt(receipt, receipt.arm, receipt.turn_index)


def _summarize(fixture: SessionFixture, receipts: list[ProviderReceipt]) -> dict[str, Any]:
    costs: dict[OrderingArm, dict[str, float]] = {arm: defaultdict(float) for arm in OrderingArm}
    for receipt in receipts:
        costs[receipt.arm][receipt.session_id] += receipt.upstream_inference_prompt_cost
    deterministic = costs[OrderingArm.DETERMINISTIC]
    ann = costs[OrderingArm.ANN_BASELINE]
    repo_ids = [session.session_id for session in fixture.sessions]
    observed = 100 * (sum(ann.values()) - sum(deterministic.values())) / sum(ann.values())
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.choice(repo_ids, size=len(repo_ids), replace=True)
        sampled_ids = [str(item) for item in sampled.tolist()]
        sampled_ann = sum(ann[item] for item in sampled_ids)
        sampled_deterministic = sum(deterministic[item] for item in sampled_ids)
        samples.append(100 * (sampled_ann - sampled_deterministic) / sampled_ann)
    lower, upper = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ann_baseline_input_cost_usd": sum(ann.values()),
        "deterministic_input_cost_usd": sum(deterministic.values()),
        "ann_vs_deterministic_reduction_percent": observed,
        "ann_vs_deterministic_interval_percent": [lower, upper],
        "cost_per_resolved_task": {
            arm.value: sum(costs[arm].values()) / len(repo_ids) for arm in OrderingArm
        },
        "cache_hit_rate": {
            arm.value: sum(receipt.cached_tokens > 0 for receipt in receipts if receipt.arm is arm)
            / sum(receipt.arm is arm for receipt in receipts)
            for arm in OrderingArm
        },
    }


def load_fixture(path: Path) -> SessionFixture:
    """Load a committed S7 fixture or fail with its validation reason."""
    try:
        return SessionFixture.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc:
        raise DeterminismEconomicsError(f"invalid S7 fixture {path}: {exc}") from exc


def load_artifact(path: Path) -> DeterminismEconomicsArtifact:
    """Load validated S7 evidence."""
    try:
        return DeterminismEconomicsArtifact.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc:
        raise DeterminismEconomicsError(f"invalid S7 evidence {path}: {exc}") from exc


def require_openrouter_api_key() -> str:
    """Require explicit OpenRouter credentials; never synthesize a local receipt."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise DeterminismEconomicsError(
            "OPENROUTER_API_KEY is required for S7 provider measurement"
        )
    return api_key
