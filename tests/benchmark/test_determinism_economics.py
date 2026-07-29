from __future__ import annotations

import hashlib

import pytest

from archex.benchmark.determinism_economics import (
    MODEL,
    DeterminismEconomicsError,
    FrozenChunk,
    FrozenSession,
    OrderingArm,
    ProviderReceipt,
    SessionFixture,
    ann_baseline_orders,
    perturbed_orders,
    provider_receipt_from_response,
    request_payload,
    run_preflight,
    validate_provider_receipts,
)


def _session(index: int) -> FrozenSession:
    chunks = [
        FrozenChunk(
            chunk_id=f"chunk-{index}-a",
            file_path="src/a.py",
            start_line=1,
            end_line=2,
            content="def a(): pass",
            dense_score=1.0,
        ),
        FrozenChunk(
            chunk_id=f"chunk-{index}-b",
            file_path="src/b.py",
            start_line=1,
            end_line=2,
            content="def b(): pass",
            dense_score=1.0,
        ),
    ]
    ids = [chunk.chunk_id for chunk in chunks]
    session_id = f"session-{index}"
    return FrozenSession(
        session_id=session_id,
        task_id=session_id,
        repo=f"owner/repo-{index}",
        commit="a" * 40,
        resolved=True,
        turns=("first", "second", "third"),
        chunks=chunks,
        orders={
            OrderingArm.DETERMINISTIC: (
                tuple(ids),
                tuple(ids),
                tuple(ids),
            ),
            OrderingArm.PERTURBED: perturbed_orders(session_id, ids),
            OrderingArm.ANN_BASELINE: ann_baseline_orders(session_id, chunks),
        },
    )


def _fixture() -> SessionFixture:
    return SessionFixture(
        source_revision="b" * 40,
        retrieval_timestamp="2026-07-29T00:00:00Z",
        sessions=[_session(index) for index in range(12)],
    )


def _receipt(
    session: FrozenSession,
    arm: OrderingArm,
    turn_index: int,
    phase: str,
    cached_tokens: int,
    cache_write_tokens: int,
) -> ProviderReceipt:
    _payload, prefix_sha256 = request_payload(session, arm, turn_index)
    return ProviderReceipt(
        arm=arm,
        session_id=session.session_id,
        turn_index=turn_index,
        phase=phase,
        rendered_prefix_sha256=prefix_sha256,
        request_timestamp="2026-07-29T00:00:00Z",
        response_timestamp="2026-07-29T00:00:01Z",
        model=MODEL,
        requested_provider={"only": ["anthropic"], "allow_fallbacks": False},
        provider="Anthropic",
        generation_id="generation",
        usage={},
        total_cost=0.01,
        upstream_inference_prompt_cost=0.001,
        completion_cost=0.009,
        prompt_tokens=100,
        cache_write_tokens=cache_write_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=1,
    )


def test_receipt_parser_accepts_current_openrouter_completion_cost_key() -> None:
    receipt = provider_receipt_from_response(
        raw={
            "model": MODEL,
            "provider": "Anthropic",
            "id": "generation",
            "usage": {
                "cost": 0.01,
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cache_write_tokens": 100, "cached_tokens": 0},
                "cost_details": {
                    "upstream_inference_prompt_cost": 0.001,
                    "upstream_inference_completions_cost": 0.009,
                },
            },
        },
        session=_session(0),
        arm=OrderingArm.DETERMINISTIC,
        turn_index=1,
        phase="measurement",
        prefix_sha256="0" * 64,
        request_timestamp="2026-07-29T00:00:00Z",
    )

    assert receipt.completion_cost == 0.009


def test_request_payload_keeps_question_after_cacheable_context() -> None:
    session = _session(0)

    payload, prefix_sha256 = request_payload(session, OrderingArm.DETERMINISTIC, 1)

    content = payload["messages"][0]["content"]
    assert payload["model"] == MODEL
    assert payload["provider"] == {"only": ["anthropic"], "allow_fallbacks": False}
    assert payload["max_tokens"] == 0
    assert len(content) == 3
    assert content[0]["text"].startswith("You are answering a frozen")
    assert "cache_control" not in content[0]
    assert content[1]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["text"].startswith("<selected-context>")
    assert content[2]["text"] == "first"
    assert (
        prefix_sha256
        == hashlib.sha256((content[0]["text"] + "\n\n" + content[1]["text"]).encode()).hexdigest()
    )


def test_session_rejects_non_immutable_source_revision() -> None:
    payload = _session(0).model_dump()
    payload["commit"] = "v5.4.0"

    with pytest.raises(ValueError, match="immutable 40-character"):
        FrozenSession.model_validate(payload)


def test_perturbed_turns_change_the_prefix() -> None:
    session = _session(0)

    prefixes = [
        request_payload(session, OrderingArm.PERTURBED, turn_index)[1] for turn_index in range(1, 4)
    ]

    assert prefixes[0] != prefixes[1]
    assert prefixes[1] != prefixes[2]


def test_preflight_requires_every_unique_prefix_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    calls: list[tuple[str, str, int]] = []

    def fake_call(
        *, session: FrozenSession, arm: OrderingArm, turn_index: int, phase: str, api_key: str
    ) -> ProviderReceipt:
        del api_key
        calls.append((phase, session.session_id, turn_index))
        return _receipt(
            session,
            arm,
            turn_index,
            phase,
            cached_tokens=100 if phase == "replay" else 0,
            cache_write_tokens=100 if phase == "prewarm" else 0,
        )

    monkeypatch.setattr(
        "archex.benchmark.determinism_economics.call_openrouter",
        fake_call,
    )

    receipts = run_preflight(fixture, "test-key")

    assert len(receipts) == 168
    assert len(calls) == 168
    assert sum(phase == "prewarm" for phase, _, _ in calls) == 84
    assert sum(phase == "replay" for phase, _, _ in calls) == 84


def test_receipt_validation_rejects_prefix_not_in_frozen_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()

    def fake_call(
        *, session: FrozenSession, arm: OrderingArm, turn_index: int, phase: str, api_key: str
    ) -> ProviderReceipt:
        del api_key
        return _receipt(
            session,
            arm,
            turn_index,
            phase,
            cached_tokens=100 if phase == "replay" else 0,
            cache_write_tokens=100 if phase == "prewarm" else 0,
        )

    monkeypatch.setattr("archex.benchmark.determinism_economics.call_openrouter", fake_call)
    receipts = run_preflight(fixture, "test-key")
    receipts[0] = receipts[0].model_copy(update={"rendered_prefix_sha256": "0" * 64})

    with pytest.raises(ValueError, match="rendered-prefix SHA-256"):
        validate_provider_receipts(receipts, fixture, preflight=True)


def test_preflight_fails_on_zero_cache_read(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()

    def fake_call(
        *, session: FrozenSession, arm: OrderingArm, turn_index: int, phase: str, api_key: str
    ) -> ProviderReceipt:
        del api_key
        return _receipt(
            session,
            arm,
            turn_index,
            phase,
            cached_tokens=0,
            cache_write_tokens=100,
        )

    monkeypatch.setattr(
        "archex.benchmark.determinism_economics.call_openrouter",
        fake_call,
    )

    with pytest.raises(DeterminismEconomicsError, match="zero cached_tokens"):
        run_preflight(fixture, "test-key")
