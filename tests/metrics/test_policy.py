from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from archex.metrics.math import BASELINE_RETURNED_FULL_FILES, compute_token_savings
from archex.metrics.policy import (
    METRICS_ENV,
    TRACE_ENV,
    resolve_metrics_policy,
    set_metrics_enabled,
    set_trace_enabled,
)
from archex.metrics.registry import RepoRegistry
from archex.metrics.storage import MetricsStore


def test_policy_defaults_to_counters_disabled_trace_disabled(tmp_path: Path) -> None:
    policy = resolve_metrics_policy(db_path=tmp_path / "usage.sqlite", env={})

    assert policy.metrics_enabled is False
    assert policy.trace_enabled is False
    assert policy.raw_event_retention_days == 90
    assert policy.trace_retention_days == 14
    assert policy.hosted_upload_enabled is False


def test_policy_env_controls_metrics_and_trace(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    set_trace_enabled(True, db_path=db_path)
    set_metrics_enabled(True, db_path=db_path)

    disabled = resolve_metrics_policy(
        db_path=db_path,
        env={METRICS_ENV: "off", TRACE_ENV: "off"},
    )
    traced = resolve_metrics_policy(db_path=db_path, env={TRACE_ENV: "on"})
    enabled = resolve_metrics_policy(db_path=db_path, env={METRICS_ENV: "on"})

    assert disabled.metrics_enabled is False
    assert disabled.trace_enabled is False
    assert traced.metrics_enabled is True
    assert traced.trace_enabled is True
    assert enabled.metrics_enabled is True


@pytest.mark.parametrize(
    ("returned", "raw", "repo_total", "saved", "pct", "whole_repo_avoided"),
    [
        (40, 100, 1000, 60, 60.0, 960),
        (120, 100, 1000, 0, 0.0, 880),
        (0, 0, None, 0, 0.0, None),
        (1200, 100, 1000, 0, 0.0, 0),
    ],
)
def test_token_savings_math_matches_design(
    returned: int,
    raw: int,
    repo_total: int | None,
    saved: int,
    pct: float,
    whole_repo_avoided: int | None,
) -> None:
    result = compute_token_savings(
        tokens_returned=returned,
        tokens_raw_equivalent=raw,
        whole_repo_tokens=repo_total,
    )

    assert result.tokens_returned == returned
    assert result.tokens_raw_equivalent == raw
    assert result.tokens_saved == saved
    assert result.savings_pct == pct
    assert result.whole_repo_tokens == repo_total
    assert result.whole_repo_tokens_avoided == whole_repo_avoided
    assert result.baseline_type == BASELINE_RETURNED_FULL_FILES


def test_token_savings_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="tokens_returned must be non-negative"):
        compute_token_savings(tokens_returned=-1, tokens_raw_equivalent=1)


def test_repo_registry_uses_stable_random_local_uuid(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    repo_root = tmp_path / "workspace" / "secret-repo-name"
    repo_root.mkdir(parents=True)
    registry = RepoRegistry(db_path)

    first = registry.get_or_create(repo_root)
    second = registry.get_or_create(repo_root)

    assert first == second
    assert UUID(first.repo_id).version == 4
    assert first.repo_root == repo_root.resolve()
    assert first.display_name == "secret-repo-name"


def test_repo_registry_keeps_paths_only_in_repos_mapping(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    repo_root = tmp_path / "org-name" / "repo-name"
    repo_root.mkdir(parents=True)
    registered = RepoRegistry(db_path).get_or_create(repo_root)

    with MetricsStore(db_path).connect() as conn, conn:
        conn.execute(
            """
            INSERT INTO usage_events(
                event_id, occurred_at, repo_id, surface, tool_name, category,
                tokens_returned, tokens_raw_equivalent, tokens_saved, savings_pct,
                whole_repo_tokens, whole_repo_tokens_avoided, baseline_type, file_count,
                freshness, index_revision, trace_id
            ) VALUES (
                'event-1', '2026-06-16T00:00:00+00:00', ?, 'cli', 'query',
                'context_retrieval', 10, 100, 90, 90.0, 1000, 990,
                'returned_full_files', 2, 'clean', 'rev', NULL
            )
            """,
            (registered.repo_id,),
        )
        event = conn.execute("SELECT * FROM usage_events WHERE event_id = 'event-1'").fetchone()
        repo = conn.execute(
            "SELECT repo_root, display_name FROM repos WHERE repo_id = ?",
            (registered.repo_id,),
        ).fetchone()

    event_values = {str(value) for value in event if value is not None}
    assert str(repo_root.resolve()) in str(repo["repo_root"])
    assert repo["display_name"] == "repo-name"
    assert str(repo_root.resolve()) not in event_values
    assert "org-name" not in event_values
    assert "repo-name" not in event_values


def test_usage_event_rows_have_no_sensitive_default_columns(tmp_path: Path) -> None:
    forbidden = {
        "query_text",
        "file_path",
        "file_paths",
        "path_hash",
        "symbols",
        "handles",
        "source_snippet",
        "rendered_output",
        "prompt_body",
        "remote_url",
        "org_name",
        "repo_name",
    }
    with MetricsStore(tmp_path / "usage.sqlite").connect() as conn:
        event_columns = {row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")}

    assert event_columns.isdisjoint(forbidden)
