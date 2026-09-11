"""Behavioral tests for the benchmark-only scope-aware strategy and sidecar."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from archex.api import query
from archex.benchmark.models import BenchmarkRetrievalOptions, BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.scope_aware_receipt import (
    MultiScopeReceipt,
    SingleScopeReceipt,
    canonical_context_payload,
    canonical_receipt_json,
)
from archex.benchmark.scope_aware_strategy import (
    execute_scope_aware_query,
    frozen_scope_aware_config,
    frozen_scope_aware_index_config,
    scope_aware_repo_source,
)
from archex.benchmark.strategies import (
    benchmark_repo_source,
    default_strategy_registry,
    reset_benchmark_retrieval_options,
    run_scope_aware_candidate,
    set_benchmark_retrieval_options,
)
from archex.models import ContextBundle, RepoSource


def _write_python_scope(root: Path, scope: str, *, term: str) -> None:
    scope_root = root / scope
    scope_root.mkdir(parents=True)
    if not (root / ".git").exists():
        subprocess.run(
            ["git", "init", "--quiet", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
    (scope_root / "package.json").write_text("{}\n", encoding="utf-8")
    repeated = " ".join([term] * 80)
    (scope_root / "service.py").write_text(
        f"def retrieve_scope():\n    payload = {repeated!r}\n    return payload\n",
        encoding="utf-8",
    )


def test_single_scope_delegates_to_byte_identical_archex_query(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_python_scope(repo, "package", term="needle")
    source = RepoSource(local_path=str(repo), stable_identity="single-scope-fixture@1")
    config = frozen_scope_aware_config(
        languages=["python"],
        cache_dir=str(tmp_path / "cache"),
    )
    index_config = frozen_scope_aware_index_config()

    outcome = execute_scope_aware_query(
        source,
        "needle",
        repo_root=repo,
        task_id="fixture-single",
        repository_id="fixture-single",
        token_budget=4000,
        config=config,
        index_config=index_config,
    )
    control = query(
        source,
        "needle",
        token_budget=4000,
        config=config,
        index_config=index_config,
        explicit_token_budget=True,
        refresh=False,
    )

    assert isinstance(outcome.receipt, SingleScopeReceipt)
    assert canonical_context_payload(outcome.bundle) == canonical_context_payload(control)
    assert (
        outcome.receipt.payload_sha256
        == hashlib.sha256(canonical_context_payload(control)).hexdigest()
    )


def test_multi_scope_smoke_emits_reconciling_sidecar_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_python_scope(repo, "packages/a", term="scopealpha")
    _write_python_scope(repo, "packages/b", term="unrelated")
    source = RepoSource(local_path=str(repo), stable_identity="multi-scope-fixture@1")

    outcome = execute_scope_aware_query(
        source,
        "scopealpha",
        repo_root=repo,
        task_id="fixture-multi",
        repository_id="fixture-multi",
        token_budget=4000,
        config=frozen_scope_aware_config(
            languages=["python"],
            cache_dir=str(tmp_path / "cache"),
        ),
        index_config=frozen_scope_aware_index_config(),
    )

    receipt = outcome.receipt
    assert isinstance(receipt, MultiScopeReceipt)
    assert [scope.scope for scope in receipt.searched_scopes] == ["packages/a", "packages/b"]
    assert receipt.included_scope_count == 1
    assert receipt.rejected_scope_count == 1
    assert receipt.final_candidate_count == sum(
        scope.post_dedup_contribution_count
        for scope in receipt.searched_scopes
        if scope.decision == "included"
    )
    assert "scope_receipt" not in outcome.bundle.model_dump(mode="json")
    assert json.loads(canonical_receipt_json(receipt))["mode"] == "scope_aware_ranking"


def test_multi_scope_no_match_rejects_every_scope_with_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_python_scope(repo, "packages/a", term="scopealpha")
    _write_python_scope(repo, "packages/b", term="scopebeta")

    outcome = execute_scope_aware_query(
        RepoSource(local_path=str(repo), stable_identity="no-match-fixture@1"),
        "term-that-is-not-indexed",
        repo_root=repo,
        task_id="fixture-no-match",
        repository_id="fixture-no-match",
        token_budget=4000,
        config=frozen_scope_aware_config(
            languages=["python"],
            cache_dir=str(tmp_path / "cache"),
        ),
        index_config=frozen_scope_aware_index_config(),
    )

    receipt = outcome.receipt
    assert isinstance(receipt, MultiScopeReceipt)
    assert receipt.repository_global_max_raw_score == 0.0
    assert receipt.included_scope_count == 0
    assert receipt.final_candidate_count == 0
    assert {scope.reason for scope in receipt.searched_scopes} == {"non_positive_global_max"}


def test_multi_scope_receipt_fails_closed_on_count_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_python_scope(repo, "packages/a", term="scopealpha")
    _write_python_scope(repo, "packages/b", term="unrelated")
    outcome = execute_scope_aware_query(
        RepoSource(local_path=str(repo), stable_identity="receipt-drift-fixture@1"),
        "scopealpha",
        repo_root=repo,
        task_id="fixture-drift",
        repository_id="fixture-drift",
        token_budget=4000,
        config=frozen_scope_aware_config(
            languages=["python"],
            cache_dir=str(tmp_path / "cache"),
        ),
        index_config=frozen_scope_aware_index_config(),
    )
    payload = outcome.receipt.model_dump(mode="json")
    payload["final_candidate_count"] += 1

    with pytest.raises(ValidationError, match="final candidate count does not reconcile"):
        MultiScopeReceipt.model_validate(payload)


def test_cache_identity_is_distinct_and_independent_of_cli_options(tmp_path: Path) -> None:
    task = BenchmarkTask(
        task_id="cache-identity",
        repo="fixture/repo",
        commit="fixture",
        question="needle",
        expected_files=["package/service.py"],
        languages=["python"],
    )
    config = frozen_scope_aware_config(
        languages=task.languages,
        cache_dir=str(tmp_path / "cache"),
    )
    index_config = frozen_scope_aware_index_config()
    before = scope_aware_repo_source(
        task,
        tmp_path,
        config=config,
        index_config=index_config,
    )
    token = set_benchmark_retrieval_options(
        BenchmarkRetrievalOptions(
            chunker="cast",
            allow_remote_code=True,
            module_prefilter=True,
        )
    )
    try:
        after = scope_aware_repo_source(
            task,
            tmp_path,
            config=config,
            index_config=index_config,
        )
        control = benchmark_repo_source(task, tmp_path, strategy=Strategy.ARCHEX_QUERY)
    finally:
        reset_benchmark_retrieval_options(token)

    assert before.stable_identity == after.stable_identity
    assert before.stable_identity != control.stable_identity
    assert "#scope-aware-candidate:" in (before.stable_identity or "")


def test_strategy_is_registered_but_never_default() -> None:
    assert Strategy.SCOPE_AWARE_CANDIDATE in AVAILABLE_STRATEGIES
    assert Strategy.SCOPE_AWARE_CANDIDATE not in DEFAULT_STRATEGIES
    assert (
        default_strategy_registry.get(Strategy.SCOPE_AWARE_CANDIDATE) is run_scope_aware_candidate
    )
    assert "scope_receipt" not in ContextBundle.model_fields
