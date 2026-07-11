"""Regression tests for the benchmark-only required-file coverage candidate."""

from __future__ import annotations

import subprocess
from pathlib import Path

from archex.benchmark.coverage_candidate import CoverageSeedDecision as _CoverageSeedDecision
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    _COVERAGE_DIRECT_EVIDENCE_CAP,  # pyright: ignore[reportPrivateUsage]
    _COVERAGE_SEED_CAP,  # pyright: ignore[reportPrivateUsage]
    _apply_coverage_seed_admission,  # pyright: ignore[reportPrivateUsage]
    _coverage_seed_decisions,  # pyright: ignore[reportPrivateUsage]
    default_strategy_registry,
    run_archex_query_coverage_candidate,
)
from archex.models import (
    CodeChunk,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    IndexConfig,
    RankedChunk,
    RepoSource,
)
from archex.scout import chunk_handle


def _index_store(repo_path: Path):
    from archex.api import index_repository
    from archex.models import Config

    source = RepoSource(local_path=str(repo_path), stable_identity="coverage-seed-test@1")
    return index_repository(
        source, config=Config(cache=False), index_config=IndexConfig(vector=False)
    )


def _bundle_from_chunks(query: str, chunks: list[CodeChunk]) -> ContextBundle:
    ranked = [RankedChunk(chunk=chunk, final_score=1.0) for chunk in chunks]
    receipt = ContextReceipt(
        query=query,
        token_budget=ContextReceiptTokenBudget(requested=2048, consumed=0),
        index_revision="coverage-seed-test",
        returned_context=[
            ContextReceiptItem(
                handle=chunk_handle(chunk.id),
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content_hash=f"h-{chunk.id}",
            )
            for chunk in chunks
        ],
        returned_total=len(chunks),
    )
    return ContextBundle(
        query=query,
        chunks=ranked,
        token_count=sum(chunk.token_count for chunk in chunks),
        receipt=receipt,
    )


def _coverage_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "coverage_fixture"
    repo.mkdir()
    (repo / "catalog.py").write_text(
        "class InvoiceMatcher:\n"
        "    def match(self, invoice: str) -> bool:\n"
        "        return bool(invoice)\n"
    )
    (repo / "matching_rules.py").write_text(
        "def apply_rules(invoice: str) -> bool:\n    return invoice.startswith('INV-')\n"
    )
    (repo / "engine.py").write_text(
        "from catalog import InvoiceMatcher\nfrom matching_rules import apply_rules\n"
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@archex.test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "archex-test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


def test_seed_decisions_use_symbol_path_and_lexical_evidence(tmp_path: Path) -> None:
    repo = _coverage_fixture_repo(tmp_path)
    store = _index_store(repo)
    try:
        decisions = _coverage_seed_decisions(
            "Where does InvoiceMatcher apply invoice matching rules?", store, limit=3
        )
    finally:
        store.close()

    by_file = {decision.file: decision for decision in decisions}
    assert "catalog.py" in by_file
    assert "matching_rules.py" in by_file
    assert any(reason.startswith("symbol:") for reason in by_file["catalog.py"].evidence)
    assert any(reason.startswith("path:") for reason in by_file["matching_rules.py"].evidence)
    assert any(
        reason.startswith("lexical:") for decision in decisions for reason in decision.evidence
    )


def test_seed_admission_records_evidence_and_budget_cut(tmp_path: Path) -> None:
    repo = _coverage_fixture_repo(tmp_path)
    store = _index_store(repo)
    try:
        chunks_by_file = {
            path: store.get_chunks_for_files([path])
            for path in ("engine.py", "catalog.py", "matching_rules.py")
        }
        bundle = _bundle_from_chunks("invoice matching", chunks_by_file["engine.py"])
        decision = _CoverageSeedDecision(
            file="catalog.py",
            score=8,
            evidence=("symbol:invoicematcher",),
        )
        admitted = _apply_coverage_seed_admission(
            bundle,
            store,
            [decision],
            token_budget=2048,
        )
        cut = _apply_coverage_seed_admission(
            bundle,
            store,
            [decision],
            token_budget=bundle.token_count,
        )
    finally:
        store.close()

    assert "catalog.py" in {chunk.chunk.file_path for chunk in admitted.bundle.chunks}
    assert admitted.bundle.receipt is not None
    item = next(
        item for item in admitted.bundle.receipt.returned_context if item.file_path == "catalog.py"
    )
    assert item.reason_codes == ["coverage_seed:symbol:invoicematcher"]
    assert admitted.admitted == [decision]
    assert cut.admitted == []
    assert cut.budget_cuts == [decision]


def test_candidate_is_available_but_not_default() -> None:
    from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES

    assert (
        default_strategy_registry.get(Strategy.ARCHEX_QUERY_COVERAGE_CANDIDATE)
        is run_archex_query_coverage_candidate
    )
    assert Strategy.ARCHEX_QUERY_COVERAGE_CANDIDATE in AVAILABLE_STRATEGIES
    assert Strategy.ARCHEX_QUERY_COVERAGE_CANDIDATE not in DEFAULT_STRATEGIES


def test_candidate_seed_frontier_is_bounded_below_evidence_pool() -> None:
    assert 0 < _COVERAGE_SEED_CAP < _COVERAGE_DIRECT_EVIDENCE_CAP


def test_candidate_returned_files_do_not_depend_on_expected_files(
    python_simple_repo: Path,
) -> None:
    first = run_archex_query_coverage_candidate(
        BenchmarkTask(
            task_id="generic_coverage_candidate",
            repo="test/python_simple",
            commit="abc",
            question="Where is the AuthService class?",
            expected_files=["models.py"],
            token_budget=1024,
        ),
        python_simple_repo,
    )
    second = run_archex_query_coverage_candidate(
        BenchmarkTask(
            task_id="generic_coverage_candidate",
            repo="test/python_simple",
            commit="abc",
            question="Where is the AuthService class?",
            expected_files=["utils.py"],
            token_budget=1024,
        ),
        python_simple_repo,
    )

    assert first.result_files == second.result_files
    assert first.provenance == second.provenance
