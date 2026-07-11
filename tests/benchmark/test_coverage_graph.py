"""Regression tests for evidence-backed coverage graph-neighbor admission."""

from __future__ import annotations

from pathlib import Path

from archex.api import index_repository
from archex.benchmark.coverage_candidate import (
    CoverageSeedDecision,
    _select_evidence_chunk,  # pyright: ignore[reportPrivateUsage]
    apply_coverage_seed_admission,
    coverage_neighbor_decisions,
    coverage_seed_decisions,
)
from archex.benchmark.graph_multihop import GraphEdge
from archex.models import (
    CodeChunk,
    Config,
    ContextBundle,
    ContextReceipt,
    ContextReceiptTokenBudget,
    IndexConfig,
    RepoSource,
)
from archex.serve.context import generic_query_terms


def test_neighbor_admission_requires_confident_graph_evidence() -> None:
    decisions = coverage_neighbor_decisions(
        [
            GraphEdge("entry.py", "validation.py", 0.9),
            GraphEdge("entry.py", "boundary.py", 0.5),
            GraphEdge("entry.py", "unrelated.py", 0.2),
        ],
        seed_files={"entry.py"},
        existing_files=set(),
        direct_decisions=[
            CoverageSeedDecision(
                file="validation.py",
                score=8,
                evidence=("symbol:validateinvoice",),
            )
        ],
        limit=3,
    )

    assert [decision.file for decision in decisions] == ["validation.py", "boundary.py"]
    decision = decisions[0]
    assert decision.kind == "neighbor"
    assert decision.via == "entry.py"
    assert "graph_import:entry.py" in decision.evidence
    assert "symbol:validateinvoice" in decision.evidence


def test_neighbor_admission_is_bounded_and_deterministic() -> None:
    edges = [
        GraphEdge("entry.py", "b.py", 0.9),
        GraphEdge("entry.py", "a.py", 0.9),
        GraphEdge("entry.py", "c.py", 0.9),
    ]

    decisions = coverage_neighbor_decisions(
        edges,
        seed_files={"entry.py"},
        existing_files={"b.py"},
        direct_decisions=[
            CoverageSeedDecision(file="c.py", score=1_000, evidence=("symbol:entry",))
        ],
        limit=2,
    )

    assert [decision.file for decision in decisions] == ["a.py", "c.py"]


def test_generic_candidate_terms_do_not_apply_semantic_synonyms() -> None:
    terms = set(generic_query_terms("How does middleware process a request?"))

    assert "middleware" in terms
    assert {"asgi", "wsgi", "hook", "interceptor"}.isdisjoint(terms)


def test_admission_selects_smallest_equally_evidenced_chunk() -> None:
    decision = CoverageSeedDecision(
        file="catalog.py",
        score=1,
        evidence=("symbol:invoicematcher",),
    )
    chunks = [
        CodeChunk(
            id="catalog.py:large:1",
            content="class InvoiceMatcher:\n    pass\n",
            file_path="catalog.py",
            start_line=1,
            end_line=2,
            language="python",
            token_count=100,
            symbol_name="InvoiceMatcher",
        ),
        CodeChunk(
            id="catalog.py:small:10",
            content="class InvoiceMatcher:\n    pass\n",
            file_path="catalog.py",
            start_line=10,
            end_line=11,
            language="python",
            token_count=10,
            symbol_name="InvoiceMatcher",
        ),
    ]

    assert _select_evidence_chunk(chunks, decision).id == "catalog.py:small:10"


def test_seed_decisions_exclude_benchmark_artifacts(
    python_simple_repo: Path,
) -> None:
    benchmark_file = python_simple_repo / "benchmarks" / "task.py"
    benchmark_file.parent.mkdir()
    benchmark_file.write_text("class BenchmarkRegistry:\n    pass\n")
    store = index_repository(
        RepoSource(
            local_path=str(python_simple_repo),
            stable_identity="coverage-benchmark-artifact-test@1",
        ),
        config=Config(cache=False),
        index_config=IndexConfig(vector=False),
    )
    try:
        decisions = coverage_seed_decisions("Where is the BenchmarkRegistry class?", store, limit=3)
    finally:
        store.close()

    assert "benchmarks/task.py" not in {decision.file for decision in decisions}


def test_neighbor_admission_records_receipt_provenance(
    python_simple_repo: Path,
) -> None:
    store = index_repository(
        RepoSource(
            local_path=str(python_simple_repo),
            stable_identity="coverage-graph-receipt-test@1",
        ),
        config=Config(cache=False),
        index_config=IndexConfig(vector=False),
    )
    try:
        importer = coverage_neighbor_decisions(
            [GraphEdge("main.py", "models.py", 0.9)],
            seed_files={"models.py"},
            existing_files=set(),
            direct_decisions=[],
            limit=1,
        )[0]
        assert importer.file == "main.py"
        assert importer.via == "models.py"
        assert importer.edge_source == "main.py"
        assert importer.edge_target == "models.py"
        admitted = apply_coverage_seed_admission(
            ContextBundle(
                query="Where is AuthService?",
                chunks=[],
                token_count=0,
                receipt=ContextReceipt(
                    query="Where is AuthService?",
                    token_budget=ContextReceiptTokenBudget(requested=1024, consumed=0),
                    index_revision="coverage-graph-receipt-test",
                ),
            ),
            store,
            [
                CoverageSeedDecision(
                    file="models.py",
                    score=100,
                    evidence=("graph_import:entry.py",),
                    kind="neighbor",
                    via="entry.py",
                    edge_source="entry.py",
                    edge_target="models.py",
                ),
                importer,
            ],
            token_budget=1024,
        )
    finally:
        store.close()

    assert admitted.bundle.receipt is not None
    edges = {(edge.source, edge.target) for edge in admitted.bundle.receipt.included_edges}
    edge_confidences = {
        (edge.source, edge.target): edge.confidence_score
        for edge in admitted.bundle.receipt.included_edges
    }
    assert edge_confidences[("entry.py", "models.py")] == 1.0
    assert edge_confidences[("main.py", "models.py")] == 0.9
    assert edges == {("entry.py", "models.py"), ("main.py", "models.py")}
    returned = {
        item.file_path: item.reason_codes for item in admitted.bundle.receipt.returned_context
    }
    assert returned["models.py"] == ["coverage_neighbor:graph_import:entry.py"]
    assert returned["main.py"] == ["coverage_neighbor:graph_importer:models.py"]


def test_neighbor_score_rounds_fractional_confidence_and_propagates_it() -> None:
    decisions = coverage_neighbor_decisions(
        [GraphEdge("entry.py", "target.py", 0.876)],
        seed_files={"entry.py"},
        existing_files=set(),
        direct_decisions=[],
        limit=1,
    )

    decision = decisions[0]
    assert decision.score == round(87.6) + 20
    assert decision.score != int(87.6) + 20
    assert decision.edge_confidence == 0.876
