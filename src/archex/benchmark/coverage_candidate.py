"""Generic deterministic seed admission for the M0.2 benchmark candidate."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.models import (
    ContextBundle,
    ContextReceiptEdge,
    ContextReceiptItem,
    ContextSkippedCandidate,
    ContextSkippedReason,
    EdgeConfidence,
    EdgeKind,
    RankedChunk,
)
from archex.reporting import count_tokens
from archex.scout import chunk_handle
from archex.serve.context import generic_query_terms

if TYPE_CHECKING:
    from archex.benchmark.graph_multihop import GraphEdge
    from archex.index.store import IndexStore
    from archex.models import CodeChunk


@dataclass(frozen=True)
class CoverageSeedDecision:
    """One file admitted by generic query/index evidence."""

    file: str
    score: int
    evidence: tuple[str, ...]
    kind: str = "seed"
    via: str | None = None
    edge_source: str | None = None
    edge_target: str | None = None
    edge_confidence: float | None = None


@dataclass(frozen=True)
class CoverageSeedAdmission:
    """Bounded admission result with receipt-aligned budget cuts."""

    bundle: ContextBundle
    admitted: list[CoverageSeedDecision]
    budget_cuts: list[CoverageSeedDecision]


_GENERIC_QUERY_TERMS = frozenset(
    {
        "account",
        "change",
        "characters",
        "during",
        "inputs",
        "issue",
        "local",
        "needs",
        "repo",
        "should",
        "too",
        "whether",
    }
)


_NEIGHBOR_MIN_CONFIDENCE = 0.5
_EVIDENCE_HIT_CAP = 64


def coverage_seed_decisions(
    question: str, store: IndexStore, *, limit: int
) -> list[CoverageSeedDecision]:
    """Rank up to *limit* non-test files from path, symbol, and lexical evidence."""
    if limit < 1:
        return []

    terms = tuple(
        term for term in generic_query_terms(question) if term not in _GENERIC_QUERY_TERMS
    )[:24]
    evidence_by_file: dict[str, set[str]] = defaultdict(set)
    file_metadata = store.get_file_metadata()
    for metadata in file_metadata:
        file_path = str(metadata["file_path"])
        if _is_test_path(file_path):
            continue
        path_parts = _path_parts(file_path)
        for term in terms:
            if any(part.startswith(term) or term.startswith(part) for part in path_parts):
                evidence_by_file[file_path].add(f"path:{term}")

    from archex.index.bm25 import BM25Index

    bm25 = BM25Index(store)
    for identifier in _explicit_identifiers(question):
        for chunk in store.search_symbols(identifier, limit=_EVIDENCE_HIT_CAP):
            if not _is_test_path(chunk.file_path):
                evidence_by_file[chunk.file_path].add(f"identifier:{identifier.lower()}")
    for term in terms:
        for chunk in store.search_symbols(term, limit=_EVIDENCE_HIT_CAP):
            if not _is_test_path(chunk.file_path):
                evidence_by_file[chunk.file_path].add(f"symbol:{term}")
        for chunk, _score in bm25.search(term, top_k=_EVIDENCE_HIT_CAP):
            if not _is_test_path(chunk.file_path):
                evidence_by_file[chunk.file_path].add(f"lexical:{term}")

    candidate_files = sorted(evidence_by_file)
    if not candidate_files:
        return []
    for chunk in store.get_chunks_for_files(candidate_files):
        if _is_test_path(chunk.file_path) or not chunk.symbol_name:
            continue
        symbol_terms = set(generic_query_terms(chunk.symbol_name))
        for term in set(terms) & symbol_terms:
            evidence_by_file[chunk.file_path].add(f"symbol:{term}")

    evidence_frequency = Counter(
        reason for evidence in evidence_by_file.values() for reason in evidence
    )
    decisions = [
        CoverageSeedDecision(
            file=file_path,
            score=_evidence_score(evidence, evidence_frequency),
            evidence=tuple(sorted(evidence)),
        )
        for file_path, evidence in evidence_by_file.items()
    ]
    return sorted(decisions, key=lambda decision: (-decision.score, decision.file))[:limit]


def coverage_neighbor_decisions(
    edges: list[GraphEdge],
    *,
    seed_files: set[str],
    existing_files: set[str],
    direct_decisions: list[CoverageSeedDecision],
    limit: int,
    require_direct_evidence: bool = False,
) -> list[CoverageSeedDecision]:
    """Rank bounded graph neighbors, optionally requiring direct query evidence."""
    if limit < 1:
        return []

    direct_by_file = {decision.file: decision for decision in direct_decisions}
    candidates: dict[str, CoverageSeedDecision] = {}
    for edge in edges:
        if edge.confidence < _NEIGHBOR_MIN_CONFIDENCE:
            continue
        routes: list[tuple[str, str, str, int]] = []
        if edge.source in seed_files:
            routes.append((edge.target, edge.source, "graph_import", 20))
        if edge.target in seed_files:
            routes.append((edge.source, edge.target, "graph_importer", 10))
        for file_path, via, relation, direction_bonus in routes:
            if file_path in seed_files or file_path in existing_files or _is_test_path(file_path):
                continue
            direct = direct_by_file.get(file_path)
            if require_direct_evidence and direct is None:
                continue
            direct_evidence = direct.evidence if direct is not None else ()
            decision = CoverageSeedDecision(
                file=file_path,
                score=round(edge.confidence * 100) + direction_bonus,
                evidence=(f"{relation}:{via}", *direct_evidence),
                kind="neighbor",
                via=via,
                edge_source=via if relation == "graph_import" else file_path,
                edge_target=file_path if relation == "graph_import" else via,
                edge_confidence=edge.confidence,
            )
            existing = candidates.get(file_path)
            if existing is None or (decision.score, decision.via or "") > (
                existing.score,
                existing.via or "",
            ):
                candidates[file_path] = decision
    ordered = sorted(candidates.values(), key=lambda decision: (-decision.score, decision.file))
    return ordered[:limit]


def apply_coverage_seed_admission(
    bundle: ContextBundle,
    store: IndexStore,
    decisions: list[CoverageSeedDecision],
    *,
    token_budget: int,
) -> CoverageSeedAdmission:
    """Append one evidence-aligned chunk per admitted file without exceeding budget."""
    present_files = {ranked.chunk.file_path for ranked in bundle.chunks}
    appended: list[RankedChunk] = []
    receipt_items: list[ContextReceiptItem] = []
    admitted: list[CoverageSeedDecision] = []
    budget_cuts: list[CoverageSeedDecision] = []
    added_tokens = 0

    for decision in decisions:
        if decision.file in present_files:
            continue
        chunks = store.get_chunks_for_files([decision.file])
        if not chunks:
            continue
        chunk = _select_evidence_chunk(chunks, decision)
        tokens = chunk.token_count or count_tokens(chunk.content)
        if bundle.token_count + added_tokens + tokens > token_budget:
            budget_cuts.append(decision)
            continue
        present_files.add(decision.file)
        admitted.append(decision)
        added_tokens += tokens
        appended.append(RankedChunk(chunk=chunk, final_score=float(decision.score)))
        receipt_items.append(
            ContextReceiptItem(
                handle=chunk_handle(chunk.id),
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content_hash=_chunk_content_hash(chunk),
                reason_codes=[f"coverage_{decision.kind}:{reason}" for reason in decision.evidence],
            )
        )

    neighbor_edges = [
        ContextReceiptEdge(
            source=decision.edge_source,
            target=decision.edge_target,
            kind=EdgeKind.IMPORTS,
            confidence=EdgeConfidence.EXTRACTED,
            confidence_score=decision.edge_confidence
            if decision.edge_confidence is not None
            else 1.0,
            evidence=[f"coverage graph evidence: {','.join(decision.evidence)}"],
        )
        for decision in admitted
        if (
            decision.kind == "neighbor"
            and decision.edge_source is not None
            and decision.edge_target is not None
        )
    ]

    receipt = bundle.receipt
    if receipt is not None:
        skipped = list(receipt.skipped_candidates)
        skipped.extend(
            ContextSkippedCandidate(
                file_path=decision.file,
                reason=ContextSkippedReason.OVER_BUDGET,
                score=float(decision.score),
                detail=f"coverage {decision.kind} evidence: {','.join(decision.evidence)}",
            )
            for decision in budget_cuts
        )
        returned_context = [*receipt.returned_context, *receipt_items]
        included_edges = [*receipt.included_edges, *neighbor_edges]
        receipt = receipt.model_copy(
            update={
                "returned_context": returned_context,
                "returned_total": len(returned_context),
                "skipped_candidates": skipped,
                "included_edges": included_edges,
                "included_edges_total": len(included_edges),
                "skipped_total": len(skipped),
                "token_budget": receipt.token_budget.model_copy(
                    update={"consumed": bundle.token_count + added_tokens}
                ),
            }
        )

    return CoverageSeedAdmission(
        bundle=bundle.model_copy(
            update={
                "chunks": [*bundle.chunks, *appended],
                "token_count": bundle.token_count + added_tokens,
                "receipt": receipt,
            }
        ),
        admitted=admitted,
        budget_cuts=budget_cuts,
    )


def _explicit_identifiers(question: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", question)
        if "_" in token or any(character.isupper() for character in token[1:])
    )


def _path_parts(file_path: str) -> set[str]:
    return {part for part in re.split(r"[/_.-]+", file_path.lower()) if len(part) >= 3}


def _is_test_path(file_path: str) -> bool:
    return (
        file_path.startswith(("tests/", "benchmarks/"))
        or "/tests/" in file_path
        or "/benchmarks/" in file_path
    )


def _evidence_score(evidence: set[str], frequency: Counter[str]) -> int:
    weights = {"identifier": 24, "symbol": 4, "path": 2, "lexical": 1}
    return sum(weights[reason.split(":", 1)[0]] * 1000 // frequency[reason] for reason in evidence)


def _select_evidence_chunk(chunks: list[CodeChunk], decision: CoverageSeedDecision) -> CodeChunk:
    evidence_terms = {
        reason.split(":", 1)[1]
        for reason in decision.evidence
        if reason.startswith(("symbol:", "lexical:"))
    }

    def priority(chunk: CodeChunk) -> tuple[int, int, int]:
        symbol_terms: set[str] = (
            set(generic_query_terms(chunk.symbol_name)) if chunk.symbol_name else set()
        )
        return (
            -len(symbol_terms & evidence_terms),
            chunk.token_count or count_tokens(chunk.content),
            chunk.start_line,
        )

    return min(chunks, key=priority)


def _chunk_content_hash(chunk: CodeChunk) -> str:
    from archex.receipt import chunk_content_hash

    return chunk_content_hash(chunk)
