"""Deterministic context receipt construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

from archex.models import (
    CodeChunk,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextOmittedEdgeReason,
    ContextReceipt,
    ContextReceiptEdge,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
    ContextSkippedCandidate,
    ContextSkippedReason,
    Edge,
    RankedChunk,
)
from archex.scout import ScoutResult, chunk_handle, file_handle, parse_scout_handle

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph
    from archex.index.store import IndexStore
    from archex.integrations.docs.models import DocProviderReceipt
    from archex.integrations.history.eligibility import HistoryEligibilityDecision
    from archex.integrations.history.models import (
        ChangeCard,
        HistoryProviderReceipt,
        TemporalCouplingObservation,
    )
    from archex.integrations.runtime.models import RuntimeProviderReceipt
    from archex.integrations.semantic.models import SemanticProviderReceipt
    from archex.models import ContextBundle

_MAX_RECEIPT_SKIPPED = 20
_MAX_RECEIPT_OMITTED_EDGES = 20
_MAX_RECEIPT_INCLUDED_EDGES = 40


def region_content_hash(file_path: str, start_line: int, end_line: int, content: str) -> str:
    """Stable hash of a file region, used for receipt rows and compression provenance."""
    payload = f"{file_path}\0{start_line}\0{end_line}\0{content}"
    return hashlib.sha256(payload.encode()).hexdigest()


def chunk_content_hash(chunk: CodeChunk) -> str:
    return region_content_hash(chunk.file_path, chunk.start_line, chunk.end_line, chunk.content)


def index_revision_from_store(store: IndexStore) -> str:
    states = store.get_file_states()
    if states:
        payload = json.dumps(
            [
                (path, str(state.get("sha256", "")))
                for path, state in sorted(states.items(), key=lambda item: item[0])
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()
    metadata_parts = [
        store.get_metadata("working_tree_signature") or "",
        store.get_metadata("commit_hash") or "",
        store.get_metadata("chunker") or "",
        store.get_metadata("chunker_revision") or "",
        store.get_metadata("chunk_count") or "",
    ]
    payload = "\0".join(metadata_parts)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_context_receipt(
    bundle: ContextBundle,
    *,
    index_revision: str,
    freshness: ContextFreshness,
    freshness_checked_at: str | None = None,
    index_fresh_at: str | None = None,
    watch_fresh_at: str | None = None,
    included_edges: Iterable[ContextReceiptEdge] = (),
    omitted_edges: Iterable[ContextReceiptEdge] = (),
    skipped_candidates: Iterable[ContextSkippedCandidate] = (),
    semantic_providers: Iterable[SemanticProviderReceipt] = (),
    runtime_providers: Iterable[RuntimeProviderReceipt] = (),
    history_providers: Iterable[HistoryProviderReceipt] = (),
    history_change_cards: Iterable[ChangeCard] = (),
    history_coupling_observations: Iterable[TemporalCouplingObservation] = (),
    documentation_providers: Iterable[DocProviderReceipt] = (),
) -> ContextReceipt:
    returned = _returned_context(bundle)
    skipped_all = list(skipped_candidates)
    omitted_all = list(omitted_edges)
    included_all = list(included_edges)
    skipped = sorted(skipped_all, key=_skipped_sort_key)[:_MAX_RECEIPT_SKIPPED]
    omitted = sorted(
        omitted_all,
        key=lambda item: (
            item.reason.value if item.reason else "",
            item.source,
            item.target,
            item.kind.value,
        ),
    )[:_MAX_RECEIPT_OMITTED_EDGES]
    included = sorted(
        included_all,
        key=lambda item: (item.source, item.target, item.kind.value),
    )[:_MAX_RECEIPT_INCLUDED_EDGES]
    history_providers_resolved = list(history_providers)
    history_change_cards_resolved = list(history_change_cards)
    history_coupling_resolved = list(history_coupling_observations)
    history_eligibility: HistoryEligibilityDecision | None = None
    if history_providers_resolved:
        from archex.integrations.history.eligibility import evaluate_history_eligibility
        from archex.integrations.history.models import HistoryEvidenceProviderName

        git_log_receipt = next(
            (
                r
                for r in history_providers_resolved
                if r.provider == HistoryEvidenceProviderName.GIT_LOG
            ),
            None,
        )
        candidate_file_paths = {item.file_path for item in returned}
        history_eligibility = evaluate_history_eligibility(
            history_change_cards_resolved,
            history_coupling_resolved,
            candidate_file_paths,
            git_log_receipt=git_log_receipt,
            window_commit_count=git_log_receipt.window_commit_count if git_log_receipt else 0,
        )
    return ContextReceipt(
        query=bundle.query,
        expanded_query=bundle.retrieval_metadata.expanded_query,
        expansion_provenance=bundle.retrieval_metadata.expansion_provenance,
        token_budget=ContextReceiptTokenBudget(
            requested=bundle.token_budget,
            consumed=bundle.token_count,
        ),
        index_revision=index_revision,
        freshness=freshness,
        freshness_checked_at=freshness_checked_at,
        index_fresh_at=index_fresh_at,
        watch_fresh_at=watch_fresh_at,
        returned_context=returned,
        included_edges=included,
        omitted_edges=omitted,
        skipped_candidates=skipped,
        semantic_providers=list(semantic_providers),
        runtime_providers=list(runtime_providers),
        history_providers=list(history_providers_resolved),
        history_eligibility=history_eligibility,
        documentation_providers=list(documentation_providers),
        returned_total=len(returned),
        skipped_total=len(skipped_all),
        included_edges_total=len(included_all),
        omitted_edges_total=len(omitted_all),
        context_complete=_completion_status(bundle, freshness, omitted, skipped),
        context_complete_reason=_completion_reason(bundle, freshness, omitted, skipped),
        recommended_next_action=_recommended_action(bundle, freshness, omitted, skipped),
    )


def build_scout_receipt(
    result: ScoutResult,
    direct_receipt: ContextReceipt | None,
    *,
    file_hashes: dict[str, str] | None = None,
) -> ContextReceipt | None:
    if direct_receipt is None:
        return None
    file_hashes = file_hashes or {}
    selected_handles = set(result.fetch_plan.handles)
    selected_file_paths = _selected_scout_file_paths(result)
    returned = [
        ContextReceiptItem(
            handle=item.handle,
            file_path=item.path,
            start_line=1,
            end_line=item.lines,
            content_hash=file_hashes.get(item.path, ""),
            symbols=[symbol.name for symbol in result.symbols if symbol.file_path == item.path],
            score=item.score,
            reason_codes=[item.reason],
        )
        for item in sorted(result.ranked_files, key=lambda file: file.path)
    ]
    skipped = [
        item
        for item in direct_receipt.skipped_candidates
        if item.file_path not in selected_file_paths
        and (item.handle is None or item.handle not in selected_handles)
    ]
    for file_path, reason in sorted(result.fetch_plan.file_reasons.items()):
        if reason.startswith("selected"):
            continue
        if file_path in selected_file_paths or file_handle(file_path) in selected_handles:
            continue
        if reason.startswith("duplicate_handle"):
            code = ContextSkippedReason.DUPLICATE
        elif reason.startswith("pruned_by_cap"):
            code = ContextSkippedReason.OVER_BUDGET
        elif reason.startswith("no_precise_handle") or reason == "missing_precise_handle":
            code = ContextSkippedReason.UNSUPPORTED_GRAMMAR
        else:
            code = ContextSkippedReason.BELOW_THRESHOLD
        skipped.append(
            ContextSkippedCandidate(
                file_path=file_path,
                handle=file_handle(file_path),
                reason=code,
                detail=reason,
            )
        )
    merged_skipped = sorted(skipped, key=_skipped_sort_key)[:_MAX_RECEIPT_SKIPPED]
    completion_reason = _reason_from_skipped(merged_skipped)
    action = _action_from_reason(completion_reason, has_skipped=bool(merged_skipped))
    status = (
        ContextCompletenessStatus.INCOMPLETE if merged_skipped else direct_receipt.context_complete
    )
    return direct_receipt.model_copy(
        update={
            "token_budget": ContextReceiptTokenBudget(
                requested=result.budget.token_budget,
                consumed=result.budget.token_count,
            ),
            "returned_context": returned,
            "skipped_candidates": merged_skipped,
            "returned_total": len(returned),
            "skipped_total": len(skipped),
            "context_complete": status,
            "context_complete_reason": (
                completion_reason or direct_receipt.context_complete_reason
            ),
            "recommended_next_action": action or direct_receipt.recommended_next_action,
        }
    )


def _selected_scout_file_paths(result: ScoutResult) -> set[str]:
    paths = {
        file_path
        for file_path, reason in result.fetch_plan.file_reasons.items()
        if reason.startswith("selected")
    }
    for handle_value in result.fetch_plan.handles:
        handle = parse_scout_handle(handle_value)
        if handle is None:
            continue
        if handle.kind == "file":
            paths.add(handle.value)
        elif handle.kind in {"chunk", "symbol"} and "::" in handle.value:
            paths.add(handle.value.split("::", 1)[0])
    return paths


def receipt_edges_for_graph(
    graph: DependencyGraph,
    included_files: set[str],
    skipped_reason_by_file: dict[str, ContextSkippedReason] | None = None,
) -> tuple[list[ContextReceiptEdge], list[ContextReceiptEdge]]:
    skipped_reason_by_file = skipped_reason_by_file or {}
    included: list[ContextReceiptEdge] = []
    omitted: list[ContextReceiptEdge] = []
    for edge in sorted(
        graph.file_edges(),
        key=lambda item: (item.source, item.target, item.kind.value),
    ):
        receipt_edge = _receipt_edge(edge)
        if edge.source in included_files and edge.target in included_files:
            included.append(receipt_edge)
        elif edge.source in included_files or edge.target in included_files:
            missing_path = edge.target if edge.source in included_files else edge.source
            reason = (
                ContextOmittedEdgeReason.OVER_BUDGET
                if skipped_reason_by_file.get(missing_path) == ContextSkippedReason.OVER_BUDGET
                else ContextOmittedEdgeReason.BELOW_THRESHOLD
            )
            omitted.append(receipt_edge.model_copy(update={"reason": reason}))
    return included, omitted


def skipped_candidates_for_ranked(
    ranked: list[RankedChunk],
    included: list[RankedChunk],
    *,
    token_budget: int,
    total_tokens: int,
) -> list[ContextSkippedCandidate]:
    included_ids = {item.chunk.id for item in included}
    included_ranges: dict[str, list[tuple[int, int]]] = {}
    for item in included:
        included_ranges.setdefault(item.chunk.file_path, []).append(
            (item.chunk.start_line, item.chunk.end_line)
        )
    skipped: list[ContextSkippedCandidate] = []
    for item in ranked:
        if item.chunk.id in included_ids:
            continue
        reason = _skip_reason(
            item.chunk,
            included_ranges,
            token_budget=token_budget,
            total_tokens=total_tokens,
        )
        skipped.append(
            ContextSkippedCandidate(
                file_path=item.chunk.file_path,
                handle=chunk_handle(item.chunk.id),
                symbol=item.chunk.symbol_name,
                score=item.final_score,
                reason=reason,
            )
        )
    return skipped


def stale_index_skipped_candidate() -> ContextSkippedCandidate:
    return ContextSkippedCandidate(
        file_path="",
        reason=ContextSkippedReason.STALE_INDEX,
        detail="index freshness is stale or unknown",
    )


def unsupported_grammar_skipped_candidate(count: int) -> ContextSkippedCandidate:
    return ContextSkippedCandidate(
        file_path="",
        reason=ContextSkippedReason.UNSUPPORTED_GRAMMAR,
        detail=f"parse failures: {count}",
    )


def _returned_context(bundle: ContextBundle) -> list[ContextReceiptItem]:
    items: list[ContextReceiptItem] = []
    seed_paths = set(bundle.retrieval_metadata.seed_file_paths)
    expanded_paths = set(bundle.retrieval_metadata.expanded_file_paths)
    for ranked in bundle.chunks:
        chunk = ranked.chunk
        reason_codes = [bundle.retrieval_metadata.strategy or "ranked"]
        if chunk.file_path in seed_paths:
            reason_codes.append("seed")
        if chunk.file_path in expanded_paths:
            reason_codes.append("expanded")
        symbols = [value for value in (chunk.symbol_name, chunk.qualified_name) if value]
        items.append(
            ContextReceiptItem(
                handle=chunk_handle(chunk.id),
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content_hash=chunk_content_hash(chunk),
                symbols=list(dict.fromkeys(symbols)),
                score=ranked.final_score,
                reason_codes=reason_codes,
            )
        )
    return sorted(
        items,
        key=lambda item: (item.file_path, item.start_line, item.end_line, item.handle),
    )


def _receipt_edge(edge: Edge) -> ContextReceiptEdge:
    return ContextReceiptEdge(
        source=edge.source,
        target=edge.target,
        kind=edge.kind,
        confidence=edge.confidence,
        confidence_score=edge.confidence_score,
        evidence=edge.evidence,
        provider=edge.provider,
        provider_version=edge.provider_version,
    )


def _skip_reason(
    chunk: CodeChunk,
    included_ranges: dict[str, list[tuple[int, int]]],
    *,
    token_budget: int,
    total_tokens: int,
) -> ContextSkippedReason:
    for start, end in included_ranges.get(chunk.file_path, []):
        if start <= chunk.start_line and chunk.end_line <= end:
            return ContextSkippedReason.DUPLICATE
        if chunk.start_line <= start and end <= chunk.end_line:
            return ContextSkippedReason.DUPLICATE
    chunk_tokens = (
        chunk.token_count if chunk.token_count > 0 else int(len(chunk.content.split()) * 1.3)
    )
    if total_tokens >= token_budget or total_tokens + chunk_tokens > token_budget:
        return ContextSkippedReason.OVER_BUDGET
    return ContextSkippedReason.BELOW_THRESHOLD


def _completion_status(
    bundle: ContextBundle,
    freshness: ContextFreshness,
    omitted_edges: list[ContextReceiptEdge],
    skipped_candidates: list[ContextSkippedCandidate],
) -> ContextCompletenessStatus:
    if freshness in {ContextFreshness.DIRTY, ContextFreshness.UNKNOWN}:
        return ContextCompletenessStatus.UNKNOWN
    if bundle.truncated or omitted_edges or skipped_candidates:
        return ContextCompletenessStatus.INCOMPLETE
    return ContextCompletenessStatus.COMPLETE


def _skipped_sort_key(item: ContextSkippedCandidate) -> tuple[int, str, str, str, str]:
    priority = {
        ContextSkippedReason.STALE_INDEX: 0,
        ContextSkippedReason.UNSUPPORTED_GRAMMAR: 1,
        ContextSkippedReason.OVER_BUDGET: 2,
        ContextSkippedReason.DEPENDENCY_FRONTIER_CUT: 3,
        ContextSkippedReason.DUPLICATE: 4,
        ContextSkippedReason.BELOW_THRESHOLD: 5,
        ContextSkippedReason.TEST_DEPRIORITIZED: 6,
    }
    return (
        priority[item.reason],
        item.reason.value,
        item.file_path,
        item.symbol or "",
        item.handle or "",
    )


def _reason_from_skipped(
    skipped_candidates: list[ContextSkippedCandidate],
) -> ContextCompletenessReason | None:
    if any(item.reason == ContextSkippedReason.STALE_INDEX for item in skipped_candidates):
        return ContextCompletenessReason.STALE_INDEX
    if any(item.reason == ContextSkippedReason.OVER_BUDGET for item in skipped_candidates):
        return ContextCompletenessReason.BUDGET_EXHAUSTED
    if any(item.reason == ContextSkippedReason.UNSUPPORTED_GRAMMAR for item in skipped_candidates):
        return ContextCompletenessReason.UNSUPPORTED_GRAMMAR
    if any(item.reason == ContextSkippedReason.DUPLICATE for item in skipped_candidates):
        return ContextCompletenessReason.DUPLICATE_SUPPRESSED
    if skipped_candidates:
        return ContextCompletenessReason.UNKNOWN
    return None


def _action_from_reason(
    reason: ContextCompletenessReason | None,
    *,
    has_skipped: bool,
) -> ContextRecommendedAction | None:
    if reason == ContextCompletenessReason.STALE_INDEX:
        return ContextRecommendedAction.REFRESH_INDEX
    if reason == ContextCompletenessReason.BUDGET_EXHAUSTED:
        return ContextRecommendedAction.RAISE_BUDGET
    if has_skipped:
        return ContextRecommendedAction.FETCH_SKIPPED_CANDIDATE
    if reason == ContextCompletenessReason.COMPLETE:
        return ContextRecommendedAction.USE_BUNDLE
    return None


def _completion_reason(
    bundle: ContextBundle,
    freshness: ContextFreshness,
    omitted_edges: list[ContextReceiptEdge],
    skipped_candidates: list[ContextSkippedCandidate],
) -> ContextCompletenessReason:
    if freshness == ContextFreshness.DIRTY:
        return ContextCompletenessReason.STALE_INDEX
    if any(item.reason == ContextSkippedReason.STALE_INDEX for item in skipped_candidates):
        return ContextCompletenessReason.STALE_INDEX
    if omitted_edges:
        return ContextCompletenessReason.DEPENDENCY_FRONTIER_CUT
    skipped_reason = _reason_from_skipped(skipped_candidates)
    if skipped_reason is not None:
        return skipped_reason
    if bundle.truncated:
        return ContextCompletenessReason.BUDGET_EXHAUSTED
    if not bundle.chunks:
        return ContextCompletenessReason.NO_CANDIDATES
    if freshness == ContextFreshness.UNKNOWN:
        return ContextCompletenessReason.UNKNOWN
    return ContextCompletenessReason.COMPLETE


def _recommended_action(
    bundle: ContextBundle,
    freshness: ContextFreshness,
    omitted_edges: list[ContextReceiptEdge],
    skipped_candidates: list[ContextSkippedCandidate],
) -> ContextRecommendedAction:
    reason = _completion_reason(bundle, freshness, omitted_edges, skipped_candidates)
    if reason == ContextCompletenessReason.DEPENDENCY_FRONTIER_CUT:
        return (
            ContextRecommendedAction.RAISE_BUDGET
            if any(edge.reason == ContextOmittedEdgeReason.OVER_BUDGET for edge in omitted_edges)
            else ContextRecommendedAction.FETCH_SKIPPED_CANDIDATE
        )
    action = _action_from_reason(reason, has_skipped=bool(skipped_candidates))
    if action is not None:
        return action
    if reason == ContextCompletenessReason.COMPLETE:
        return ContextRecommendedAction.USE_BUNDLE
    return ContextRecommendedAction.MANUAL_REVIEW
