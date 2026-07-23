"""Primary agent-facing context facade: `query, intent, profile, filters, budgets, handles`.

`archex.api.context()` is the single documented primary path for an agent to
retrieve code context. It is a thin wrapper over the canonical `query()`
retrieval pipeline (`archex.api.query`) — it never reranks, never talks to a
provider, and never invents a second receipt or handle authority. Every
field it adds is either:

- a **route decision** (`ContextRouteDecision`): how this module interpreted
  `intent`/`profile`/`budgets`/`handles` before calling `query()`, built once
  from the same canonical classifiers `query()` and `scout()` already use
  (`archex.serve.intent.classify_intent`, `archex.serve.modality`), or
- a **read-only view** onto the `ContextBundle` `query()` already produced
  (`ContextResult`'s `candidate_map`, `fetch_handles`, `selected_code`,
  `relation_paths`, `receipt`, `next_action` are all computed properties over
  `ContextResult.bundle`, never a second copy of that data).

The only genuinely new behavior this module performs is `ContextFilters`: a
deterministic, post-retrieval include/exclude/language predicate over the
chunks `query()` already ranked. It never adds candidates, never reorders
survivors, and moves every excluded chunk into the existing
`ContextReceipt.skipped_candidates` ledger with reason
`ContextSkippedReason.FILTER_EXCLUDED` so the receipt still accounts for
every candidate `query()` considered.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

from archex.models import (
    CodeChunk,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextRecommendedAction,
    ContextSkippedCandidate,
    ContextSkippedReason,
    RankedChunk,
    RetrievalProfile,
    ScoringWeights,
    StructuralContext,
)
from archex.scout import chunk_handle
from archex.serve.context import estimate_tokens
from archex.serve.intent import (
    DEFAULT_TOKEN_BUDGET,
    INTENT_TOKEN_BUDGETS,
    INTENT_WEIGHTS,
    QueryIntent,
    classify_intent,
)
from archex.serve.modality import BudgetTier, QueryModality, budget_tier, classify_modality

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class ContextFilters(BaseModel):
    """Deterministic post-retrieval candidate filters.

    Filters only exclude candidates `query()` already ranked; they never
    change ranking, add candidates, or trigger a reindex. `languages`
    matches `CodeChunk.language` exactly. `include_paths`/`exclude_paths`
    are `fnmatch`-style glob patterns matched case-sensitively against a
    chunk's `file_path` (e.g. `"src/auth/**"`, `"*.py"`).
    """

    include_paths: list[str] = []
    exclude_paths: list[str] = []
    languages: list[str] = []

    @model_validator(mode="after")
    def _validate_non_blank(self) -> ContextFilters:
        for value in (*self.include_paths, *self.exclude_paths, *self.languages):
            if not value.strip():
                raise ValueError("filter values must not be blank")
        return self

    def is_empty(self) -> bool:
        return not (self.include_paths or self.exclude_paths or self.languages)

    def matches(self, chunk: CodeChunk) -> bool:
        """Return True when `chunk` survives every active filter clause."""
        if self.languages and chunk.language not in self.languages:
            return False
        if self.include_paths and not any(
            fnmatch.fnmatchcase(chunk.file_path, pattern) for pattern in self.include_paths
        ):
            return False
        return not (
            self.exclude_paths
            and any(fnmatch.fnmatchcase(chunk.file_path, pattern) for pattern in self.exclude_paths)
        )


class ContextBudgets(BaseModel):
    """Token-budget input for `context()`.

    `token_budget` is an explicit override. Omitted, `context()` resolves a
    budget from the pinned `intent` (`ContextRequest.intent`) when one was
    given, or falls back to `query()`'s own intent-routed auto-scaling —
    identical to `archex query`'s undecorated default. Either way the
    resolved value is receipt-explained via `ContextRouteDecision` and
    `ContextReceipt.token_budget`.
    """

    token_budget: int | None = None

    @model_validator(mode="after")
    def _validate_budget(self) -> ContextBudgets:
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")
        return self


class ContextRequest(BaseModel):
    """The shared `context()` request contract.

    `query, intent, profile, filters, budgets, handles` — the six inputs
    `archex.api.context()` documents as the primary agent path.
    """

    query: str
    intent: QueryIntent | None = None
    profile: RetrievalProfile | None = None
    filters: ContextFilters = ContextFilters()
    budgets: ContextBudgets = ContextBudgets()
    handles: list[str] = []

    @model_validator(mode="after")
    def _validate_request(self) -> ContextRequest:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        for handle in self.handles:
            if not handle.strip():
                raise ValueError("handles must not contain blank entries")
        return self


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ContextRouteDecision(BaseModel):
    """How `context()` interpreted a request before calling `query()`.

    Every field here is resolved once, deterministically, from the same
    canonical classifiers `query()`/`scout()` already use — this is a
    read-only explanation of routing, not a second retrieval authority.
    """

    resolved_intent: QueryIntent
    intent_source: Literal["explicit", "auto"]
    resolved_modality: QueryModality
    resolved_profile: RetrievalProfile | None
    profile_source: Literal["explicit", "none"]
    resolved_budget_tier: BudgetTier
    token_budget_requested: int
    budget_source: Literal["explicit", "intent_default"]
    handles_mode: bool
    filters_active: bool
    reasons: list[str] = []


class ContextResult(BaseModel):
    """The primary agent-facing `context()` result.

    `bundle` and `route` are the only stored fields; every other attribute
    is a computed, read-only view over `bundle` so there is exactly one
    receipt and one handle authority (`bundle.receipt`, chunk handles on
    `bundle.chunks`).
    """

    bundle: ContextBundle
    route: ContextRouteDecision

    @computed_field  # type: ignore[prop-decorator]
    @property
    def candidate_map(self) -> list[ContextReceiptItem]:
        """Compact map of every returned candidate: handle, location, score."""
        return list(self.bundle.receipt.returned_context) if self.bundle.receipt else []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fetch_handles(self) -> list[str]:
        """Exact fetch handles for every returned candidate (`query(handles=...)`-ready)."""
        return [item.handle for item in self.candidate_map]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def selected_code(self) -> list[RankedChunk]:
        """The ranked, budget-fit code chunks `query()` selected."""
        return list(self.bundle.chunks)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def relation_paths(self) -> StructuralContext:
        """Dependency/module relation paths for the selected code."""
        return self.bundle.structural_context

    @computed_field  # type: ignore[prop-decorator]
    @property
    def receipt(self) -> ContextReceipt | None:
        """The single receipt authority for this result (`bundle.receipt`)."""
        return self.bundle.receipt

    @computed_field  # type: ignore[prop-decorator]
    @property
    def next_action(self) -> ContextRecommendedAction | None:
        """The receipt's recommended next action, when a receipt is present."""
        return self.bundle.receipt.recommended_next_action if self.bundle.receipt else None


@dataclass(frozen=True)
class ContextRouteResolution:
    """Internal: resolved `query()` call parameters plus the route decision to report."""

    scoring_weights: ScoringWeights | None
    token_budget: int
    explicit_token_budget: bool
    route: ContextRouteDecision


# ---------------------------------------------------------------------------
# Route resolution
# ---------------------------------------------------------------------------


def resolve_context_route(request: ContextRequest) -> ContextRouteResolution:
    """Resolve `request` into `query()` call parameters plus a `ContextRouteDecision`.

    Reuses the existing intent/modality/budget-tier classifiers and preset
    tables verbatim (`classify_intent`, `classify_modality`, `budget_tier`,
    `INTENT_WEIGHTS`, `INTENT_TOKEN_BUDGETS`) — no new ranking or budget
    logic is introduced here, only explicit selection into what already
    exists.
    """
    handles_mode = bool(request.handles)
    reasons: list[str] = []

    intent_source: Literal["explicit", "auto"]
    scoring_weights: ScoringWeights | None
    if request.intent is not None:
        resolved_intent = request.intent
        intent_source = "explicit"
        scoring_weights = INTENT_WEIGHTS[resolved_intent]
        reasons.append(f"intent pinned explicitly to '{resolved_intent.value}'")
    else:
        resolved_intent = classify_intent(request.query)
        intent_source = "auto"
        scoring_weights = None
        reasons.append(f"intent auto-classified as '{resolved_intent.value}' from the query text")

    budget_source: Literal["explicit", "intent_default"]
    if request.budgets.token_budget is not None:
        token_budget = request.budgets.token_budget
        explicit_token_budget = True
        budget_source = "explicit"
        reasons.append(f"token budget pinned explicitly to {token_budget}")
    elif request.intent is not None:
        token_budget = INTENT_TOKEN_BUDGETS[resolved_intent]
        explicit_token_budget = True
        budget_source = "intent_default"
        reasons.append(
            f"token budget resolved from pinned intent '{resolved_intent.value}': {token_budget}"
        )
    else:
        token_budget = DEFAULT_TOKEN_BUDGET
        explicit_token_budget = False
        budget_source = "intent_default"
        reasons.append("token budget left to query()'s own intent-routed auto-scaling")

    if handles_mode:
        reasons.append(
            f"{len(request.handles)} exact fetch handle(s) supplied — routed to direct "
            "handle fetch instead of broad search"
        )
    if not request.filters.is_empty():
        reasons.append("deterministic post-retrieval filters active")

    route = ContextRouteDecision(
        resolved_intent=resolved_intent,
        intent_source=intent_source,
        resolved_modality=classify_modality(request.query),
        resolved_profile=request.profile,
        profile_source="explicit" if request.profile is not None else "none",
        resolved_budget_tier=budget_tier(token_budget),
        token_budget_requested=token_budget,
        budget_source=budget_source,
        handles_mode=handles_mode,
        filters_active=not request.filters.is_empty(),
        reasons=reasons,
    )
    return ContextRouteResolution(
        scoring_weights=scoring_weights,
        token_budget=token_budget,
        explicit_token_budget=explicit_token_budget,
        route=route,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def apply_context_filters(bundle: ContextBundle, filters: ContextFilters) -> ContextBundle:
    """Apply `filters` to an already-retrieved `bundle`, in place, and return it.

    Excluded chunks are removed from `bundle.chunks` and moved from
    `bundle.receipt.returned_context` into `bundle.receipt.skipped_candidates`
    with `ContextSkippedReason.FILTER_EXCLUDED` — every candidate `query()`
    considered stays accounted for in the one receipt. `token_count` and
    `receipt.token_budget.consumed` are resummed via the same per-chunk
    `estimate_tokens` helper `query()` already uses for the surviving
    chunks; no new token-estimation logic is introduced.
    """
    kept: list[RankedChunk] = []
    filtered_out_ids: set[str] = set()
    for ranked in bundle.chunks:
        if filters.matches(ranked.chunk):
            kept.append(ranked)
        else:
            filtered_out_ids.add(ranked.chunk.id)
    if not filtered_out_ids:
        return bundle

    bundle.chunks = kept
    bundle.token_count = sum(estimate_tokens(ranked.chunk) for ranked in kept)

    receipt = bundle.receipt
    if receipt is None:
        return bundle
    receipt.token_budget.consumed = bundle.token_count

    filtered_handles = {chunk_handle(chunk_id) for chunk_id in filtered_out_ids}
    survivors: list[ContextReceiptItem] = []
    newly_skipped: list[ContextReceiptItem] = []
    for item in receipt.returned_context:
        (newly_skipped if item.handle in filtered_handles else survivors).append(item)

    receipt.returned_context = survivors
    receipt.skipped_candidates = [
        *receipt.skipped_candidates,
        *(
            ContextSkippedCandidate(
                file_path=item.file_path,
                reason=ContextSkippedReason.FILTER_EXCLUDED,
                handle=item.handle,
                symbol=item.symbols[0] if item.symbols else None,
                score=item.score,
                detail="excluded by context() filters",
            )
            for item in newly_skipped
        ),
    ]
    receipt.returned_total = len(receipt.returned_context)
    receipt.skipped_total = len(receipt.skipped_candidates)
    return bundle


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_context_markdown(result: ContextResult) -> str:
    """Render a `ContextResult` as Markdown: route, receipt summary, candidate map, code."""
    from archex.serve.renderers.markdown import render_markdown

    route = result.route
    receipt = result.receipt
    lines = [f"# Context: {result.bundle.query}", "", "## Route"]
    profile_label = route.resolved_profile.value if route.resolved_profile is not None else "none"
    lines.append(f"- intent: `{route.resolved_intent.value}` ({route.intent_source})")
    lines.append(f"- modality: `{route.resolved_modality.value}`")
    lines.append(f"- profile: `{profile_label}` ({route.profile_source})")
    lines.append(
        f"- budget tier: `{route.resolved_budget_tier.value}` — requested "
        f"{route.token_budget_requested} tokens ({route.budget_source})"
    )
    lines.append(f"- handles mode: `{route.handles_mode}`")
    lines.append(f"- filters active: `{route.filters_active}`")
    for reason in route.reasons:
        lines.append(f"  - {reason}")
    lines.append("")

    if receipt is not None:
        lines.append("## Receipt")
        lines.append(f"- index revision: `{receipt.index_revision}`")
        lines.append(f"- freshness: `{receipt.freshness.value}`")
        lines.append(
            f"- token budget: {receipt.token_budget.consumed}/"
            f"{receipt.token_budget.requested} consumed"
        )
        lines.append(f"- returned: {receipt.returned_total}, skipped: {receipt.skipped_total}")
        lines.append(
            f"- complete: `{receipt.context_complete.value}` "
            f"({receipt.context_complete_reason.value})"
        )
        lines.append(f"- next action: `{receipt.recommended_next_action.value}`")
        lines.append("")

    lines.append("## Candidate map")
    if result.candidate_map:
        for item in result.candidate_map:
            lines.append(
                f"- `{item.handle}` {item.file_path}:{item.start_line}-{item.end_line} "
                f"(score {item.score:.3f})"
            )
    else:
        lines.append("_(no candidates returned)_")
    lines.append("")

    lines.append("## Selected code")
    lines.append(render_markdown(result.bundle))
    return "\n".join(lines)
