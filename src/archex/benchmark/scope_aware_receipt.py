"""Strict benchmark-sidecar receipts for R26 Candidate A."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archex.benchmark.scope_aware_candidate import (
    CANDIDATE_LIMIT_PER_SCOPE,
    PARTICIPATION_THRESHOLD,
    ScopeMap,
    ScopeRanking,
)

if TYPE_CHECKING:
    from archex.models import ContextBundle

ScopeDecisionReason = Literal[
    "normalized_top_at_or_above_threshold",
    "normalized_top_below_threshold",
    "non_positive_global_max",
]
CAMPAIGN_ID = "r27-scope-aware-monorepo-ranking"
CANDIDATE_ARM = "scope_aware_candidate"
NORMALIZATION_IDENTITY = "repository_global_max_positive_raw_bm25:v1"
_PAYLOAD_FIELDS = (
    "query",
    "chunks",
    "structural_context",
    "type_definitions",
    "dependency_summary",
    "token_count",
    "token_budget",
    "truncated",
)


class ScopeReceiptError(ValueError):
    """Raised when a scope receipt cannot reconcile to the ranked candidate set."""


class _ReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SearchedScopeReceipt(_ReceiptModel):
    """One searched scope's raw candidates, gate decision, and dedup contribution."""

    scope: str
    raw_candidate_count: int = Field(ge=0, le=CANDIDATE_LIMIT_PER_SCOPE)
    raw_top_score: float = Field(ge=0.0)
    normalized_top_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["included", "rejected"]
    reason: ScopeDecisionReason
    post_dedup_contribution_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_decision(self) -> SearchedScopeReceipt:
        if self.decision == "included":
            if self.reason != "normalized_top_at_or_above_threshold":
                raise ScopeReceiptError("included scope has a rejection reason")
            if self.normalized_top_score < PARTICIPATION_THRESHOLD:
                raise ScopeReceiptError("included scope is below the participation threshold")
        elif self.reason == "normalized_top_at_or_above_threshold":
            raise ScopeReceiptError("rejected scope has the included reason")
        elif (
            self.reason == "normalized_top_below_threshold"
            and self.normalized_top_score >= PARTICIPATION_THRESHOLD
        ):
            raise ScopeReceiptError("rejected scope is at or above the participation threshold")
        if self.post_dedup_contribution_count > self.raw_candidate_count:
            raise ScopeReceiptError("scope contribution exceeds its raw candidate count")
        return self


class MultiScopeReceipt(_ReceiptModel):
    """Fail-closed sidecar for one multi-scope candidate ranking."""

    schema_version: Literal[1] = 1
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"] = CAMPAIGN_ID
    task_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    arm: Literal["scope_aware_candidate"] = CANDIDATE_ARM
    mode: Literal["scope_aware_ranking"] = "scope_aware_ranking"
    searched_scopes: tuple[SearchedScopeReceipt, ...] = Field(min_length=2)
    shared_idf_identity: str = Field(min_length=1)
    normalization_identity: Literal["repository_global_max_positive_raw_bm25:v1"] = (
        NORMALIZATION_IDENTITY
    )
    candidate_limit_per_scope: Literal[150] = CANDIDATE_LIMIT_PER_SCOPE
    participation_threshold: float = Field(
        default=PARTICIPATION_THRESHOLD,
        ge=PARTICIPATION_THRESHOLD,
        le=PARTICIPATION_THRESHOLD,
    )
    repository_global_max_raw_score: float = Field(ge=0.0)
    included_scope_count: int = Field(ge=0)
    rejected_scope_count: int = Field(ge=0)
    pre_dedup_candidate_count: int = Field(ge=0)
    duplicate_candidate_count: int = Field(ge=0)
    post_dedup_candidate_count: int = Field(ge=0)
    final_candidate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _reconcile(self) -> MultiScopeReceipt:
        scopes = [scope.scope for scope in self.searched_scopes]
        if scopes != sorted(set(scopes)):
            raise ScopeReceiptError("searched scopes must be unique and bytewise sorted")
        included = [scope for scope in self.searched_scopes if scope.decision == "included"]
        rejected = [scope for scope in self.searched_scopes if scope.decision == "rejected"]
        if self.included_scope_count != len(included):
            raise ScopeReceiptError("included scope count does not reconcile")
        if self.rejected_scope_count != len(rejected):
            raise ScopeReceiptError("rejected scope count does not reconcile")
        if self.included_scope_count + self.rejected_scope_count != len(self.searched_scopes):
            raise ScopeReceiptError("searched scope count does not reconcile")
        raw_count = sum(scope.raw_candidate_count for scope in self.searched_scopes)
        if self.pre_dedup_candidate_count != raw_count:
            raise ScopeReceiptError("pre-dedup candidate count does not reconcile")
        if self.pre_dedup_candidate_count - self.duplicate_candidate_count != (
            self.post_dedup_candidate_count
        ):
            raise ScopeReceiptError("post-dedup candidate count does not reconcile")
        contribution_count = sum(
            scope.post_dedup_contribution_count for scope in self.searched_scopes
        )
        if contribution_count != self.post_dedup_candidate_count:
            raise ScopeReceiptError("scope contribution count does not reconcile")
        included_contribution_count = sum(scope.post_dedup_contribution_count for scope in included)
        if self.final_candidate_count != included_contribution_count:
            raise ScopeReceiptError("final candidate count does not reconcile")
        if self.final_candidate_count > self.post_dedup_candidate_count:
            raise ScopeReceiptError("final candidate count exceeds post-dedup candidates")
        observed_global_max = max(scope.raw_top_score for scope in self.searched_scopes)
        if not math.isclose(
            self.repository_global_max_raw_score,
            observed_global_max,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ScopeReceiptError("repository-global maximum does not reconcile")
        if self.repository_global_max_raw_score == 0.0:
            if included or self.final_candidate_count:
                raise ScopeReceiptError("non-positive global maximum must reject every scope")
            if any(
                scope.reason != "non_positive_global_max" or scope.normalized_top_score != 0.0
                for scope in self.searched_scopes
            ):
                raise ScopeReceiptError("non-positive global maximum receipt is inconsistent")
            return self
        if not included:
            raise ScopeReceiptError("positive global maximum must include at least one scope")
        for scope in self.searched_scopes:
            expected_normalized = scope.raw_top_score / self.repository_global_max_raw_score
            if not math.isclose(
                scope.normalized_top_score,
                expected_normalized,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ScopeReceiptError("normalized top score does not reconcile")
            if scope.reason == "non_positive_global_max":
                raise ScopeReceiptError("positive global maximum has a non-positive reason")
        return self


class SingleScopeReceipt(_ReceiptModel):
    """Sidecar proving that one scope used unchanged archex_query end to end."""

    schema_version: Literal[1] = 1
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"] = CAMPAIGN_ID
    task_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    arm: Literal["scope_aware_candidate"] = CANDIDATE_ARM
    mode: Literal["archex_query_bypass"] = "archex_query_bypass"
    scope_count: Literal[1] = 1
    scope: str
    decision: Literal["included"] = "included"
    reason: Literal["single_scope_bypass"] = "single_scope_bypass"
    included_scope_count: Literal[1] = 1
    rejected_scope_count: Literal[0] = 0
    participation_threshold: float = Field(
        default=PARTICIPATION_THRESHOLD,
        ge=PARTICIPATION_THRESHOLD,
        le=PARTICIPATION_THRESHOLD,
    )
    shared_idf_identity: Literal["archex_query_bypass:not_applicable"] = (
        "archex_query_bypass:not_applicable"
    )
    normalization_identity: Literal["archex_query_bypass:not_applicable"] = (
        "archex_query_bypass:not_applicable"
    )
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


ScopeAwareReceipt = MultiScopeReceipt | SingleScopeReceipt


def canonical_context_payload(bundle: ContextBundle) -> bytes:
    """Return the exact receipt/timing-free payload projection frozen by R27."""
    raw = bundle.model_dump(mode="json")
    projection = {field: raw[field] for field in _PAYLOAD_FIELDS}
    return json.dumps(
        projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def context_payload_sha256(bundle: ContextBundle) -> str:
    """Hash the frozen canonical ContextBundle payload projection."""
    return hashlib.sha256(canonical_context_payload(bundle)).hexdigest()


def build_single_scope_receipt(
    *,
    task_id: str,
    repository_id: str,
    scope_map: ScopeMap,
    bundle: ContextBundle,
) -> SingleScopeReceipt:
    """Build the out-of-band receipt for an exact archex_query bypass."""
    if len(scope_map.scopes) != 1:
        raise ScopeReceiptError("single-scope receipt requires exactly one non-empty scope")
    return SingleScopeReceipt(
        task_id=task_id,
        repository_id=repository_id,
        scope=scope_map.scopes[0],
        payload_sha256=context_payload_sha256(bundle),
    )


def build_multi_scope_receipt(
    *,
    task_id: str,
    repository_id: str,
    ranking: ScopeRanking,
    shared_idf_identity: str,
) -> MultiScopeReceipt:
    """Build and validate the strict out-of-band receipt for a candidate ranking."""
    searched = tuple(
        SearchedScopeReceipt(
            scope=search.scope,
            raw_candidate_count=len(search.candidates),
            raw_top_score=search.raw_top_score,
            normalized_top_score=search.normalized_top_score,
            decision="included" if search.included else "rejected",
            reason=cast("ScopeDecisionReason", search.reason),
            post_dedup_contribution_count=search.post_dedup_contribution_count,
        )
        for search in ranking.searches
    )
    included = sum(scope.decision == "included" for scope in searched)
    return MultiScopeReceipt(
        task_id=task_id,
        repository_id=repository_id,
        searched_scopes=searched,
        shared_idf_identity=shared_idf_identity,
        repository_global_max_raw_score=ranking.repository_global_max_raw_score,
        included_scope_count=included,
        rejected_scope_count=len(searched) - included,
        pre_dedup_candidate_count=ranking.pre_dedup_candidate_count,
        duplicate_candidate_count=ranking.duplicate_candidate_count,
        post_dedup_candidate_count=ranking.post_dedup_candidate_count,
        final_candidate_count=len(ranking.candidates),
    )


def canonical_receipt_json(receipt: ScopeAwareReceipt) -> str:
    """Serialize a validated sidecar deterministically for BenchmarkResult provenance."""
    return json.dumps(
        receipt.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
