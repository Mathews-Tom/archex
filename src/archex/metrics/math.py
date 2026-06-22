"""Token savings calculations for local usage metrics."""

from __future__ import annotations

from dataclasses import dataclass

BASELINE_RETURNED_FULL_FILES = "returned_full_files"


@dataclass(frozen=True)
class TokenSavings:
    tokens_returned: int
    tokens_raw_equivalent: int
    tokens_saved: int
    savings_pct: float
    whole_repo_tokens: int | None = None
    whole_repo_tokens_avoided: int | None = None
    targeted_read_tokens: int | None = None
    tokens_saved_vs_targeted_read: int | None = None
    savings_pct_vs_targeted_read: float | None = None
    baseline_type: str = BASELINE_RETURNED_FULL_FILES


def compute_token_savings(
    *,
    tokens_returned: int,
    tokens_raw_equivalent: int,
    whole_repo_tokens: int | None = None,
    targeted_read_tokens: int | None = None,
) -> TokenSavings:
    """Compute savings against the full-file and (optional) targeted-read baselines.

    The full-file baseline is compression vs a naive full-file paste. The
    targeted-read baseline (when provided) is the realistic counterfactual of
    reading the matched line ranges plus a small context window; it is omitted
    (None) when spans are unavailable for the event.
    """
    _validate_non_negative("tokens_returned", tokens_returned)
    _validate_non_negative("tokens_raw_equivalent", tokens_raw_equivalent)
    if whole_repo_tokens is not None:
        _validate_non_negative("whole_repo_tokens", whole_repo_tokens)
    if targeted_read_tokens is not None:
        _validate_non_negative("targeted_read_tokens", targeted_read_tokens)

    tokens_saved = max(tokens_raw_equivalent - tokens_returned, 0)
    savings_pct = (
        (tokens_saved / tokens_raw_equivalent * 100.0) if tokens_raw_equivalent > 0 else 0.0
    )
    whole_repo_tokens_avoided = (
        max(whole_repo_tokens - tokens_returned, 0) if whole_repo_tokens is not None else None
    )
    if targeted_read_tokens is not None:
        tokens_saved_vs_targeted_read: int | None = max(targeted_read_tokens - tokens_returned, 0)
        savings_pct_vs_targeted_read: float | None = (
            (tokens_saved_vs_targeted_read / targeted_read_tokens * 100.0)
            if targeted_read_tokens > 0
            else 0.0
        )
    else:
        tokens_saved_vs_targeted_read = None
        savings_pct_vs_targeted_read = None
    return TokenSavings(
        tokens_returned=tokens_returned,
        tokens_raw_equivalent=tokens_raw_equivalent,
        tokens_saved=tokens_saved,
        savings_pct=savings_pct,
        whole_repo_tokens=whole_repo_tokens,
        whole_repo_tokens_avoided=whole_repo_tokens_avoided,
        targeted_read_tokens=targeted_read_tokens,
        tokens_saved_vs_targeted_read=tokens_saved_vs_targeted_read,
        savings_pct_vs_targeted_read=savings_pct_vs_targeted_read,
    )


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
