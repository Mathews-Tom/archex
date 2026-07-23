"""Named retrieval profiles: fast/balanced/deep IndexConfig presets.

A `RetrievalProfile` (`archex.models.RetrievalProfile`) is a convenience
shorthand for a common retrieval-feature combination — a preset of
`IndexConfig` boolean feature toggles selecting a cost/quality tradeoff,
not a separate configuration system. `index_config_for_profile` layers a
profile's toggles onto a caller-supplied base `IndexConfig`, preserving
every other field (embedder, chunker, token encoding, quantization, and so
on) from that base so a profile composes with repo-level and CLI-level
configuration instead of discarding it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.models import RetrievalProfile

if TYPE_CHECKING:
    from archex.models import IndexConfig

#: Boolean feature toggles that define each profile's cost/quality tradeoff.
#: Every other IndexConfig field (embedder, chunker, token budgets, etc.) is
#: left untouched — profiles select *which* retrieval signals run, not how
#: they're configured.
_PROFILE_OVERRIDES: dict[RetrievalProfile, dict[str, bool]] = {
    RetrievalProfile.FAST: {
        "vector": False,
        "module_prefilter": False,
        "rerank": False,
    },
    RetrievalProfile.BALANCED: {
        "vector": False,
        "module_prefilter": True,
        "rerank": False,
    },
    RetrievalProfile.DEEP: {
        "vector": True,
        "module_prefilter": True,
        "rerank": True,
    },
}


def index_config_for_profile(profile: RetrievalProfile, base: IndexConfig) -> IndexConfig:
    """Return `base` with `profile`'s retrieval feature flags applied.

    `FAST` performs zero vector/model thread work: both `vector` and
    `rerank` are forced off, so no embedder or cross-encoder model ever
    loads or runs for a fast-profile query, regardless of what `base`
    specifies.
    """
    return base.model_copy(update=_PROFILE_OVERRIDES[profile])
