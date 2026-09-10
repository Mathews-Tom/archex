"""Index-config compatibility predicates over stored index metadata.

An index records the `IndexConfig` fields that shaped its contents (chunker
identity, quantization, evidence providers). Anything that wants to reuse an
existing store — the ordinary cache hit path, delta indexing, or worktree
seeding — must first decide whether that recorded shape still matches the
config it is about to serve. This module holds that decision in one place,
keyed on a plain metadata mapping so a caller can answer it without opening
the store through `IndexStore` (whose constructor migrates the schema).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.pipeline.chunker import chunker_revision

if TYPE_CHECKING:
    from collections.abc import Mapping

    from archex.models import IndexConfig

#: Metadata keys that record the index-config shape of a built store. Read
#: these to build the mapping `index_config_metadata_mismatch` expects.
INDEX_CONFIG_METADATA_KEYS: tuple[str, ...] = (
    "chunker",
    "chunker_revision",
    "quantize_vectors",
    "quantize_bits",
    "semantic_evidence_providers",
    "runtime_evidence_providers",
    "history_evidence_providers",
    "documentation_evidence_providers",
)


def index_config_metadata_mismatch(
    metadata: Mapping[str, str | None],
    index_config: IndexConfig,
) -> str | None:
    """Return the first metadata key that disagrees with `index_config`, else None.

    A missing key is normalized to the value a store built with default
    settings would have recorded (absent quantization means disabled; an
    absent provider list means no providers), so a store written before a
    key existed is not treated as incompatible.
    """
    if metadata.get("chunker") != index_config.chunker:
        return "chunker"
    if metadata.get("chunker_revision") != chunker_revision(index_config.chunker):
        return "chunker_revision"

    stored_quantize = metadata.get("quantize_vectors")
    stored_quantize_enabled = stored_quantize == "True" if stored_quantize is not None else False
    if stored_quantize_enabled != index_config.quantize_vectors:
        return "quantize_vectors"
    if index_config.quantize_vectors and metadata.get("quantize_bits") != str(
        index_config.quantize_bits
    ):
        return "quantize_bits"

    providers = (
        ("semantic_evidence_providers", index_config.semantic_evidence_providers),
        ("runtime_evidence_providers", index_config.runtime_evidence_providers),
        ("history_evidence_providers", index_config.history_evidence_providers),
        ("documentation_evidence_providers", index_config.documentation_evidence_providers),
    )
    for key, configured in providers:
        if (metadata.get(key) or "") != ",".join(configured):
            return key
    return None
