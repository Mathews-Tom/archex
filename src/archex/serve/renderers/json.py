"""JSON renderer: serialize ArchProfile and ContextBundle to structured JSON."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pydantic import BaseModel

    from archex.models import ContextBundle

JsonValue: TypeAlias = "dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None"

# `CodeChunk`/`ScoutSymbol`-style fields that default to `None` when unset.
# Safe to drop unconditionally: a real value is always truthy JSON, so
# dropping only `None` never hides a meaningful falsy value (unlike, say,
# `RankedChunk.structural_score`, which defaults to `0.0` — a real score,
# not absence — and is intentionally excluded from this set).
_OMIT_WHEN_NONE = frozenset(
    {
        "symbol_name",
        "symbol_kind",
        "symbol_id",
        "qualified_name",
        "visibility",
        "signature",
        "docstring",
        "summary",
    }
)
# Fields that default to `""` rather than `None` but carry the same
# "unset" meaning (mirrors the truthy-check convention `render_xml` uses).
_OMIT_WHEN_EMPTY = frozenset({"imports_context", "breadcrumbs"})


def render_json(bundle: ContextBundle, *, full: bool = False) -> str:
    """Render a ContextBundle as a JSON string.

    By default, chunk/symbol fields that are unset (`None`) or empty
    (`""`) are dropped to keep the bundle token-lean. Pass `full=True` to
    restore the unfiltered dump (all fields present, `None` included).
    """
    if full:
        return bundle.model_dump_json(indent=2)
    return json.dumps(minimal_dump(bundle), indent=2)


def minimal_dump(model: BaseModel, *, exclude: set[str] | None = None) -> JsonValue:
    """Dump `model` to a JSON-shaped structure with noise fields stripped.

    Shared by `render_json` and `archex.scout.render_scout`'s JSON branch
    so both surfaces apply the same minimal-by-default field selection.
    """
    return _strip_noise(model.model_dump(mode="json", exclude=exclude))


def _strip_noise(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _strip_noise(val)
            for key, val in value.items()
            if not (key in _OMIT_WHEN_NONE and val is None)
            and not (key in _OMIT_WHEN_EMPTY and val == "")
        }
    if isinstance(value, list):
        return [_strip_noise(item) for item in value]
    return value
