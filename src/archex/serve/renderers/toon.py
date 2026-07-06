"""TOON renderer: serialize ContextBundle to Token-Oriented Object Notation.

Requires the optional ``toons`` package (Rust/PyO3, TOON spec v3.0):
``uv add 'archex[toon]'``. This module is only imported when
``--format toon`` is requested, so the core install is unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import toons

from archex.serve.renderers.json import minimal_dump

if TYPE_CHECKING:
    from archex.models import ContextBundle


def render_toon(bundle: ContextBundle, *, full: bool = False) -> str:
    """Render a ContextBundle as a TOON string.

    Reuses `render_json`'s minimal-by-default field selection (M1): by
    default, chunk/symbol fields that are unset (`None`) or empty (`""`)
    are dropped to keep the bundle token-lean. Pass `full=True` to
    restore the unfiltered dump (all fields present, `None` included).
    """
    if full:
        return toons.dumps(bundle.model_dump(mode="json"))
    return toons.dumps(minimal_dump(bundle))
