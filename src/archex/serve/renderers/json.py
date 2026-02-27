"""JSON renderer: serialize ArchProfile and ContextBundle to structured JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import ContextBundle


def render_json(bundle: ContextBundle) -> str:
    """Render a ContextBundle as a JSON string."""
    return bundle.model_dump_json(indent=2)
