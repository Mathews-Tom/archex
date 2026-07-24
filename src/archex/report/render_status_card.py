"""Markdown projection of a StatusCard (M9).

Projects ``StatusCard`` semantics without adding any -- in particular,
never computes or displays a composite score, letter grade, or health
rating across dimensions. Suitable for pasting into a project's own
README as a "Status" section (never auto-injected; that decision and edit
stay with the project's own maintainers).
"""

from __future__ import annotations

from archex.report.status_card import StatusCard, StatusDimension, StatusDimensionState


def _render_dimension(dimension: StatusDimension) -> list[str]:
    label = "Evidenced" if dimension.state == StatusDimensionState.EVIDENCED else "Unknown"
    lines = [f"### {dimension.name}", "", f"**{label}** — {dimension.detail}", ""]
    if dimension.evidence:
        lines.append("Evidence:")
        lines.extend(f"- `{item.location}` — {item.description}" for item in dimension.evidence)
        lines.append("")
    return lines


def render_status_card_markdown(card: StatusCard) -> str:
    lines: list[str] = [
        f"# Documentation & Release Status: {card.source_identity}",
        "",
        f"Generated at `{card.generated_at}` for revision `{card.revision[:12] or 'unknown'}`.",
        "",
        "Every dimension below is independent and links to immutable local "
        "evidence. There is no composite score or letter grade — evaluate "
        "each dimension on its own.",
        "",
    ]
    for dimension in card.dimensions:
        lines.extend(_render_dimension(dimension))
    return "\n".join(lines).rstrip() + "\n"
