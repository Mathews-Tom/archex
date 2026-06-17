"""Markdown renderer: format ArchProfile and ContextBundle as human-readable Markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import ContextBundle, ContextReceipt


def render_markdown(bundle: ContextBundle) -> str:
    """Render a ContextBundle as a Markdown string."""
    lines: list[str] = []
    lines.append(f"# Context: {bundle.query}")
    lines.append("")

    # File tree
    sc = bundle.structural_context
    if sc.file_tree:
        lines.append("## File Tree")
        lines.append("")
        lines.append("```")
        lines.append(sc.file_tree)
        lines.append("```")
        lines.append("")

    # Chunks
    total_tokens = bundle.token_count
    chunk_count = len(bundle.chunks)
    lines.append(f"## Chunks ({chunk_count} results, {total_tokens} tokens)")
    lines.append("")
    for rc in bundle.chunks:
        chunk = rc.chunk
        header = f"{chunk.file_path}"
        if chunk.symbol_name:
            header += f":{chunk.symbol_name}"
        header += f" (score: {rc.final_score:.2f})"
        lines.append(f"### {header}")
        lang = chunk.language or ""
        lines.append(f"```{lang}")
        lines.append(chunk.content)
        lines.append("```")
        lines.append("")

    # Type definitions
    if bundle.type_definitions:
        lines.append("## Type Definitions")
        lines.append("")
        for td in bundle.type_definitions:
            lines.append(f"### {td.symbol} ({td.file_path}:{td.start_line}-{td.end_line})")
            lines.append("```")
            lines.append(td.content)
            lines.append("```")
            lines.append("")

    # Dependencies
    dep = bundle.dependency_summary
    if dep.internal or dep.external:
        lines.append("## Dependencies")
        lines.append("")
        if dep.internal:
            lines.append("### Internal")
            for item in dep.internal:
                lines.append(f"- {item}")
            lines.append("")
        if dep.external:
            lines.append("### External")
            for item in dep.external:
                lines.append(f"- {item}")
            lines.append("")

    if bundle.receipt is not None:
        lines.extend(_receipt_lines(bundle.receipt))
        lines.append("")

    return "\n".join(lines)


def _receipt_lines(receipt: ContextReceipt) -> list[str]:
    lines = [
        "## Receipt",
        "",
        f"- Freshness: {receipt.freshness.value}",
        f"- Index revision: {receipt.index_revision}",
        (
            f"- Budget: {receipt.token_budget.consumed} / "
            f"{receipt.token_budget.requested} tokens"
        ),
        f"- Context complete: {receipt.context_complete.value}",
        f"- Reason: {receipt.context_complete_reason.value}",
        f"- Recommended action: {receipt.recommended_next_action.value}",
        f"- Returned: {len(receipt.returned_context)} shown / {receipt.returned_total} total",
        f"- Skipped: {len(receipt.skipped_candidates)} shown / {receipt.skipped_total} total",
        (
            f"- Omitted dependency edges: {len(receipt.omitted_edges)} shown / "
            f"{receipt.omitted_edges_total} total"
        ),
    ]
    if receipt.skipped_candidates:
        lines.extend(["", "### Skipped candidates"])
        for item in receipt.skipped_candidates[:8]:
            handle = f" `{item.handle}`" if item.handle else ""
            symbol = f" symbol={item.symbol}" if item.symbol else ""
            detail = f" ({item.detail})" if item.detail else ""
            lines.append(
                f"- {item.file_path or '(index)'}{handle}: {item.reason.value}, "
                f"score={item.score:.3f}{symbol}{detail}"
            )
    if receipt.omitted_edges:
        lines.extend(["", "### Omitted dependency edges"])
        for edge in receipt.omitted_edges[:8]:
            reason = edge.reason.value if edge.reason else "unknown"
            lines.append(
                f"- {edge.source} --{edge.kind.value}--> {edge.target}: "
                f"{reason}, confidence={edge.confidence_score or 0.0:.3f}"
            )
    return lines
