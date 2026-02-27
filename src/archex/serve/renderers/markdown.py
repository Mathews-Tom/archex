"""Markdown renderer: format ArchProfile and ContextBundle as human-readable Markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import ContextBundle


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

    return "\n".join(lines)
