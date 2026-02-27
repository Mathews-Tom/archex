"""XML renderer: serialize ArchProfile and ContextBundle to XML for LLM context windows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from archex.models import ContextBundle


def render_xml(bundle: ContextBundle) -> str:
    """Render a ContextBundle as an XML string suitable for LLM consumption."""
    lines: list[str] = []
    lines.append(f"<context query={_attr(bundle.query)}>")

    # Structural context
    sc = bundle.structural_context
    lines.append("  <structural-context>")
    if sc.file_tree:
        lines.append(f"    <file-tree><![CDATA[\n{sc.file_tree}\n    ]]></file-tree>")
    lines.append("  </structural-context>")

    # Chunks
    lines.append("  <chunks>")
    for rc in bundle.chunks:
        chunk = rc.chunk
        attrs = (
            f" file={_attr(chunk.file_path)} lines={_attr(f'{chunk.start_line}-{chunk.end_line}')}"
        )
        if chunk.symbol_name:
            attrs += f" symbol={_attr(chunk.symbol_name)}"
        attrs += f" score={_attr(f'{rc.final_score:.4f}')}"
        attrs += f" tokens={_attr(str(chunk.token_count))}"
        lines.append(f"    <chunk{attrs}>")
        if chunk.imports_context:
            lines.append(f"      <imports><![CDATA[{chunk.imports_context}]]></imports>")
        lines.append(f"      <code><![CDATA[\n{chunk.content}\n      ]]></code>")
        lines.append("    </chunk>")
    lines.append("  </chunks>")

    # Type definitions
    if bundle.type_definitions:
        lines.append("  <type-definitions>")
        for td in bundle.type_definitions:
            td_attrs = (
                f" file={_attr(td.file_path)}"
                f" symbol={_attr(td.symbol)}"
                f" lines={_attr(f'{td.start_line}-{td.end_line}')}"
            )
            lines.append(f"    <type-def{td_attrs}>")
            lines.append(f"      <![CDATA[{td.content}]]>")
            lines.append("    </type-def>")
        lines.append("  </type-definitions>")

    # Dependencies
    dep = bundle.dependency_summary
    if dep.internal or dep.external:
        lines.append("  <dependencies>")
        for item in dep.internal:
            lines.append(f"    <internal>{escape(item)}</internal>")
        for item in dep.external:
            lines.append(f"    <external>{escape(item)}</external>")
        lines.append("  </dependencies>")

    lines.append("</context>")
    return "\n".join(lines)


def _attr(value: str) -> str:
    """Return a double-quoted XML attribute value with escaping."""
    return f'"{escape(value, {chr(34): "&quot;"})}"'
