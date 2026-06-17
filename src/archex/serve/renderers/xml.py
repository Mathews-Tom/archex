"""XML renderer: serialize ArchProfile and ContextBundle to XML for LLM context windows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from archex.models import ContextBundle, ContextReceipt


def render_xml(bundle: ContextBundle, *, include_receipt: bool = True) -> str:
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

    if include_receipt and bundle.receipt is not None:
        lines.extend(_render_receipt_xml(bundle.receipt))

    lines.append("</context>")
    return "\n".join(lines)


def render_xml_envelope(bundle: ContextBundle) -> str:
    """Render only the XML wrapper structure for token-overhead accounting."""
    lines: list[str] = []
    lines.append('<context query="">')

    sc = bundle.structural_context
    lines.append("  <structural-context>")
    if sc.file_tree:
        lines.append("    <file-tree><![CDATA[\n\n    ]]></file-tree>")
    lines.append("  </structural-context>")

    lines.append("  <chunks>")
    for rc in bundle.chunks:
        chunk = rc.chunk
        attrs = ' file="" lines=""'
        if chunk.symbol_name:
            attrs += ' symbol=""'
        attrs += ' score=""'
        attrs += ' tokens=""'
        lines.append(f"    <chunk{attrs}>")
        if chunk.imports_context:
            lines.append("      <imports><![CDATA[]]></imports>")
        lines.append("      <code><![CDATA[\n\n      ]]></code>")
        lines.append("    </chunk>")
    lines.append("  </chunks>")

    if bundle.type_definitions:
        lines.append("  <type-definitions>")
        for _ in bundle.type_definitions:
            lines.append('    <type-def file="" symbol="" lines="">')
            lines.append("      <![CDATA[]]>")
            lines.append("    </type-def>")
        lines.append("  </type-definitions>")

    dep = bundle.dependency_summary
    if dep.internal or dep.external:
        lines.append("  <dependencies>")
        for _ in dep.internal:
            lines.append("    <internal></internal>")
        for _ in dep.external:
            lines.append("    <external></external>")
        lines.append("  </dependencies>")

    lines.append("</context>")
    return "\n".join(lines)


def _render_receipt_xml(receipt: ContextReceipt) -> list[str]:
    lines = [
        "  <receipt"
        f" freshness={_attr(receipt.freshness.value)}"
        f" complete={_attr(receipt.context_complete.value)}"
        f" reason={_attr(receipt.context_complete_reason.value)}"
        f" action={_attr(receipt.recommended_next_action.value)}"
        f" index_revision={_attr(receipt.index_revision)}"
        f" budget_requested={_attr(str(receipt.token_budget.requested))}"
        f" budget_consumed={_attr(str(receipt.token_budget.consumed))}"
        f" returned_shown={_attr(str(len(receipt.returned_context)))}"
        f" returned_total={_attr(str(receipt.returned_total))}"
        f" skipped_shown={_attr(str(len(receipt.skipped_candidates)))}"
        f" skipped_total={_attr(str(receipt.skipped_total))}"
        f" omitted_edges_shown={_attr(str(len(receipt.omitted_edges)))}"
        f" omitted_edges_total={_attr(str(receipt.omitted_edges_total))}"
        ">",
    ]
    for item in receipt.returned_context[:8]:
        lines.append(
            "    <returned"
            f" handle={_attr(item.handle)}"
            f" file={_attr(item.file_path)}"
            f" lines={_attr(f'{item.start_line}-{item.end_line}')}"
            f" hash={_attr(item.content_hash)}"
            f" score={_attr(f'{item.score:.4f}')}"
            " />"
        )
    for item in receipt.skipped_candidates[:8]:
        lines.append(
            "    <skipped"
            f" file={_attr(item.file_path)}"
            f" reason={_attr(item.reason.value)}"
            f" handle={_attr(item.handle or '')}"
            " />"
        )
    lines.append("  </receipt>")
    return lines


def _attr(value: str) -> str:
    """Return a double-quoted XML attribute value with escaping."""
    return f'"{escape(value, {chr(34): "&quot;"})}"'
