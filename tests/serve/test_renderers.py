"""Tests for markdown and XML renderers covering type definitions, dependencies, and edge cases."""

from __future__ import annotations

from archex.models import (
    CodeChunk,
    ContextBundle,
    DependencySummary,
    RankedChunk,
    StructuralContext,
    SymbolKind,
    TypeDefinition,
)
from archex.serve.renderers.markdown import render_markdown
from archex.serve.renderers.xml import render_xml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str = "c1",
    file_path: str = "src/app.py",
    content: str = "def run(): pass",
    symbol_name: str | None = None,
    imports_context: str = "",
    token_count: int = 10,
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=5,
        symbol_name=symbol_name,
        symbol_kind=SymbolKind.FUNCTION if symbol_name else None,
        language="python",
        imports_context=imports_context,
        token_count=token_count,
    )


def _ranked(chunk: CodeChunk, score: float = 0.75) -> RankedChunk:
    return RankedChunk(chunk=chunk, relevance_score=score, final_score=score)


def _type_def(
    symbol: str = "User",
    file_path: str = "src/models.py",
    content: str = "class User: ...",
    start_line: int = 1,
    end_line: int = 3,
) -> TypeDefinition:
    return TypeDefinition(
        symbol=symbol,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
    )


def _base_bundle(**overrides: object) -> ContextBundle:
    defaults: dict[str, object] = {
        "query": "how does auth work?",
        "chunks": [_ranked(_chunk())],
        "structural_context": StructuralContext(file_tree="src/app.py"),
        "token_count": 10,
        "token_budget": 1000,
    }
    defaults.update(overrides)
    return ContextBundle(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Markdown renderer tests
# ---------------------------------------------------------------------------


def test_markdown_type_definitions_section_rendered() -> None:
    td = _type_def(
        symbol="Config",
        file_path="src/config.py",
        content="class Config: ...",
        start_line=5,
        end_line=10,
    )
    bundle = _base_bundle(type_definitions=[td])
    md = render_markdown(bundle)
    assert "## Type Definitions" in md
    assert "### Config (src/config.py:5-10)" in md
    assert "class Config: ..." in md


def test_markdown_internal_deps_only() -> None:
    bundle = _base_bundle(
        dependency_summary=DependencySummary(internal=["src/auth.py", "src/models.py"], external=[])
    )
    md = render_markdown(bundle)
    assert "## Dependencies" in md
    assert "### Internal" in md
    assert "- src/auth.py" in md
    assert "- src/models.py" in md
    assert "### External" not in md


def test_markdown_external_deps_only() -> None:
    bundle = _base_bundle(
        dependency_summary=DependencySummary(internal=[], external=["requests", "pydantic"])
    )
    md = render_markdown(bundle)
    assert "## Dependencies" in md
    assert "### External" in md
    assert "- requests" in md
    assert "- pydantic" in md
    assert "### Internal" not in md


def test_markdown_both_internal_and_external_deps() -> None:
    bundle = _base_bundle(
        dependency_summary=DependencySummary(
            internal=["src/utils.py"],
            external=["httpx"],
        )
    )
    md = render_markdown(bundle)
    assert "### Internal" in md
    assert "- src/utils.py" in md
    assert "### External" in md
    assert "- httpx" in md


def test_markdown_no_file_tree_when_empty() -> None:
    bundle = _base_bundle(structural_context=StructuralContext(file_tree=""))
    md = render_markdown(bundle)
    assert "## File Tree" not in md
    assert "```\n```" not in md


# ---------------------------------------------------------------------------
# XML renderer tests
# ---------------------------------------------------------------------------


def test_xml_imports_context_cdata_rendered() -> None:
    chunk = _chunk(imports_context="import os\nimport sys")
    bundle = _base_bundle(chunks=[_ranked(chunk)])
    xml = render_xml(bundle)
    assert "<imports><![CDATA[import os\nimport sys]]></imports>" in xml


def test_xml_type_definitions_block_rendered() -> None:
    td = _type_def(
        symbol="Request",
        file_path="src/http.py",
        content="class Request: ...",
        start_line=2,
        end_line=8,
    )
    bundle = _base_bundle(type_definitions=[td])
    xml = render_xml(bundle)
    assert "<type-definitions>" in xml
    assert 'symbol="Request"' in xml
    assert 'file="src/http.py"' in xml
    assert 'lines="2-8"' in xml
    assert "<![CDATA[class Request: ...]]>" in xml
    assert "</type-definitions>" in xml


def test_xml_dependencies_block_rendered() -> None:
    bundle = _base_bundle(
        dependency_summary=DependencySummary(
            internal=["src/db.py"],
            external=["sqlalchemy"],
        )
    )
    xml = render_xml(bundle)
    assert "<dependencies>" in xml
    assert "<internal>src/db.py</internal>" in xml
    assert "<external>sqlalchemy</external>" in xml
    assert "</dependencies>" in xml


def test_xml_no_type_defs_or_deps_tags_when_empty() -> None:
    bundle = _base_bundle(
        type_definitions=[],
        dependency_summary=DependencySummary(internal=[], external=[]),
    )
    xml = render_xml(bundle)
    assert "<type-definitions>" not in xml
    assert "<dependencies>" not in xml
