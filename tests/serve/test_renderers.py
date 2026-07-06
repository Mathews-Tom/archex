"""Tests for markdown, XML, JSON, and TOON renderers covering type definitions,
dependencies, and edge cases."""

from __future__ import annotations

import toons

from archex.models import (
    CodeChunk,
    CompressionLossRisk,
    CompressionMetadata,
    CompressionMode,
    ContextBundle,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextOmittedEdgeReason,
    ContextReceipt,
    ContextReceiptEdge,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
    ContextSkippedCandidate,
    ContextSkippedReason,
    DependencySummary,
    EdgeKind,
    RankedChunk,
    StructuralContext,
    SymbolKind,
    TypeDefinition,
)
from archex.scout import chunk_handle
from archex.serve.renderers.json import render_json
from archex.serve.renderers.markdown import render_markdown
from archex.serve.renderers.toon import render_toon
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


def _receipt() -> ContextReceipt:
    return ContextReceipt(
        query="how does auth work?",
        token_budget=ContextReceiptTokenBudget(requested=1000, consumed=250),
        index_revision="rev",
        freshness=ContextFreshness.CLEAN,
        skipped_candidates=[
            ContextSkippedCandidate(
                file_path="src/extra.py",
                reason=ContextSkippedReason.BELOW_THRESHOLD,
                handle="file:src/extra.py",
                score=0.25,
            )
        ],
        omitted_edges=[
            ContextReceiptEdge(
                source="src/app.py",
                target="src/db.py",
                kind=EdgeKind.IMPORTS,
                reason=ContextOmittedEdgeReason.OVER_BUDGET,
                confidence_score=0.8,
            )
        ],
        returned_total=3,
        skipped_total=5,
        omitted_edges_total=4,
        context_complete=ContextCompletenessStatus.INCOMPLETE,
        context_complete_reason=ContextCompletenessReason.DEPENDENCY_FRONTIER_CUT,
        recommended_next_action=ContextRecommendedAction.FETCH_SKIPPED_CANDIDATE,
    )


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


def test_markdown_receipt_block_shows_totals_and_next_action() -> None:
    md = render_markdown(_base_bundle(receipt=_receipt()))
    assert "## Receipt" in md
    assert "- Budget: 250 / 1000 tokens" in md
    assert "- Recommended action: fetch_skipped_candidate" in md
    assert "- Returned: 0 shown / 3 total" in md
    assert "- Skipped: 1 shown / 5 total" in md
    assert "- Omitted dependency edges: 1 shown / 4 total" in md
    assert "src/extra.py" in md
    assert "src/app.py --imports--> src/db.py" in md


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


def test_xml_receipt_uses_shown_and_total_counts() -> None:
    xml = render_xml(_base_bundle(receipt=_receipt()))
    assert 'returned_shown="0"' in xml
    assert 'returned_total="3"' in xml
    assert 'skipped_shown="1"' in xml
    assert 'skipped_total="5"' in xml
    assert 'omitted_edges_shown="1"' in xml
    assert 'omitted_edges_total="4"' in xml


def test_xml_no_type_defs_or_deps_tags_when_empty() -> None:
    bundle = _base_bundle(
        type_definitions=[],
        dependency_summary=DependencySummary(internal=[], external=[]),
    )
    xml = render_xml(bundle)
    assert "<type-definitions>" not in xml
    assert "<dependencies>" not in xml


# ---------------------------------------------------------------------------
# Compression rendering
# ---------------------------------------------------------------------------


def _compressed_bundle() -> ContextBundle:
    chunk = CodeChunk(
        id="c1",
        content="def f():\n    # ... [archex elided 30 line(s); fetch original: chunk:c1]",
        file_path="src/app.py",
        start_line=1,
        end_line=40,
        language="python",
    )
    handle = chunk_handle("c1")
    item = ContextReceiptItem(
        handle=handle,
        file_path="src/app.py",
        start_line=1,
        end_line=40,
        content_hash="orig",
        compression=CompressionMetadata(
            compression_mode=CompressionMode.STRUCTURAL_CODE_ELISION,
            original_tokens=200,
            compressed_tokens=60,
            compression_ratio=0.3,
            original_content_hash="orig",
            compressed_content_hash="comp",
            fetch_original_handle=handle,
            compression_loss_risk=CompressionLossRisk.LOW,
        ),
    )
    receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=500, consumed=100),
        index_revision="rev",
        returned_context=[item],
        returned_total=1,
    )
    return ContextBundle(
        query="q",
        chunks=[RankedChunk(chunk=chunk, relevance_score=0.4, final_score=0.4)],
        token_count=60,
        token_budget=500,
        receipt=receipt,
    )


def test_markdown_marks_compressed_region_and_fetch_handle() -> None:
    md = render_markdown(_compressed_bundle())
    assert "Compressed (structural_code_elision" in md
    assert "fetch original" in md.lower()
    assert chunk_handle("c1") in md
    assert "### Compressed regions" in md
    assert "Compressed regions: 1 of 1" in md


def test_markdown_uncompressed_bundle_has_no_compression_markers() -> None:
    md = render_markdown(_base_bundle(receipt=_receipt()))
    assert "Compressed (" not in md
    assert "### Compressed regions" not in md


def test_json_exposes_compression_metadata() -> None:
    import json

    data = json.loads(render_json(_compressed_bundle()))
    item = data["receipt"]["returned_context"][0]
    assert item["compression"]["compression_mode"] == "structural_code_elision"
    assert item["compression"]["fetch_original_handle"] == chunk_handle("c1")
    assert item["compression"]["compression_loss_risk"] == "low"


def test_json_uncompressed_row_has_null_compression() -> None:
    import json

    receipt = _receipt().model_copy(
        update={
            "returned_context": [
                ContextReceiptItem(
                    handle=chunk_handle("c1"),
                    file_path="src/app.py",
                    start_line=1,
                    end_line=5,
                    content_hash="h",
                )
            ]
        }
    )
    data = json.loads(render_json(_base_bundle(receipt=receipt)))
    rows = data["receipt"]["returned_context"]
    assert rows  # non-empty so the assertion below actually runs
    for item in rows:
        assert item["compression"] is None


# ---------------------------------------------------------------------------
# JSON renderer: minimal-by-default field selection (M1)
# ---------------------------------------------------------------------------

_NONE_VALUED_CHUNK_KEYS = (
    "symbol_name",
    "symbol_kind",
    "symbol_id",
    "qualified_name",
    "visibility",
    "signature",
    "docstring",
    "summary",
)
_EMPTY_STRING_CHUNK_KEYS = ("imports_context", "breadcrumbs")


def _none_heavy_chunk() -> CodeChunk:
    return CodeChunk(
        id="c-none",
        content="def run(): pass",
        file_path="src/app.py",
        start_line=1,
        end_line=5,
        language="python",
        symbol_name=None,
        symbol_kind=None,
        symbol_id=None,
        qualified_name=None,
        visibility=None,
        signature=None,
        docstring=None,
        summary=None,
        imports_context="",
        breadcrumbs="",
    )


def test_json_default_omits_none_and_empty_chunk_fields() -> None:
    import json

    bundle = _base_bundle(chunks=[_ranked(_none_heavy_chunk())])
    data = json.loads(render_json(bundle))
    chunk = data["chunks"][0]["chunk"]
    for key in (*_NONE_VALUED_CHUNK_KEYS, *_EMPTY_STRING_CHUNK_KEYS):
        assert key not in chunk, f"{key} should be omitted from minimal output"


def test_json_full_restores_all_chunk_fields() -> None:
    import json

    bundle = _base_bundle(chunks=[_ranked(_none_heavy_chunk())])
    data = json.loads(render_json(bundle, full=True))
    chunk = data["chunks"][0]["chunk"]
    for key in _NONE_VALUED_CHUNK_KEYS:
        assert key in chunk
        assert chunk[key] is None
    for key in _EMPTY_STRING_CHUNK_KEYS:
        assert chunk[key] == ""


def test_json_default_keeps_populated_chunk_fields() -> None:
    import json

    populated = CodeChunk(
        id="c-populated",
        content="def run(): pass",
        file_path="src/app.py",
        start_line=1,
        end_line=5,
        language="python",
        symbol_name="run",
        symbol_kind=SymbolKind.FUNCTION,
        symbol_id="src/app.py::run#function",
        qualified_name="app.run",
        visibility="public",
        signature="def run() -> None",
        docstring="Run the app.",
        summary="Entry point.",
        imports_context="import os",
        breadcrumbs="app > run",
    )
    bundle = _base_bundle(chunks=[_ranked(populated)])
    data = json.loads(render_json(bundle))
    chunk = data["chunks"][0]["chunk"]
    assert chunk["symbol_name"] == "run"
    assert chunk["symbol_kind"] == "function"
    assert chunk["symbol_id"] == "src/app.py::run#function"
    assert chunk["qualified_name"] == "app.run"
    assert chunk["visibility"] == "public"
    assert chunk["signature"] == "def run() -> None"
    assert chunk["docstring"] == "Run the app."
    assert chunk["summary"] == "Entry point."
    assert chunk["imports_context"] == "import os"
    assert chunk["breadcrumbs"] == "app > run"


def test_json_zero_score_ranked_chunk_present_in_default_and_full() -> None:
    import json

    zero_ranked = RankedChunk(
        chunk=_none_heavy_chunk(),
        relevance_score=0.0,
        structural_score=0.0,
        type_coverage_score=0.0,
        cohesion_score=0.0,
        final_score=0.0,
    )
    bundle = _base_bundle(chunks=[zero_ranked])

    minimal = json.loads(render_json(bundle))["chunks"][0]
    full = json.loads(render_json(bundle, full=True))["chunks"][0]
    for payload in (minimal, full):
        assert payload["relevance_score"] == 0.0
        assert payload["structural_score"] == 0.0
        assert payload["type_coverage_score"] == 0.0
        assert payload["cohesion_score"] == 0.0
        assert payload["final_score"] == 0.0


def test_json_minimal_output_smaller_than_full_for_none_heavy_chunk() -> None:
    bundle = _base_bundle(chunks=[_ranked(_none_heavy_chunk())])
    assert len(render_json(bundle)) < len(render_json(bundle, full=True))


# ---------------------------------------------------------------------------
# TOON renderer: size comparison and round-trip coverage (M2)
# ---------------------------------------------------------------------------


def _toon_realistic_chunk(i: int, *, populated: bool) -> CodeChunk:
    """Build one chunk of a multi-chunk fixture: alternates between a
    populated symbol match (all fields set) and a markdown-paragraph-style
    match (mostly None/empty fields), mirroring the mixed shape a real
    archex query bundle returns.
    """
    if populated:
        return CodeChunk(
            id=f"src/service_{i}.py:handler_{i}:10",
            content=(
                f"def handler_{i}(request: Request) -> Response:\n"
                f'    """Handle inbound request {i}."""\n'
                "    return process(request)"
            ),
            file_path=f"src/service_{i}.py",
            start_line=10,
            end_line=13,
            symbol_name=f"handler_{i}",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            imports_context="import os\nfrom archex.models import Request, Response",
            token_count=42,
            symbol_id=f"src/service_{i}.py::handler_{i}#function",
            qualified_name=f"service_{i}.handler_{i}",
            visibility="public",
            signature=f"def handler_{i}(request: Request) -> Response",
            docstring=f"Handle inbound request {i}.",
            breadcrumbs=f"service_{i} > handler_{i}",
            summary=f"Request handler {i}.",
        )
    return CodeChunk(
        id=f"README.md:_module:{i}",
        content=f"Paragraph {i} describing the architecture in prose with no code symbols.",
        file_path="README.md",
        start_line=i,
        end_line=i + 2,
        language="markdown",
        imports_context="",
        token_count=18,
    )


def _toon_realistic_bundle(n: int = 15) -> ContextBundle:
    """A 15-chunk fixture mirroring the mixed populated/null shape measured
    against this repo's own bundles in `.docs/2026-07-06-axi-surface-analysis.md`.
    """
    chunks = [
        RankedChunk(
            chunk=_toon_realistic_chunk(i, populated=(i % 2 == 0)),
            relevance_score=round(0.9 - i * 0.03, 2),
            structural_score=round(0.4 - i * 0.01, 2) if i % 2 == 0 else 0.0,
            type_coverage_score=0.0,
            cohesion_score=0.2 if i % 3 == 0 else 0.0,
            final_score=round(0.85 - i * 0.03, 2),
        )
        for i in range(n)
    ]
    return ContextBundle(
        query="how does the request handler pipeline work?",
        chunks=chunks,
        structural_context=StructuralContext(
            file_tree="src/\n  service_0.py\n  service_1.py\nREADME.md"
        ),
        type_definitions=[
            TypeDefinition(
                symbol="Request",
                file_path="src/models.py",
                start_line=1,
                end_line=5,
                content="class Request: ...",
            ),
            TypeDefinition(
                symbol="Response",
                file_path="src/models.py",
                start_line=7,
                end_line=11,
                content="class Response: ...",
            ),
        ],
        dependency_summary=DependencySummary(internal=["src/models.py"], external=["httpx"]),
        token_count=sum(c.chunk.token_count for c in chunks),
        token_budget=4000,
    )


def test_toon_smaller_than_json_for_realistic_bundle() -> None:
    """Measured, non-fixed-threshold reduction (M2 acceptance #3): direction
    and existence of a byte reduction is the gate, not a specific percentage.
    On this 15-chunk fixture TOON measures ~17% smaller than JSON.
    """
    bundle = _toon_realistic_bundle()
    json_bytes = render_json(bundle).encode()
    toon_bytes = render_toon(bundle).encode()
    assert len(toon_bytes) < len(json_bytes)


def test_toon_round_trip_preserves_query_and_chunk_count() -> None:
    """TOON has no `json.loads`-equivalent schema validation, so decode the
    output back and check the content that matters survives the round trip.
    """
    bundle = _toon_realistic_bundle()
    decoded = toons.loads(render_toon(bundle))
    assert decoded["query"] == bundle.query
    assert len(decoded["chunks"]) == len(bundle.chunks)
    assert decoded["chunks"][0]["chunk"]["symbol_name"] == "handler_0"
    assert decoded["token_count"] == bundle.token_count
    assert decoded["token_budget"] == bundle.token_budget


def test_toon_default_omits_none_and_empty_chunk_fields() -> None:
    bundle = _base_bundle(chunks=[_ranked(_none_heavy_chunk())])
    decoded = toons.loads(render_toon(bundle))
    chunk = decoded["chunks"][0]["chunk"]
    for key in (*_NONE_VALUED_CHUNK_KEYS, *_EMPTY_STRING_CHUNK_KEYS):
        assert key not in chunk, f"{key} should be omitted from minimal TOON output"


def test_toon_full_restores_all_chunk_fields() -> None:
    bundle = _base_bundle(chunks=[_ranked(_none_heavy_chunk())])
    decoded = toons.loads(render_toon(bundle, full=True))
    chunk = decoded["chunks"][0]["chunk"]
    for key in _NONE_VALUED_CHUNK_KEYS:
        assert key in chunk
        assert chunk[key] is None
    for key in _EMPTY_STRING_CHUNK_KEYS:
        assert chunk[key] == ""


def test_render_toon_exported_from_renderers_package() -> None:
    """`render_toon` is reachable via the renderers package's lazy
    `__getattr__`, not just the `archex.serve.renderers.toon` submodule.
    """
    from archex.serve.renderers import render_toon as package_render_toon

    assert package_render_toon is render_toon
