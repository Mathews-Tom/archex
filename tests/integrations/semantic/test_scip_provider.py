"""Tests for the SCIP evidence provider."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from archex.integrations.semantic import scip_pb2
from archex.integrations.semantic.models import (
    ProviderAvailability,
    SemanticEdgeKind,
    SemanticProviderName,
)
from archex.integrations.semantic.scip_provider import ScipEvidenceProvider
from archex.models import ParsedFile

if TYPE_CHECKING:
    import pytest

_DEFINITION_BIT = scip_pb2.SymbolRole.Definition


def _write_index(path: Path, *, tool_version: str = "0.5.0") -> None:
    index = scip_pb2.Index()
    index.metadata.tool_info.name = "scip-python"
    index.metadata.tool_info.version = tool_version

    a_doc = index.documents.add()
    a_doc.relative_path = "a.py"
    a_doc.language = "python"
    a_def = a_doc.occurrences.add()
    a_def.symbol = "scip-python python . . a/foo()."
    a_def.symbol_roles = _DEFINITION_BIT
    a_def.single_line_range.line = 0
    a_def.single_line_range.start_character = 4
    a_def.single_line_range.end_character = 7

    b_doc = index.documents.add()
    b_doc.relative_path = "b.py"
    b_doc.language = "python"
    b_use = b_doc.occurrences.add()
    b_use.symbol = "scip-python python . . a/foo()."
    b_use.range.extend([2, 0, 3])

    base_sym = index.documents[0].symbols.add()
    base_sym.symbol = "scip-python python . . a/Base#"
    impl_sym = index.documents[1].symbols.add()
    impl_sym.symbol = "scip-python python . . b/Impl#"
    rel = impl_sym.relationships.add()
    rel.symbol = base_sym.symbol
    rel.is_implementation = True

    base_def = a_doc.occurrences.add()
    base_def.symbol = base_sym.symbol
    base_def.symbol_roles = _DEFINITION_BIT
    base_def.single_line_range.line = 10
    base_def.single_line_range.start_character = 6
    base_def.single_line_range.end_character = 10

    impl_def = b_doc.occurrences.add()
    impl_def.symbol = impl_sym.symbol
    impl_def.symbol_roles = _DEFINITION_BIT
    impl_def.single_line_range.line = 5
    impl_def.single_line_range.start_character = 6
    impl_def.single_line_range.end_character = 10

    path.write_bytes(index.SerializeToString())


def _parsed_files() -> list[ParsedFile]:
    return [
        ParsedFile(path="a.py", language="python"),
        ParsedFile(path="b.py", language="python"),
    ]


class TestProbe:
    def test_unavailable_when_index_missing(self, tmp_path: Path) -> None:
        provider = ScipEvidenceProvider(index_path="index.scip")
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert receipt.provider == SemanticProviderName.SCIP
        assert "no SCIP index found" in receipt.reason

    def test_unavailable_when_index_empty(self, tmp_path: Path) -> None:
        (tmp_path / "index.scip").write_bytes(b"")
        provider = ScipEvidenceProvider(index_path="index.scip")
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "empty" in receipt.reason

    def test_unavailable_when_index_path_escapes_repo_root(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.scip"
        outside.write_bytes(b"not empty")
        try:
            provider = ScipEvidenceProvider(index_path="../outside.scip")
            receipt = provider.probe(tmp_path)
            assert receipt.availability == ProviderAvailability.UNAVAILABLE
            assert "outside the repository root" in receipt.reason
        finally:
            outside.unlink(missing_ok=True)

    def test_available_when_index_present(self, tmp_path: Path) -> None:
        _write_index(tmp_path / "index.scip")
        provider = ScipEvidenceProvider(index_path="index.scip")
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.reason == ""


class TestCollect:
    def test_unavailable_index_produces_no_edges_and_reason(self, tmp_path: Path) -> None:
        provider = ScipEvidenceProvider(index_path="index.scip")
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert receipt.reason

    def test_corrupt_index_is_stale_not_invented(self, tmp_path: Path) -> None:
        (tmp_path / "index.scip").write_bytes(b"\xff\xfe\x00not-a-valid-index")
        provider = ScipEvidenceProvider(index_path="index.scip")
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.STALE
        assert "could not decode" in receipt.reason

    def test_low_coverage_index_is_stale(self, tmp_path: Path) -> None:
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-python"
        index.metadata.tool_info.version = "0.1.0"
        doc = index.documents.add()
        doc.relative_path = "unrelated/other.py"
        doc.language = "python"
        (tmp_path / "index.scip").write_bytes(index.SerializeToString())

        provider = ScipEvidenceProvider(index_path="index.scip")
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.STALE
        assert "stale" in receipt.reason

    def test_cross_file_definition_and_reference_edges(self, tmp_path: Path) -> None:
        _write_index(tmp_path / "index.scip")
        provider = ScipEvidenceProvider(index_path="index.scip")
        evidence, receipt = provider.collect(_parsed_files(), tmp_path)

        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.provider == SemanticProviderName.SCIP
        assert receipt.tool_name == "scip-python"
        assert receipt.tool_version == "0.5.0"
        assert receipt.evidence_count == len(evidence)
        assert receipt.evidence_count > 0

        definition_edges = [e for e in evidence if e.kind == SemanticEdgeKind.DEFINITION]
        reference_edges = [e for e in evidence if e.kind == SemanticEdgeKind.REFERENCE]
        assert len(definition_edges) == 1
        assert definition_edges[0].source.file_path == "b.py"
        assert definition_edges[0].source.line == 2
        assert definition_edges[0].target.file_path == "a.py"
        assert definition_edges[0].target.line == 0
        assert definition_edges[0].provider == SemanticProviderName.SCIP
        assert definition_edges[0].provider_version == "0.5.0"
        assert 0.0 < definition_edges[0].confidence <= 1.0

        assert len(reference_edges) == 1
        assert reference_edges[0].source.file_path == "a.py"
        assert reference_edges[0].target.file_path == "b.py"

    def test_implementation_edge_from_relationship(self, tmp_path: Path) -> None:
        _write_index(tmp_path / "index.scip")
        provider = ScipEvidenceProvider(index_path="index.scip")
        evidence, _receipt = provider.collect(_parsed_files(), tmp_path)

        impl_edges = [e for e in evidence if e.kind == SemanticEdgeKind.IMPLEMENTATION]
        assert len(impl_edges) == 1
        assert impl_edges[0].source.file_path == "b.py"
        assert impl_edges[0].source.line == 5
        assert impl_edges[0].target.file_path == "a.py"
        assert impl_edges[0].target.line == 10

    def test_same_file_occurrences_produce_no_edges(self, tmp_path: Path) -> None:
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-python"
        index.metadata.tool_info.version = "0.1.0"
        doc = index.documents.add()
        doc.relative_path = "a.py"
        doc.language = "python"
        definition = doc.occurrences.add()
        definition.symbol = "scip-python python . . a/foo()."
        definition.symbol_roles = _DEFINITION_BIT
        definition.single_line_range.line = 0
        usage = doc.occurrences.add()
        usage.symbol = "scip-python python . . a/foo()."
        usage.single_line_range.line = 5
        (tmp_path / "index.scip").write_bytes(index.SerializeToString())

        provider = ScipEvidenceProvider(index_path="index.scip")
        evidence, receipt = provider.collect([ParsedFile(path="a.py", language="python")], tmp_path)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.evidence_count == 0

    def test_missing_protobuf_runtime_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import archex.integrations.semantic.scip_provider as scip_provider_module

        monkeypatch.setattr(scip_provider_module, "_scip_runtime_available", False)
        provider = ScipEvidenceProvider(index_path="index.scip")
        receipt = provider.probe(tmp_path)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "protobuf runtime unavailable" in receipt.reason
