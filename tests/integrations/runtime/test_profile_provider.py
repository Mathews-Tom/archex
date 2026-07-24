"""Tests for the runtime-profile evidence provider."""

from __future__ import annotations

import json
from pathlib import Path

from archex.integrations.runtime.models import ProviderAvailability
from archex.integrations.runtime.profile_provider import RuntimeProfileProvider

_REVISION = "a" * 40
_OTHER_REVISION = "b" * 40

_FOLDED_STACK = (
    "src/a.py:outer;src/b.py:inner 3\n"
    "src/a.py:outer 5\n"
    "\n"
    "not a valid frame line\n"
    "src/a.py:outer;../outside.py:escape 1\n"
)


def _write_evidence(
    tmp_path: Path,
    *,
    revision: str = _REVISION,
    tool: str = "cProfile",
    stack: str = _FOLDED_STACK,
) -> Path:
    evidence_dir = tmp_path / ".archex" / "runtime-evidence" / "profile"
    evidence_dir.mkdir(parents=True)
    manifest = {"revision": revision, "tool": tool, "tool_version": "3.12"}
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
    (evidence_dir / "profile.folded").write_text(stack)
    return tmp_path


class TestRuntimeProfileProviderProbe:
    def test_unavailable_when_no_evidence(self, tmp_path: Path) -> None:
        provider = RuntimeProfileProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no runtime-profile evidence" in receipt.reason

    def test_stale_on_revision_mismatch(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, revision=_OTHER_REVISION)
        provider = RuntimeProfileProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.STALE
        assert receipt.observed_revision == _OTHER_REVISION

    def test_available_on_revision_match(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = RuntimeProfileProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.tool_name == "cProfile"
        assert receipt.tool_version == "3.12"


class TestRuntimeProfileProviderCollect:
    def test_collect_returns_unavailable_and_empty_when_missing(self, tmp_path: Path) -> None:
        provider = RuntimeProfileProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_collect_returns_stale_and_empty_on_mismatch(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, revision=_OTHER_REVISION)
        provider = RuntimeProfileProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.STALE

    def test_collect_parses_valid_samples_and_drops_malformed_lines(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = RuntimeProfileProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert len(evidence) == 1
        profile = evidence[0]
        # 2 valid samples parsed; "not a valid frame line" and the
        # outside-repo-path sample are dropped rather than guessed at.
        assert len(profile.samples) == 2
        assert profile.total_samples == 8
        assert profile.revision == _REVISION
        assert "2 malformed frame line(s) dropped" in receipt.reason
        assert receipt.records_collected == 2

    def test_collect_normalizes_frame_paths(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, stack="src/a.py:outer;src/b.py:inner 1\n")
        provider = RuntimeProfileProvider()
        evidence, _receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence[0].samples[0].frames == ("src/a.py:outer", "src/b.py:inner")

    def test_collect_drops_sample_with_zero_count(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, stack="src/a.py:outer 0\n")
        provider = RuntimeProfileProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence[0].samples == []
        assert receipt.records_collected == 0
