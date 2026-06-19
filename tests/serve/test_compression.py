"""Tests for deterministic post-retrieval compression primitives."""

from __future__ import annotations

from archex.models import CompressionLossRisk, CompressionMode
from archex.reporting import count_tokens
from archex.serve.compression import compress_region

_HANDLE = "chunk:abc123def456"


def _code_region(body_lines: int = 40) -> str:
    body = "\n".join(f"    accumulator = accumulator + value_{i}" for i in range(body_lines))
    header = "import math\n\n\ndef compute(values):\n    accumulator = 0\n"
    return f"{header}{body}\n    return accumulator"


def _literal_region(items: int = 30) -> str:
    rows = "\n".join(f'    "color_{i}",' for i in range(items))
    return f"PALETTE = (\n{rows}\n)"


def _log_region(lines: int = 24) -> str:
    rows = [f"2026-06-18T10:00:{i:02d} INFO processed record {i}" for i in range(lines)]
    rows[lines // 2] = "2026-06-18T10:00:12 ERROR boom: write failure on record 12"
    return "\n".join(rows)


def test_required_region_passes_through_unchanged() -> None:
    region = _code_region()
    result = compress_region(
        region, language="python", fetch_original_handle=_HANDLE, required=True
    )
    assert result is not None
    assert result.mode == CompressionMode.PASSTHROUGH_REQUIRED
    assert result.content == region
    assert result.loss_risk == CompressionLossRisk.NONE


def test_structural_code_elision_keeps_signatures_and_marks_handle() -> None:
    region = _code_region()
    result = compress_region(
        region, language="python", fetch_original_handle=_HANDLE, required=False
    )
    assert result is not None
    assert result.mode == CompressionMode.STRUCTURAL_CODE_ELISION
    assert result.loss_risk == CompressionLossRisk.LOW
    # Imports and the signature survive; bodies become a marked, fetchable stub.
    assert "import math" in result.content
    assert "def compute(values):" in result.content
    assert "[archex" in result.content
    assert f"fetch original: {_HANDLE}" in result.content
    assert count_tokens(result.content) < count_tokens(region)


def test_large_literal_summary_keeps_first_last_and_count() -> None:
    region = _literal_region(items=30)
    result = compress_region(
        region, language="text", fetch_original_handle=_HANDLE, required=False
    )
    assert result is not None
    assert result.mode == CompressionMode.LARGE_LITERAL_SUMMARIZATION
    assert result.loss_risk == CompressionLossRisk.MEDIUM
    assert '"color_0"' in result.content  # first example preserved
    assert '"color_29"' in result.content  # last example preserved
    assert "item(s) omitted" in result.content
    assert f"fetch original: {_HANDLE}" in result.content
    assert count_tokens(result.content) < count_tokens(region)


def test_json_log_crushing_preserves_anomaly_lines() -> None:
    region = _log_region()
    result = compress_region(
        region, language="log", fetch_original_handle=_HANDLE, required=False
    )
    assert result is not None
    assert result.mode == CompressionMode.JSON_LOG_SMART_CRUSHING
    # The error line must survive even though it sits in the omitted middle band.
    assert "ERROR boom: write failure on record 12" in result.content
    assert "line(s) omitted" in result.content
    assert f"fetch original: {_HANDLE}" in result.content
    assert count_tokens(result.content) < count_tokens(region)


def test_json_object_routes_to_json_log_crushing() -> None:
    rows = "\n".join(f'  "key_{i}": {i},' for i in range(30))
    region = "{\n" + rows + "\n}"
    result = compress_region(
        region, language="json", fetch_original_handle=_HANDLE, required=False
    )
    assert result is not None
    assert result.mode == CompressionMode.JSON_LOG_SMART_CRUSHING


def test_comment_and_whitespace_slimming_drops_banners() -> None:
    region = (
        "# =========================\n"
        "value_a = 1\n"
        "\n"
        "\n"
        "\n"
        "value_b = 2\n"
        "# -------------------------\n"
        "value_c = 3\n"
    )
    result = compress_region(
        region, language="text", fetch_original_handle=_HANDLE, required=False
    )
    assert result is not None
    assert result.mode == CompressionMode.COMMENT_AND_WHITESPACE_SLIMMING
    assert result.loss_risk == CompressionLossRisk.LOW
    assert "=========================" not in result.content
    assert "-------------------------" not in result.content
    assert "\n\n\n" not in result.content  # repeated blanks collapsed
    assert "value_a = 1" in result.content
    assert "value_c = 3" in result.content


def test_compression_larger_than_original_falls_back_to_none() -> None:
    # Two trivial statements: any elision marker (with a handle) costs more tokens
    # than it saves, and nothing else applies, so the region stays uncompressed.
    region = "x = 1\ny = 2"
    assert (
        compress_region(
            region, language="python", fetch_original_handle=_HANDLE, required=False
        )
        is None
    )


def test_protect_code_disables_structural_elision() -> None:
    region = "# ======\n" + _code_region()
    result = compress_region(
        region,
        language="python",
        fetch_original_handle=_HANDLE,
        required=False,
        protect_code=True,
    )
    assert result is not None
    # Elision is off for fix/debug/review; only low-risk slimming may run.
    assert result.mode != CompressionMode.STRUCTURAL_CODE_ELISION


def test_compression_is_deterministic() -> None:
    region = _code_region()
    first = compress_region(
        region, language="python", fetch_original_handle=_HANDLE, required=False
    )
    second = compress_region(
        region, language="python", fetch_original_handle=_HANDLE, required=False
    )
    assert first == second
