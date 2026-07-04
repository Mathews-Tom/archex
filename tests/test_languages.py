from __future__ import annotations

import pytest

from archex.languages import (
    LANGUAGE_SUPPORT,
    LanguageSupport,
    get_language_tier,
)
from archex.models import LanguageTier


def test_stub_registered_structured_language_reports_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = LanguageSupport(
        language_id="structured_stub",
        display_name="Structured Stub",
        extensions=(".stub",),
        tier=LanguageTier.STRUCTURED,
        pack_name="json",
        chunk_node_types=frozenset({"document"}),
    )
    monkeypatch.setitem(LANGUAGE_SUPPORT, "structured_stub", support)

    assert get_language_tier("structured_stub") == LanguageTier.STRUCTURED
    chunk_only_ids = {
        language_id
        for language_id, registered_support in LANGUAGE_SUPPORT.items()
        if registered_support.tier == LanguageTier.CHUNK_ONLY
    }
    assert "structured_stub" not in chunk_only_ids


def test_unknown_language_still_reports_unknown() -> None:
    assert get_language_tier("missing-language") == LanguageTier.UNKNOWN


def test_structured_language_requires_outline_chunk_nodes() -> None:
    with pytest.raises(ValueError, match="STRUCTURED languages must declare"):
        LanguageSupport(
            language_id="structured_stub",
            display_name="Structured Stub",
            extensions=(".stub",),
            tier=LanguageTier.STRUCTURED,
            pack_name="json",
        )
