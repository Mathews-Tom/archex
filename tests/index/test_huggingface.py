"""Tests for Hugging Face model resolution helpers."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from archex.exceptions import ArchexIndexError
from archex.index.huggingface import resolve_hf_model_path

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_hf_model_path_returns_existing_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert resolve_hf_model_path(str(model_dir)) == str(model_dir)


def test_resolve_hf_model_path_prefers_cached_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cache/model"

    module = ModuleType("huggingface_hub")
    module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    assert resolve_hf_model_path("owner/model", revision="abc123") == "/cache/model"
    assert calls == [
        {
            "repo_id": "owner/model",
            "revision": "abc123",
            "local_files_only": True,
        }
    ]


def test_resolve_hf_model_path_downloads_when_cache_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs["local_files_only"] is True:
            raise RuntimeError("cache miss")
        return "/downloaded/model"

    module = ModuleType("huggingface_hub")
    module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    assert resolve_hf_model_path("owner/model") == "/downloaded/model"
    assert [call["local_files_only"] for call in calls] == [True, False]


def test_resolve_hf_model_path_fails_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        del kwargs
        raise RuntimeError("dns failed")

    module = ModuleType("huggingface_hub")
    module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(ArchexIndexError, match="Cache it locally first"):
        resolve_hf_model_path("owner/model", revision="abc123")
