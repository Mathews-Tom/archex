"""Proves the explorer is artifact-only: no parsing, indexing, or new graph edges.

Two independent checks:
1. A static import-graph check -- no `archex.explorer` module imports the
   parsing/indexing/acquisition machinery at all, so there is no code path
   through which the explorer *could* re-analyze source.
2. A functional check -- the full load-render-serve pipeline succeeds
   against a bare artifact file (no git repository present) while
   `archex.api.index_repository` is patched to raise if called.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from archex import explorer
from archex.explorer.loader import load_explorer_data
from archex.explorer.render import render_diff_page, render_page
from archex.explorer.viewmodel import build_diff_view, build_manifest_view

_FORBIDDEN_IMPORT_PREFIXES = (
    "archex.api",
    "archex.pipeline",
    "archex.parse",
    "archex.acquire",
    "archex.index",
    "archex.serve",
    "tree_sitter",
)


def _module_paths() -> list[Path]:
    package_dir = Path(explorer.__file__).parent
    return sorted(package_dir.glob("*.py"))


def _imported_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _offenders(imported: set[str]) -> set[str]:
    return {
        name
        for name in imported
        if any(
            name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_IMPORT_PREFIXES
        )
    }


def test_explorer_modules_never_import_analyzer_machinery() -> None:
    for module_path in _module_paths():
        imported = _imported_names(module_path.read_text())
        offenders = _offenders(imported)
        assert not offenders, f"{module_path.name} imports analyzer machinery: {offenders}"


def _artifact_json(path: Path) -> Path:
    payload = {
        "schema_version": {"value": "1.0.0"},
        "archex_version": "0.22.0",
        "generated_at": "2026-07-24T00:00:00Z",
        "source_identity": "acme/widget",
        "source_root": "/repo",
        "source_revision": "deadbeef",
        "working_tree_fingerprint": "fp",
        "index_generation": "gen1",
        "index_schema_version": "1",
        "chunker_revision": "c1",
        "config_fingerprint": "cfg1",
        "diff": {
            "base_ref": "main",
            "changed_files": [{"path": "a.py", "status": "M", "handle": "file:a.py"}],
            "changed_files_total": 1,
        },
    }
    artifact_path = path / "artifact.json"
    artifact_path.write_text(json.dumps(payload))
    return artifact_path


def test_explorer_renders_without_touching_the_repository_indexer(tmp_path: Path) -> None:
    """No git repository, no index -- only the artifact file exists on disk."""
    artifact_path = _artifact_json(tmp_path)

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("explorer must never call index_repository")

    with patch("archex.api.index_repository", side_effect=_fail_if_called):
        data = load_explorer_data(artifact_path)
        manifest = build_manifest_view(data)
        diff_view = build_diff_view(data)
        html = render_diff_page(manifest, diff_view)

    assert "acme/widget" in html
    assert "a.py" in html


def test_explorer_shell_page_renders_from_artifact_alone(tmp_path: Path) -> None:
    artifact_path = _artifact_json(tmp_path)
    data = load_explorer_data(artifact_path)

    html = render_page("archex explorer", build_manifest_view(data), "<h2>Views</h2>")

    assert html.startswith("<!DOCTYPE html>")
    assert "acme/widget" in html


@pytest.mark.parametrize("forbidden", ["import tree_sitter", "from archex.pipeline import x"])
def test_forbidden_import_detector_actually_detects(forbidden: str) -> None:
    """Meta-test: confirms the detector used above is not a silent no-op."""
    imported = _imported_names(forbidden)
    assert _offenders(imported)
