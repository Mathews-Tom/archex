from __future__ import annotations

import json
import tempfile
import tomllib
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from archex.cli.main import cli
from archex.project import init_project


class FakeStore:
    db_path = Path("/tmp/archex-index.db")
    ephemeral = False

    def get_file_metadata(self) -> list[dict[str, str | int]]:
        return []

    def get_metadata(self, key: str) -> str | None:
        if key == "commit_hash":
            return "abc123"
        return None

    def get_file_count(self) -> int:
        return 0

    def get_chunk_count(self) -> int:
        return 0

    def close(self) -> None:
        pass


def test_index_quantize_flags_persist_project_settings(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    runner = CliRunner()

    with patch("archex.cli.indexing.index_repository", return_value=FakeStore()):
        result = runner.invoke(
            cli,
            [
                "index",
                str(python_simple_repo),
                "--quantize-vectors",
                "--quantize-bits",
                "2",
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["commit_hash"] == "abc123"

    settings = tomllib.loads((python_simple_repo / ".archex" / "settings.toml").read_text())
    assert settings["index"]["vector"] is True
    assert settings["index"]["embedder"] == "jina-v2"
    assert settings["index"]["quantize_vectors"] is True
    assert settings["index"]["quantize_bits"] == 2


def test_index_no_quantize_vectors_persists_false(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    runner = CliRunner()

    with patch("archex.cli.indexing.index_repository", return_value=FakeStore()):
        result = runner.invoke(
            cli,
            [
                "index",
                str(python_simple_repo),
                "--no-quantize-vectors",
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0, result.output
    settings = tomllib.loads((python_simple_repo / ".archex" / "settings.toml").read_text())
    assert settings["index"]["quantize_vectors"] is False


def test_index_ephemeral_no_cache_cleans_up_scratch_dir(python_simple_repo: Path) -> None:
    """`archex index` with caching disabled must not leak its scratch tempdir and
    must not report an `index_path` that no longer exists once the command returns.
    """
    init_project(python_simple_repo)
    settings_path = python_simple_repo / ".archex" / "settings.toml"
    content = settings_path.read_text(encoding="utf-8")
    settings_path.write_text(
        content.replace("[index]\n", "[index]\ncache = false\n", 1), encoding="utf-8"
    )

    created_dirs: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp() -> str:
        created = real_mkdtemp()
        created_dirs.append(Path(created))
        return created

    runner = CliRunner()
    with patch("archex.api.tempfile.mkdtemp", side_effect=_tracking_mkdtemp):
        result = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["index_path"] is None

    assert created_dirs, "expected the ephemeral full-index path to allocate a scratch dir"
    for scratch_dir in created_dirs:
        assert not scratch_dir.exists(), f"leaked ephemeral scratch dir: {scratch_dir}"


def test_index_ephemeral_no_cache_text_output_is_clear(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    settings_path = python_simple_repo / ".archex" / "settings.toml"
    content = settings_path.read_text(encoding="utf-8")
    settings_path.write_text(
        content.replace("[index]\n", "[index]\ncache = false\n", 1), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["index", str(python_simple_repo)])

    assert result.exit_code == 0, result.output
    assert "Index path:         (ephemeral, not cached to disk)" in result.output
