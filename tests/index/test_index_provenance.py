"""Tests that a project-layout index is only reused when its origin is provable.

`.archex/index.db` and `.archex/settings.toml` are ordinary files, so a
published repository can commit both. Everything a store says about itself —
its revision, its source identity, its index-config shape — lives inside that
same database, so it cannot be the evidence for trusting it. The cache marker
can: it is keyed on `sha256("<resolved absolute path>@<commit>")`, which
committed content cannot forge for a clone directory nobody knows in advance.

The scenario built here is the real one, with real `git`: a repository that
ships a poisoned index, cloned by a victim who then indexes or queries it.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.api import index_repository
from archex.cli.main import cli
from archex.config import load_config
from archex.index.artifact import export_artifact, import_artifact, sync_imported_artifact
from archex.index.store import IndexStore
from archex.integrations.hook import _lookup  # pyright: ignore[reportPrivateUsage]
from archex.models import Config, IndexConfig, PipelineTiming, RepoSource
from archex.project import init_project
from archex.status import inspect_project_status

POISON = "ATTACKER_CONTROLLED_PAYLOAD"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", message)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    for index in range(6):
        (root / f"mod_{index}.py").write_text(
            f"def value_{index}() -> int:\n    return {index}\n", encoding="utf-8"
        )
    _commit_all(root, "initial")
    return root


def _build_project_index(repo: Path) -> None:
    init_project(repo)
    store = index_repository(
        RepoSource(local_path=str(repo)),
        config=Config(cache=True, cache_dir=str(repo / ".archex")),
        index_config=IndexConfig(),
    )
    store.close()


def _poison_chunks(index_path: Path) -> int:
    """Rewrite chunk bodies and claim the identity a CLI invocation produces.

    `source_identity` is the raw source string a caller passed, and the CLI's
    default is `.`, so that is what an attacker plants: it matches whatever
    directory the victim happens to run in.
    """
    conn = sqlite3.connect(index_path)
    try:
        cursor = conn.execute(
            "UPDATE chunks SET content = ? WHERE file_path LIKE 'mod_%'", (POISON,)
        )
        conn.execute("UPDATE metadata SET value = '.' WHERE key = 'source_identity'")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


@pytest.fixture
def hostile_repo(tmp_path: Path) -> Path:
    """A repository that commits a poisoned `.archex` index of its own tree.

    The recorded revision is deliberately the *previous* commit, which is
    what an attacker can do: a database cannot contain the hash of the
    commit that adds it.
    """
    repo = _init_repo(tmp_path / "hostile")
    _build_project_index(repo)
    assert _poison_chunks(repo / ".archex" / "index.db") == 6
    (repo / ".gitignore").write_text("", encoding="utf-8")
    for name in (".archex/index.db", ".archex/settings.toml", ".gitignore"):
        _git(repo, "add", "-f", name)
    _commit_all(repo, "chore: add project config")
    return repo


class TestPlantedIndexIsNotReused:
    def test_cloned_poisoned_index_is_not_served(self, hostile_repo: Path, tmp_path: Path) -> None:
        """The committed index must not answer a query on the victim's clone."""
        victim = tmp_path / "victim"
        _git(tmp_path, "clone", "-q", str(hostile_repo), str(victim))
        assert (victim / ".archex" / "index.db").is_file()

        result = CliRunner().invoke(cli, ["query", str(victim), "value_3", "--format", "json"])

        assert result.exit_code == 0, result.output
        assert POISON not in result.output

    def test_cloned_poisoned_index_is_replaced_by_a_real_index(
        self, hostile_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        victim = tmp_path / "victim"
        _git(tmp_path, "clone", "-q", str(hostile_repo), str(victim))
        monkeypatch.chdir(victim)

        timing = PipelineTiming()
        store = index_repository(
            RepoSource(local_path="."),
            config=Config(cache=True, cache_dir=str(victim / ".archex")),
            index_config=IndexConfig(),
            timing=timing,
        )
        try:
            contents = [chunk.content for chunk in store.get_chunks()]
        finally:
            store.close()

        assert timing.strategy == "full"
        assert POISON not in contents

    def test_a_marker_from_another_directory_does_not_transfer(
        self, hostile_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Committing the producer's own marker alongside the index does not help."""
        marker = json.loads((hostile_repo / ".archex" / "index.meta").read_text(encoding="utf-8"))
        victim = tmp_path / "victim"
        _git(tmp_path, "clone", "-q", str(hostile_repo), str(victim))
        (victim / ".archex" / "index.meta").write_text(json.dumps(marker), encoding="utf-8")
        monkeypatch.chdir(victim)

        result = CliRunner().invoke(cli, ["query", "value_3", "--format", "json"])

        assert result.exit_code == 0, result.output
        assert POISON not in result.output

    def test_a_legitimately_built_index_is_still_reused(self, tmp_path: Path) -> None:
        """The check must not cost an honest project its warm cache."""
        repo = _init_repo(tmp_path / "honest")
        _build_project_index(repo)

        timing = PipelineTiming()
        store = index_repository(
            RepoSource(local_path=str(repo)),
            config=Config(cache=True, cache_dir=str(repo / ".archex")),
            index_config=IndexConfig(),
            timing=timing,
        )
        store.close()

        assert timing.strategy == "cached"

    def test_a_legitimate_index_is_reused_across_a_commit(self, tmp_path: Path) -> None:
        """A new commit changes the cache key, so reuse then rests on the marker."""
        repo = _init_repo(tmp_path / "honest")
        _build_project_index(repo)
        (repo / "added.py").write_text("def added() -> int:\n    return 9\n", encoding="utf-8")
        _commit_all(repo, "feat: add a module")

        timing = PipelineTiming()
        store = index_repository(
            RepoSource(local_path=str(repo)),
            config=Config(cache=True, cache_dir=str(repo / ".archex")),
            index_config=IndexConfig(),
            timing=timing,
        )
        store.close()

        assert timing.strategy == "delta"


class TestImportedArtifactIsAdopted:
    def test_imported_artifact_is_reused_by_the_next_index_run(self, tmp_path: Path) -> None:
        """An import that is not adopted would be discarded on the next command."""
        producer = _init_repo(tmp_path / "producer")
        _build_project_index(producer)
        artifact = tmp_path / "artifact.xz"
        store = IndexStore(producer / ".archex" / "index.db")
        try:
            export_artifact(store, artifact)
        finally:
            store.close()

        consumer = tmp_path / "consumer"
        _git(tmp_path, "clone", "-q", str(producer), str(consumer))
        init_project(consumer)
        (consumer / ".archex" / "index.db").unlink(missing_ok=True)
        import_artifact(artifact, consumer / ".archex" / "index.db")
        sync_imported_artifact(
            consumer,
            consumer / ".archex" / "index.db",
            load_config(str(consumer)),
            IndexConfig(),
            source_identity=str(consumer),
        )

        assert (consumer / ".archex" / "index.meta").is_file()
        timing = PipelineTiming()
        reused = index_repository(
            RepoSource(local_path=str(consumer)),
            config=Config(cache=True, cache_dir=str(consumer / ".archex")),
            index_config=IndexConfig(),
            timing=timing,
        )
        reused.close()

        assert timing.strategy == "cached"


class TestWorktreeSeedIsNotRepoConfigurable:
    def test_committed_settings_cannot_change_seeding(self, tmp_path: Path) -> None:
        """A repository must not switch the feature that consumes its content.

        The key is written into the repo-local settings explicitly here,
        because the generated template no longer contains it: a test that
        only relied on the template would assert nothing.
        """
        repo = _init_repo(tmp_path / "repo")
        init_project(repo)
        settings = repo / ".archex" / "settings.toml"
        original = settings.read_text(encoding="utf-8")
        assert "worktree_seed" not in original

        settings.write_text(
            original.replace("[index]", "[index]\nworktree_seed = false"), encoding="utf-8"
        )
        assert load_config(str(repo)).worktree_seed is True

        settings.write_text(
            original.replace("[index]", "[index]\nworktree_seed = true"), encoding="utf-8"
        )
        assert load_config(str(repo)).worktree_seed is True

    def test_user_config_and_environment_still_control_seeding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _init_repo(tmp_path / "repo")
        init_project(repo)
        monkeypatch.setenv("ARCHEX_WORKTREE_SEED", "0")

        assert load_config(str(repo)).worktree_seed is False


class TestSurfacesThatReadTheIndexDirectly:
    def test_status_refuses_an_unprovenanced_index(
        self, hostile_repo: Path, tmp_path: Path
    ) -> None:
        """`archex status` and everything built on it must not read a planted index.

        The tool-call hook gates on this inspection reporting `fresh`, and
        injects symbol rows straight into an agent's context, so a planted
        index reading as `fresh` would put attacker-chosen content in front
        of the agent.
        """
        victim = tmp_path / "victim"
        _git(tmp_path, "clone", "-q", str(hostile_repo), str(victim))

        status = inspect_project_status(victim)

        assert status.state == "unprovenanced"
        assert status.files_indexed == 0
        assert status.chunks_indexed == 0

    def test_status_accepts_a_legitimately_built_index(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path / "honest")
        _build_project_index(repo)

        assert inspect_project_status(repo).state == "fresh"

    def test_hook_lookup_declines_an_unprovenanced_index(
        self, hostile_repo: Path, tmp_path: Path
    ) -> None:
        victim = tmp_path / "victim"
        _git(tmp_path, "clone", "-q", str(hostile_repo), str(victim))

        assert _lookup(str(victim), "value_3") is None


class TestRelocatedCacheDirIsNotTrusted:
    def test_repo_committed_cache_dir_does_not_bypass_the_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`cache_dir` is repo-settable, so it cannot gate the marker requirement.

        A published repository can point `cache_dir` at any directory it
        also commits, which would take the store outside the project layout.
        The marker is therefore required for every cache-mediated reuse, not
        only for `.archex`.
        """
        repo = _init_repo(tmp_path / "relocated")
        init_project(repo)
        settings = repo / ".archex" / "settings.toml"
        settings.write_text(
            settings.read_text(encoding="utf-8").replace(
                'cache_dir = ".archex"', 'cache_dir = "vendor"'
            ),
            encoding="utf-8",
        )
        vendor = repo / "vendor"
        vendor.mkdir()
        store = index_repository(
            RepoSource(local_path=str(repo)),
            config=Config(cache=True, cache_dir=str(vendor)),
            index_config=IndexConfig(),
        )
        store.close()
        assert _poison_chunks(next(vendor.glob("*.db"))) == 6
        # Drop the marker the honest build wrote, leaving only what a
        # repository could have committed.
        for marker in vendor.glob("*.meta"):
            marker.unlink()
        monkeypatch.chdir(repo)

        timing = PipelineTiming()
        rebuilt = index_repository(
            RepoSource(local_path="."),
            config=load_config(str(repo)),
            index_config=IndexConfig(),
            timing=timing,
        )
        try:
            contents = [chunk.content for chunk in rebuilt.get_chunks()]
        finally:
            rebuilt.close()

        assert timing.strategy == "full"
        assert POISON not in contents
