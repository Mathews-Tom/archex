"""Tests for same-repository worktree seed discovery against real Git shapes.

Every checkout shape here is created with real `git` commands rather than
mocked, because the whole point of the module under test is that the shapes
are distinguishable on disk: a linked worktree's `.git` is a file, a
submodule's and a `--separate-git-dir` checkout's Git directory equals the
repository common directory, and a stale worktree pointer makes
`git worktree list` name a checkout that belongs to another repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from archex import cache as cache_module
from archex.api import index_repository
from archex.cache import CacheManager
from archex.cli.main import cli
from archex.index import worktree_seed
from archex.index.store import CURRENT_SCHEMA_VERSION, IndexStore
from archex.index.worktree_seed import (
    SEED_LOCK_FILENAME,
    WorktreeSeedResult,
    resolve_checkout_identity,
    seed_worktree_index,
    select_worktree_seed,
)
from archex.models import Config, IndexConfig, PipelineTiming, RepoSource
from archex.project import init_project
from archex.state_file import ExclusiveLock, StateFileDiagnostics

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


#: Files a project-layout index covers beyond `_DEFAULT_FILES`: the
#: `.gitignore` entry `archex init` writes is untracked repository content
#: and is discovered like any other file.
_INIT_WRITTEN_FILES = 1

#: Default corpus for a fixture repository. Wide enough that changing one
#: file stays well below `Config.delta_threshold` (0.5), so a delta case in
#: these tests exercises the delta path rather than the staleness fallback.
_DEFAULT_FILES = {
    f"mod_{index}.py": f"def value_{index}() -> int:\n    return {index}\n" for index in range(8)
}


def _init_repo(root: Path, *, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@archex.test")
    _git(root, "config", "user.name", "archex-test")
    for name, body in (files or _DEFAULT_FILES).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "initial")
    return root


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _project_config(repo: Path) -> Config:
    """The config `archex index` itself loads for a project-layout repository.

    Notably `languages` stays unset, matching `.archex/settings.toml`'s
    `languages = []`, so a fixture-built index covers the same file set the
    CLI would discover — otherwise every seeded comparison would carry a
    spurious delta for the files a narrower language filter had skipped.
    """
    return Config(cache=True, cache_dir=str(repo / ".archex"))


def _build_project_index(repo: Path) -> None:
    """Initialize repo-local project state and build its `.archex/index.db`."""
    init_project(repo)
    store = index_repository(
        RepoSource(local_path=str(repo)),
        config=_project_config(repo),
        index_config=IndexConfig(),
    )
    store.close()


def _set_metadata(index_path: Path, key: str, value: str) -> None:
    conn = sqlite3.connect(index_path)
    try:
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def _read_metadata(index_path: Path, key: str) -> str | None:
    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    return None if row is None else str(row[0])


@pytest.fixture
def seeded_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A main checkout with a built index plus a linked worktree with none."""
    main = _init_repo(tmp_path / "main")
    _build_project_index(main)
    linked = tmp_path / "linked"
    _git(main, "worktree", "add", "-b", "feature", str(linked))
    return main, linked


class TestResolveCheckoutIdentity:
    def test_ordinary_checkout_is_not_a_linked_worktree(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")

        identity = resolve_checkout_identity(main)

        assert identity is not None
        assert identity.root == main.resolve()
        assert identity.git_dir == identity.common_dir
        assert identity.is_linked_worktree is False

    def test_linked_worktree_shares_the_common_directory(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        linked = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(linked))

        main_identity = resolve_checkout_identity(main)
        linked_identity = resolve_checkout_identity(linked)

        assert main_identity is not None
        assert linked_identity is not None
        assert (linked / ".git").is_file()
        assert linked_identity.is_linked_worktree is True
        assert linked_identity.common_dir == main_identity.common_dir
        assert linked_identity.git_dir != linked_identity.common_dir

    def test_identity_resolves_from_a_subdirectory(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        linked = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(linked))
        nested = linked / "pkg" / "deep"
        nested.mkdir(parents=True)

        identity = resolve_checkout_identity(nested)

        assert identity is not None
        assert identity.root == linked.resolve()
        assert identity.is_linked_worktree is True

    def test_submodule_is_not_a_linked_worktree(self, tmp_path: Path) -> None:
        child = _init_repo(tmp_path / "child", files={"c.py": "C = 1\n"})
        parent = _init_repo(tmp_path / "parent", files={"p.py": "P = 1\n"})
        _git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(child),
            "vendor/child",
        )
        _git(parent, "commit", "-m", "add submodule")

        identity = resolve_checkout_identity(parent / "vendor" / "child")

        assert identity is not None
        assert (parent / "vendor" / "child" / ".git").is_file()
        assert identity.git_dir == identity.common_dir
        assert identity.is_linked_worktree is False

    def test_separate_git_dir_checkout_is_not_a_linked_worktree(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()
        _git(work, "init", "--separate-git-dir", str(tmp_path / "elsewhere.git"))

        identity = resolve_checkout_identity(work)

        assert identity is not None
        assert (work / ".git").is_file()
        assert identity.git_dir == identity.common_dir
        assert identity.is_linked_worktree is False

    def test_bare_repository_has_no_usable_identity(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.git"
        bare.mkdir()
        _git(bare, "init", "--bare")

        assert resolve_checkout_identity(bare) is None

    def test_non_repository_has_no_identity(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()

        assert resolve_checkout_identity(plain) is None
        assert resolve_checkout_identity(tmp_path / "missing") is None


class TestSelectWorktreeSeed:
    def test_selects_the_main_checkout_index(self, seeded_pair: tuple[Path, Path]) -> None:
        main, linked = seeded_pair

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert selection.eligible
        assert selection.reason == "eligible"
        candidate = selection.candidate
        assert candidate is not None
        assert candidate.root == main.resolve()
        assert candidate.index_path == main.resolve() / ".archex" / "index.db"
        assert candidate.commit_hash == _head(main)
        assert candidate.same_commit is True

    def test_ordinary_checkout_is_refused(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        _build_project_index(main)

        selection = select_worktree_seed(
            main, destination_head=_head(main), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert selection.reason == "not_a_linked_worktree"

    def test_submodule_checkout_is_refused(self, tmp_path: Path) -> None:
        child = _init_repo(tmp_path / "child", files={"c.py": "C = 1\n"})
        parent = _init_repo(tmp_path / "parent", files={"p.py": "P = 1\n"})
        _git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(child),
            "vendor/child",
        )
        _git(parent, "commit", "-m", "add submodule")
        submodule = parent / "vendor" / "child"

        selection = select_worktree_seed(
            submodule, destination_head=_head(submodule), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert selection.reason == "not_a_linked_worktree"

    def test_unrelated_repository_behind_a_stale_pointer_is_refused(self, tmp_path: Path) -> None:
        """A repointed `worktrees/<name>/gitdir` must not serve another repo's index.

        `git worktree list` reports whatever that pointer names, so the
        listing alone would offer `unrelated` as a seed. Only the
        candidate-side common-directory check catches it.
        """
        main = _init_repo(tmp_path / "main")
        linked = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(linked))
        unrelated = _init_repo(tmp_path / "unrelated", files={"u.py": "U = 1\n"})
        _build_project_index(unrelated)

        # Repoint the main checkout's record for `linked` at the unrelated tree.
        pointer = main / ".git" / "worktrees" / "linked" / "gitdir"
        pointer.write_text(f"{unrelated / '.git'}\n", encoding="utf-8")
        listing = _git(linked, "worktree", "list", "--porcelain")
        assert str(unrelated) in listing

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert selection.reason == "no_eligible_seed"
        reasons = {rejection.reason for rejection in selection.rejections}
        assert "different_repository" in reasons
        assert unrelated.resolve() in {rejection.root for rejection in selection.rejections}

    def test_pruned_worktree_directory_is_not_offered(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        _build_project_index(main)
        linked = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(linked))
        gone = tmp_path / "gone"
        _git(main, "worktree", "add", "-b", "gone-branch", str(gone))
        shutil.rmtree(gone)

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert selection.eligible
        assert gone.resolve() not in {rejection.root for rejection in selection.rejections}

    def test_worktree_path_containing_a_newline_is_parsed_whole(self, tmp_path: Path) -> None:
        """Git does not escape a newline in a worktree path; the parser must not split on it."""
        main = _init_repo(tmp_path / "main")
        linked = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(linked))
        awkward = tmp_path / "wt\nINJECTED"
        _git(main, "worktree", "add", "--detach", str(awkward))
        _build_project_index(awkward)

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert selection.eligible
        candidate = selection.candidate
        assert candidate is not None
        assert candidate.root == awkward.resolve()

    def test_ambient_git_location_env_cannot_redirect_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An inherited GIT_DIR/GIT_WORK_TREE pair must not make two repos look like one.

        The same-repository proof rests on a directory reporting its own
        common directory. Git honours these variables over the working
        directory, so leaving them in the environment would let an unrelated
        checkout answer for this one.
        """
        main = _init_repo(tmp_path / "main")
        unrelated = _init_repo(tmp_path / "unrelated", files={"u.py": "U = 1\n"})
        monkeypatch.setenv("GIT_DIR", str(main / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(unrelated))

        identity = resolve_checkout_identity(unrelated)

        assert identity is not None
        assert identity.root == unrelated.resolve()
        assert identity.common_dir == (unrelated / ".git").resolve()
        assert identity.is_linked_worktree is False

    def test_planted_index_without_a_cache_marker_is_refused(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """A committed or planted `index.db` must not be trusted on its own.

        Everything else about a candidate is read out of the candidate's own
        database, so whoever supplies the database supplies the evidence for
        accepting it. `.archex/index.db` and `.archex/settings.toml` are
        ordinary files, so a published repository can commit both. The cache
        marker is the one thing committed content cannot forge, because it
        is keyed on the checkout's absolute path.
        """
        main, linked = seeded_pair
        (main / ".archex" / "index.meta").unlink()

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["unprovenanced_index"]

    def test_marker_for_a_different_directory_is_refused(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """A marker copied from elsewhere does not bind to this checkout."""
        main, linked = seeded_pair
        marker = main / ".archex" / "index.meta"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["cache_key"] = "0" * 64
        marker.write_text(json.dumps(payload), encoding="utf-8")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["unprovenanced_index"]

    def test_marker_forged_for_a_predictable_path_is_refused(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """Knowing the checkout path is not enough to produce a valid marker.

        Both halves of the cache key are guessable — the commit can be the
        attacker's own, and checkout paths are fixed on CI runners and in
        container images — so the marker is authenticated with a secret
        that only this machine holds.
        """
        main, linked = seeded_pair
        marker = main / ".archex" / "index.meta"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        # The forger knows the path and the revision, so it computes the
        # key correctly; it does not hold this machine's secret.
        cache = CacheManager(cache_dir=str(main / ".archex"), project_layout=True)
        forged_key = cache.cache_key(RepoSource(local_path=str(main)), head_override=_head(main))
        assert payload["cache_key"] == forged_key
        payload["marker_hmac"] = hashlib.sha256(forged_key.encode()).hexdigest()
        marker.write_text(json.dumps(payload), encoding="utf-8")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["unprovenanced_index"]

    def test_marker_from_another_machine_is_refused(
        self, seeded_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A marker copied from a machine with a different secret does not verify."""
        _main, linked = seeded_pair
        monkeypatch.setattr(cache_module, "_machine_secret", lambda: b"a-different-machine")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["unprovenanced_index"]

    def test_unreadable_marker_is_a_refusal_not_a_crash(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """Reading a sibling checkout must never fail the command that asked."""
        main, linked = seeded_pair
        (main / ".archex" / "index.meta").write_bytes(b"\xff\xfe\x00not utf-8")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["unprovenanced_index"]

    def test_candidate_failing_mid_evaluation_is_a_refusal(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """A candidate can be reset or replaced while this checkout examines it.

        A candidate is another checkout this process does not own, so any
        read of it can fail at any point. That must produce a refusal, not
        fail the indexing command that merely asked whether a seed exists.
        """
        _main, linked = seeded_pair

        with patch.object(
            worktree_seed,
            "_read_store_metadata",
            side_effect=OSError(2, "No such file or directory"),
        ):
            selection = select_worktree_seed(
                linked, destination_head=_head(linked), index_config=IndexConfig()
            )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["candidate_unreadable"]

    def test_oversized_candidate_index_is_refused(
        self, seeded_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        main, linked = seeded_pair
        size = (main / ".archex" / "index.db").stat().st_size
        monkeypatch.setattr(worktree_seed, "MAX_SEED_INDEX_BYTES", size - 1)

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["index_too_large"]

    def test_candidate_list_is_bounded(
        self, seeded_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interrogating every checkout of a many-worktree repository is not free."""
        main, linked = seeded_pair
        for name in ("extra_one", "extra_two"):
            _git(main, "worktree", "add", "--detach", str(linked.parent / name))
        monkeypatch.setattr(worktree_seed, "MAX_SEED_CANDIDATES", 1)

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        reasons = [rejection.reason for rejection in selection.rejections]
        assert reasons.count("candidate_limit_reached") == 3

    def test_missing_candidate_index_is_reported(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        linked = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(linked))

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert selection.reason == "no_eligible_seed"
        assert [rejection.reason for rejection in selection.rejections] == ["no_index"]

    def test_incompatible_schema_is_refused(self, seeded_pair: tuple[Path, Path]) -> None:
        main, linked = seeded_pair
        index_path = main / ".archex" / "index.db"
        _set_metadata(index_path, "schema_version", "4")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["incompatible_schema"]

    def test_candidate_index_is_never_written_by_inspection(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """Inspecting a candidate must not modify a checkout this process does not own.

        `IndexStore`'s constructor migrates schema on open, which would
        rewrite `schema_version` in the sibling's database; the read-only
        connection the module uses cannot. SQLite may still materialize
        empty `-wal`/`-shm` sidecars for any WAL-mode reader — the
        sibling's own commands do the same — so the invariant asserted
        here is that the database's bytes are untouched.
        """
        main, linked = seeded_pair
        index_path = main / ".archex" / "index.db"
        _set_metadata(index_path, "schema_version", "4")
        digest_before = hashlib.sha256(index_path.read_bytes()).hexdigest()

        select_worktree_seed(linked, destination_head=_head(linked), index_config=IndexConfig())

        assert _read_metadata(index_path, "schema_version") == "4"
        assert hashlib.sha256(index_path.read_bytes()).hexdigest() == digest_before

    def test_needs_reindex_candidate_is_refused(self, seeded_pair: tuple[Path, Path]) -> None:
        main, linked = seeded_pair
        _set_metadata(main / ".archex" / "index.db", "needs_reindex", "true")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["needs_reindex"]

    def test_incompatible_index_config_is_refused(self, seeded_pair: tuple[Path, Path]) -> None:
        main, linked = seeded_pair
        _set_metadata(main / ".archex" / "index.db", "chunker_revision", "not-this-build")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        rejection = selection.rejections[0]
        assert rejection.reason == "incompatible_index_config"
        assert "chunker_revision" in rejection.detail

    def test_candidate_without_a_revision_is_refused(self, seeded_pair: tuple[Path, Path]) -> None:
        main, linked = seeded_pair
        _set_metadata(main / ".archex" / "index.db", "commit_hash", "")

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["no_revision"]

    def test_vector_or_splade_index_config_is_unsupported(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        _main, linked = seeded_pair

        vector = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig(vector=True)
        )
        splade = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig(splade=True)
        )

        assert vector.reason == "unsupported_index_config"
        assert "vector" in vector.detail
        assert splade.reason == "unsupported_index_config"
        assert "SPLADE" in splade.detail

    def test_symlinked_candidate_project_dir_is_refused(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        main, linked = seeded_pair
        project_dir = main / ".archex"
        elsewhere = main.parent / "elsewhere"
        project_dir.rename(elsewhere)
        project_dir.symlink_to(elsewhere, target_is_directory=True)

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert not selection.eligible
        assert [rejection.reason for rejection in selection.rejections] == ["symlinked_path"]

    def test_same_commit_candidate_wins_over_a_newer_diverged_one(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        _build_project_index(main)
        same_commit_worktree = tmp_path / "sibling"
        _git(main, "worktree", "add", "--detach", str(same_commit_worktree), "main")
        _build_project_index(same_commit_worktree)

        # Move the main checkout to a different revision and re-index it, so
        # it is the most recently measured candidate but no longer matches.
        (main / "b.py").write_text("def b() -> int:\n    return 2\n", encoding="utf-8")
        _git(main, "add", "-A")
        _git(main, "commit", "-m", "second")
        _build_project_index(main)

        destination = tmp_path / "linked"
        _git(main, "worktree", "add", "-b", "feature", str(destination), "main")
        _git(destination, "reset", "--hard", _head(same_commit_worktree))

        selection = select_worktree_seed(
            destination, destination_head=_head(destination), index_config=IndexConfig()
        )

        assert selection.eligible
        candidate = selection.candidate
        assert candidate is not None
        assert candidate.root == same_commit_worktree.resolve()
        assert candidate.same_commit is True

    def test_selected_candidate_index_is_a_usable_store(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        _main, linked = seeded_pair

        selection = select_worktree_seed(
            linked, destination_head=_head(linked), index_config=IndexConfig()
        )

        assert selection.candidate is not None
        store = IndexStore(selection.candidate.index_path)
        try:
            assert store.get_metadata("schema_version") == CURRENT_SCHEMA_VERSION
            assert store.get_chunk_count() > 0
        finally:
            store.close()


def _destination_cache(repo: Path) -> tuple[CacheManager, str]:
    """The project-layout cache manager and cache key a destination would use."""
    cache = CacheManager(cache_dir=str(repo / ".archex"), project_layout=True)
    return cache, cache.cache_key(RepoSource(local_path=str(repo)))


def _seed(repo: Path, *, config: Config | None = None) -> WorktreeSeedResult:
    """Run a seeding attempt exactly as the indexing path would."""
    cache, cache_key = _destination_cache(repo)
    return seed_worktree_index(
        repo,
        cache=cache,
        cache_key=cache_key,
        source_identity=str(repo),
        config=config or _project_config(repo),
        index_config=IndexConfig(),
    )


def _open_store_snapshot(store: IndexStore) -> tuple[list[str], list[str]]:
    """Sorted chunk ids and indexed file paths of an open store."""
    return (
        sorted(chunk.id for chunk in store.get_chunks()),
        sorted(store.get_file_states()),
    )


def _full_store_snapshot(index_path: Path) -> dict[str, object]:
    """Everything two stores of the same tree must agree on, read directly.

    Chunk *content* and edge *evidence* are included deliberately: they are
    the fields that differ between a delta-updated store and a freshly
    parsed one, so comparing them is what makes an equivalence claim
    falsifiable. `indexed_at` and `source_identity` are excluded because
    they describe the run and the checkout, not the index.
    """
    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        return {
            "chunks": sorted(conn.execute("SELECT id, file_path, content FROM chunks")),
            "edges": sorted(
                conn.execute("SELECT source, target, kind, location, evidence FROM edges")
            ),
            "file_states": sorted(conn.execute("SELECT file_path, sha256 FROM file_states")),
            "metadata": {
                key: value
                for key, value in conn.execute("SELECT key, value FROM metadata")
                if key not in {"indexed_at", "source_identity"}
            },
        }
    finally:
        conn.close()


def _store_snapshot(index_path: Path) -> tuple[list[str], list[str]]:
    """Sorted chunk ids and indexed file paths of a store on disk."""
    store = IndexStore(index_path)
    try:
        return _open_store_snapshot(store)
    finally:
        store.close()


@pytest.fixture
def seed_destination(seeded_pair: tuple[Path, Path]) -> tuple[Path, Path]:
    """A linked worktree with initialized project state but no index yet."""
    main, linked = seeded_pair
    init_project(linked)
    assert not (linked / ".archex" / "index.db").exists()
    return main, linked


class TestSeedWorktreeIndex:
    def test_clean_seed_publishes_a_usable_destination_index(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        main, linked = seed_destination
        _cache, cache_key = _destination_cache(linked)

        result = _seed(linked)

        assert result.installed
        assert result.reason == "seeded"
        assert result.sync_strategy == "clean"
        assert result.files_changed == 0
        assert result.source_root == main.resolve()

        index_path = linked / ".archex" / "index.db"
        assert index_path.is_file()
        meta = json.loads((linked / ".archex" / "index.meta").read_text(encoding="utf-8"))
        assert meta["cache_key"] == cache_key
        assert meta["resolved_commit"] == _head(linked)

        store = IndexStore(index_path)
        try:
            assert store.get_chunk_count() > 0
            assert store.get_metadata("source_identity") == str(linked)
            assert store.get_metadata("commit_hash") == _head(linked)
            assert store.get_metadata("working_tree_signature") == "clean"
            assert store.get_metadata("generation_id")
        finally:
            store.close()

    def test_seeded_index_is_reused_instead_of_rebuilt(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """The point of stamping identity: the next index run must not rebuild."""
        _main, linked = seed_destination
        assert _seed(linked).installed

        timing = PipelineTiming()
        store = index_repository(
            RepoSource(local_path=str(linked)),
            config=_project_config(linked),
            index_config=IndexConfig(),
            timing=timing,
        )
        store.close()

        assert timing.strategy == "cached"

    def test_seeded_index_survives_a_new_destination_commit(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """A new commit changes the cache key, so reuse then rests on stamped identity.

        The `index.meta` marker is keyed on the destination's resolved path
        *and* its HEAD, so the exact-hit path stops matching as soon as the
        worktree commits. What keeps the seeded store from being thrown away
        at that point is its `source_identity`, which the seed rewrote to the
        destination — without it, the next run indexes from scratch.
        """
        _main, linked = seed_destination
        assert _seed(linked).installed
        (linked / "committed.py").write_text("def committed() -> int:\n    return 5\n")
        _git(linked, "add", "-A")
        _git(linked, "commit", "-m", "destination commit")

        timing = PipelineTiming()
        store = index_repository(
            RepoSource(local_path=str(linked)),
            config=_project_config(linked),
            index_config=IndexConfig(),
            timing=timing,
        )
        store.close()

        assert timing.strategy == "delta"

    def test_delta_seed_applies_destination_changes(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        _main, linked = seed_destination
        (linked / "added.py").write_text("def added() -> int:\n    return 3\n", encoding="utf-8")

        result = _seed(linked)

        assert result.installed
        assert result.sync_strategy == "delta"
        assert result.files_changed == 1
        chunks, files = _store_snapshot(linked / ".archex" / "index.db")
        assert "added.py" in files
        assert any("added.py" in chunk for chunk in chunks)

    def test_seeded_and_clean_built_indexes_agree(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """Retrieval equivalence: identical corpora and identical generation identity."""
        _main, linked = seed_destination
        (linked / "extra.py").write_text("def extra() -> int:\n    return 4\n", encoding="utf-8")
        assert _seed(linked).installed
        seeded_store = IndexStore(linked / ".archex" / "index.db")
        try:
            seeded = _open_store_snapshot(seeded_store)
            seeded_generation = seeded_store.get_metadata("generation_id")
        finally:
            seeded_store.close()

        # The comparison build gets a cache directory outside the repository:
        # an untracked directory inside it would itself be discovered content.
        clean_store = index_repository(
            RepoSource(local_path=str(linked)),
            config=Config(cache=True, cache_dir=str(linked.parent / "isolated")),
            index_config=IndexConfig(),
        )
        try:
            clean = _open_store_snapshot(clean_store)
            clean_generation = clean_store.get_metadata("generation_id")
        finally:
            clean_store.close()

        assert seeded == clean
        assert seeded_generation == clean_generation

    def test_seeded_store_equals_the_same_tree_delta_indexed_in_place(self, tmp_path: Path) -> None:
        """The load-bearing equivalence: a seeded worktree is a delta-indexed worktree.

        A seed copies an index built at one revision and synchronizes it to a
        tree at another, which is exactly what ordinary delta indexing does
        when a checkout moves forward. Both stores must therefore be the
        same store, down to edge evidence — the field `apply_delta` and a
        full parse disagree on, and so the field that would expose any
        divergence between the two routes.
        """
        main = _init_repo(tmp_path / "main")
        _build_project_index(main)

        (main / "later.py").write_text(
            "from mod_1 import value_1\n\n\ndef later() -> int:\n    return value_1()\n",
            encoding="utf-8",
        )
        _git(main, "add", "-A")
        _git(main, "commit", "-m", "second")

        # Seed first, while the source index still describes the old revision.
        destination = tmp_path / "linked"
        _git(main, "worktree", "add", "--detach", str(destination))
        init_project(destination)
        seeded = _seed(destination)
        assert seeded.installed
        assert seeded.sync_strategy == "delta"
        seeded_snapshot = _full_store_snapshot(destination / ".archex" / "index.db")

        # Then let the source index itself forward to the same tree.
        timing = PipelineTiming()
        store = index_repository(
            RepoSource(local_path=str(main)),
            config=_project_config(main),
            index_config=IndexConfig(),
            timing=timing,
        )
        store.close()
        assert timing.strategy == "delta"

        assert seeded_snapshot == _full_store_snapshot(main / ".archex" / "index.db")

    def test_large_delta_installs_nothing(self, seed_destination: tuple[Path, Path]) -> None:
        _main, linked = seed_destination
        for existing in linked.glob("*.py"):
            existing.write_text("REWRITTEN = True\n", encoding="utf-8")

        result = _seed(
            linked,
            config=Config(
                cache=True,
                cache_dir=str(linked / ".archex"),
                delta_threshold=0.1,
            ),
        )

        assert not result.installed
        assert result.reason == "large_delta"
        assert not (linked / ".archex" / "index.db").exists()
        assert not (linked / ".archex" / "index.meta").exists()

    def test_incompatible_source_installs_nothing(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        main, linked = seed_destination
        _set_metadata(main / ".archex" / "index.db", "needs_reindex", "true")

        result = _seed(linked)

        assert not result.installed
        assert result.reason == "no_eligible_seed"
        assert [rejection.reason for rejection in result.rejections] == ["needs_reindex"]
        assert not (linked / ".archex" / "index.db").exists()

    def test_second_attempt_reports_the_existing_index(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        _main, linked = seed_destination
        assert _seed(linked).installed

        result = _seed(linked)

        assert not result.installed
        assert result.reason == "destination_index_present"

    def test_a_concurrent_seeder_is_not_queued_behind(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        _main, linked = seed_destination
        holder = ExclusiveLock(
            linked / ".archex" / SEED_LOCK_FILENAME,
            timeout=1.0,
            cwd=linked,
            diagnostics=StateFileDiagnostics(
                lock_error="test-lock-error",
                lock_timeout="test-lock-timeout",
                write_error="test-write-error",
            ),
        )
        with holder as acquired:
            assert acquired
            result = _seed(linked)

        assert not result.installed
        assert result.reason == "seed_in_progress"
        assert not (linked / ".archex" / "index.db").exists()

    def test_failure_mid_seed_leaves_the_destination_untouched(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        _main, linked = seed_destination

        with patch(
            "archex.index.worktree_seed._sync_staged_seed",
            side_effect=sqlite3.DatabaseError("staged copy is corrupt"),
        ):
            result = _seed(linked)

        assert not result.installed
        assert result.reason == "seed_failed"
        assert "staged copy is corrupt" in result.detail
        assert not (linked / ".archex" / "index.db").exists()
        assert not (linked / ".archex" / "index.meta").exists()
        assert list((linked / ".archex").glob(".seed-*")) == []

    def test_snapshot_swapped_after_validation_is_refused(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """The database that was validated must be the database that is installed.

        Eligibility is decided against the candidate's path; the snapshot is
        taken from that path again afterwards. Anything able to write there
        in between — another process publishing its own index, say — would
        otherwise get unvalidated bytes installed.
        """
        _main, linked = seed_destination
        foreign = _init_repo(linked.parent / "foreign", files={"f.py": "F = 1\n"})
        _build_project_index(foreign)

        original = worktree_seed._snapshot_index  # pyright: ignore[reportPrivateUsage]

        def swapped(source_db: Path, dest_db: Path) -> None:
            del source_db
            original(foreign / ".archex" / "index.db", dest_db)

        with patch.object(worktree_seed, "_snapshot_index", swapped):
            result = _seed(linked)

        assert not result.installed
        assert result.reason == "staged_copy_rejected"
        assert "revision" in result.detail
        assert not (linked / ".archex" / "index.db").exists()

    def test_snapshot_declaring_a_trigger_is_refused(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """archex's schema declares no trigger; a snapshot that does is not ours."""
        _main, linked = seed_destination
        original = worktree_seed._snapshot_index  # pyright: ignore[reportPrivateUsage]

        def with_trigger(source_db: Path, dest_db: Path) -> None:
            original(source_db, dest_db)
            conn = sqlite3.connect(dest_db)
            try:
                conn.execute(
                    "CREATE TRIGGER poison AFTER INSERT ON chunks BEGIN SELECT randomblob(1); END"
                )
                conn.commit()
            finally:
                conn.close()

        with patch.object(worktree_seed, "_snapshot_index", with_trigger):
            result = _seed(linked)

        assert not result.installed
        assert result.reason == "staged_copy_rejected"
        assert "trigger poison" in result.detail
        assert not (linked / ".archex" / "index.db").exists()

    def test_a_failure_while_looking_for_a_seed_never_fails_indexing(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """Looking for a seed reads other checkouts; that must not fail the command."""
        _main, linked = seed_destination

        with patch.object(
            worktree_seed, "select_worktree_seed", side_effect=OSError(5, "I/O error")
        ):
            result = _seed(linked)

        assert not result.installed
        assert result.reason == "seed_failed"
        assert "I/O error" in result.detail
        assert not (linked / ".archex" / "index.db").exists()

    def test_staging_left_by_a_dead_seeder_is_reclaimed(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        _main, linked = seed_destination
        abandoned = linked / ".archex" / ".seed-999999-1"
        abandoned.mkdir()
        (abandoned / "index.db").write_text("partial", encoding="utf-8")

        result = _seed(linked)

        assert result.installed
        assert not abandoned.exists()
        assert list((linked / ".archex").glob(".seed-*")) == []

    def test_only_index_state_crosses_over(self, seed_destination: tuple[Path, Path]) -> None:
        """Locks, WAL/SHM, sessions, metrics, and settings must never be copied."""
        main, linked = seed_destination
        source_project = main / ".archex"
        excluded = (
            "index.db-wal",
            "index.db-shm",
            "status-snapshot.json",
            "status-snapshot.lock",
            "post-edit-state.json",
            "session.db",
        )
        for name in excluded:
            (source_project / name).write_text("source-only", encoding="utf-8")
        (source_project / "metrics").mkdir(exist_ok=True)
        (source_project / "metrics" / "events.jsonl").write_text("{}\n", encoding="utf-8")
        source_settings = (source_project / "settings.toml").read_text(encoding="utf-8")
        (source_project / "settings.toml").write_text(
            source_settings + '\n[marker]\nfrom_source = "yes"\n', encoding="utf-8"
        )

        assert _seed(linked).installed

        destination_names = {path.name for path in (linked / ".archex").iterdir()}
        assert destination_names.isdisjoint({*excluded, "metrics"})
        assert "from_source" not in (linked / ".archex" / "settings.toml").read_text(
            encoding="utf-8"
        )


class TestIndexingIntegration:
    """The seed path as reached through `index_repository` and the CLI."""

    def test_index_reports_the_seed_strategy(self, seed_destination: tuple[Path, Path]) -> None:
        main, linked = seed_destination

        result = CliRunner().invoke(cli, ["index", str(linked), "--format", "json"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert summary["strategy"] == "seeded"
        assert summary["seed_disposition"] == "seeded"
        assert summary["seed_strategy"] == "clean"
        assert summary["seed_source"] == str(main.resolve())
        assert summary["files_indexed"] == len(_DEFAULT_FILES) + _INIT_WRITTEN_FILES

    def test_index_text_output_names_the_seed(self, seed_destination: tuple[Path, Path]) -> None:
        main, linked = seed_destination

        result = CliRunner().invoke(cli, ["index", str(linked)])

        assert result.exit_code == 0, result.output
        assert "Strategy:           seeded" in result.output
        assert f"Worktree seed:      seeded from {main.resolve()}" in result.output

    def test_refused_seed_is_reported_and_indexes_normally(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        main, linked = seed_destination
        _set_metadata(main / ".archex" / "index.db", "needs_reindex", "true")

        result = CliRunner().invoke(cli, ["index", str(linked), "--format", "json"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert summary["strategy"] == "full"
        assert summary["seed_disposition"] == "no_eligible_seed"
        assert summary["seed_strategy"] is None
        assert summary["files_indexed"] == len(_DEFAULT_FILES) + _INIT_WRITTEN_FILES

    def test_ordinary_checkout_reports_no_seed_section(self, tmp_path: Path) -> None:
        main = _init_repo(tmp_path / "main")
        init_project(main)

        result = CliRunner().invoke(cli, ["index", str(main), "--format", "json"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert "seed_disposition" not in summary
        assert "Worktree seed" not in result.output

    def test_seeding_can_be_disabled_in_project_settings(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        _main, linked = seed_destination
        settings = linked / ".archex" / "settings.toml"
        settings.write_text(
            settings.read_text(encoding="utf-8").replace(
                "worktree_seed = true", "worktree_seed = false"
            ),
            encoding="utf-8",
        )

        result = CliRunner().invoke(cli, ["index", str(linked), "--format", "json"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert summary["strategy"] == "full"
        assert "seed_disposition" not in summary

    def test_query_seeds_a_fresh_worktree(self, seed_destination: tuple[Path, Path]) -> None:
        """A query in a fresh worktree must reach the same seeded index."""
        _main, linked = seed_destination

        result = CliRunner().invoke(cli, ["query", str(linked), "value_3", "--format", "json"])

        assert result.exit_code == 0, result.output
        assert (linked / ".archex" / "index.db").is_file()
        assert (linked / ".archex" / "index.meta").is_file()
        assert "mod_3.py" in result.output

    def test_discarded_seed_reports_the_real_strategy(
        self, seed_destination: tuple[Path, Path]
    ) -> None:
        """A published seed the resolution declines must not be reported as the outcome.

        Here the seed is published and then immediately flagged
        `needs_reindex`, which the resolution refuses to reuse. The reported
        strategy has to be what indexing actually did, not `seeded`.
        """
        _main, linked = seed_destination
        original = worktree_seed.seed_worktree_index

        def seed_then_spoil(*args: object, **kwargs: object) -> WorktreeSeedResult:
            result = original(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]
            if result.installed:
                _set_metadata(linked / ".archex" / "index.db", "needs_reindex", "true")
            return result

        with patch("archex.api.seed_worktree_index", seed_then_spoil):
            result = CliRunner().invoke(cli, ["index", str(linked), "--format", "json"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert summary["strategy"] == "full"
        assert summary["seed_disposition"] == "seed_discarded"
        assert summary["seed_strategy"] is None

    def test_seed_source_control_characters_are_not_echoed(
        self, seeded_pair: tuple[Path, Path]
    ) -> None:
        """A checkout name is filesystem input, so it must not carry escapes to a terminal."""
        main, _linked = seeded_pair
        hostile = main.parent / "wt\x1b[31mred"
        _git(main, "worktree", "add", "--detach", str(hostile))
        _build_project_index(hostile)
        destination = main.parent / "destination"
        _git(main, "worktree", "add", "--detach", str(destination))
        init_project(destination)

        result = CliRunner().invoke(cli, ["index", str(destination)])

        assert result.exit_code == 0, result.output
        assert "Worktree seed:      seeded from" in result.output
        assert "\x1b" not in result.output
        assert "wt?[31mred" in result.output
