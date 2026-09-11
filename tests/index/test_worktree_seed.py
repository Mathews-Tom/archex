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

from archex import cache as cache_module
from archex.api import index_repository
from archex.cache import CacheManager
from archex.index import worktree_seed
from archex.index.store import CURRENT_SCHEMA_VERSION, IndexStore
from archex.index.worktree_seed import (
    resolve_checkout_identity,
    select_worktree_seed,
)
from archex.models import Config, IndexConfig, RepoSource
from archex.project import init_project

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _init_repo(root: Path, *, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@archex.test")
    _git(root, "config", "user.name", "archex-test")
    for name, body in (files or {"a.py": "def a() -> int:\n    return 1\n"}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "initial")
    return root


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _build_project_index(repo: Path) -> None:
    """Initialize repo-local project state and build its `.archex/index.db`."""
    init_project(repo)
    store = index_repository(
        RepoSource(local_path=str(repo)),
        config=Config(languages=["python"], cache=True, cache_dir=str(repo / ".archex")),
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
