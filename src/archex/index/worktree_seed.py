"""Same-repository worktree index seed discovery.

A linked Git worktree (`git worktree add`) starts with no `.archex/index.db`
and, left alone, pays a full cold index for a tree that is usually a few
commits away from a checkout that is already indexed on the same machine.
This module answers one question — *is there a checkout whose index this
worktree may legitimately start from?* — and answers it conservatively:

* **Identity is proven twice.** Candidates are enumerated with
  `git worktree list --porcelain`, then each candidate is re-interrogated
  from its own directory and must report the same Git *common directory*.
  A one-sided listing is not proof: a stale or hand-edited
  `.git/worktrees/<name>/gitdir` makes `git worktree list` report an
  unrelated checkout as a worktree, and that checkout would then serve
  another repository's code.
* **Only `git rev-parse` runs inside a candidate.** `rev-parse` executes no
  hook, no `core.fsmonitor` program, and no filter or diff driver, so
  identity is established without running repository-configured code.
  Commands that can (`git status`, `git ls-files`) run only against the
  destination working tree, which the caller already owns.
* **Submodules, bare repositories, and `--separate-git-dir` checkouts are
  refused structurally.** In all three shapes the Git directory *equals*
  the common directory, so they never satisfy the linked-worktree test —
  no name heuristics are involved.
* **Every refusal is a named disposition.** Nothing is silently skipped.

Once a source is eligible, its index is snapshotted into a staging
directory, delta-synchronized against the *destination* working tree there,
and only then published — so a failure at any earlier point leaves the
destination exactly as it was, and ordinary full indexing remains the
fallback for every disposition other than `seeded`.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from archex.cache import CacheManager
from archex.exceptions import ArchexError
from archex.index.adopt import stamp_project_index_identity
from archex.index.compat import INDEX_CONFIG_METADATA_KEYS, index_config_metadata_mismatch
from archex.index.store import CURRENT_SCHEMA_VERSION, IndexStore
from archex.models import RepoSource
from archex.state_file import ExclusiveLock, StateFileDiagnostics

if TYPE_CHECKING:
    from archex.models import Config, IndexConfig

logger = logging.getLogger(__name__)

#: Timeout for every `git` invocation this module makes. Matches
#: `ProjectState.resolve`'s existing `rev-parse` budget; all calls here are
#: metadata reads that touch no object database.
GIT_TIMEOUT_SECONDS = 10

_REV_PARSE_FLAGS = (
    "--git-common-dir",
    "--git-dir",
    "--show-toplevel",
    "--is-bare-repository",
    "--is-inside-work-tree",
)

#: Store metadata a candidate must carry before it can be a seed. The
#: index-config keys come from `archex.index.compat`; the rest establish
#: schema compatibility and a delta base.
_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "needs_reindex",
    "commit_hash",
    "indexed_at",
    *INDEX_CONFIG_METADATA_KEYS,
)


@dataclass(frozen=True)
class CheckoutIdentity:
    """Resolved Git identity of one directory, as reported by `git rev-parse`."""

    root: Path
    """The checkout's own top level (`--show-toplevel`), resolved."""

    git_dir: Path
    """This checkout's Git directory (`--git-dir`), resolved."""

    common_dir: Path
    """The repository-wide Git directory (`--git-common-dir`), resolved."""

    @property
    def is_linked_worktree(self) -> bool:
        """Whether this checkout is a linked worktree of another checkout.

        True only when the per-checkout Git directory differs from the
        repository-wide common directory. Ordinary checkouts, submodules,
        and `--separate-git-dir` checkouts all report the two as equal.
        """
        return self.git_dir != self.common_dir


@dataclass(frozen=True)
class SeedRejection:
    """One candidate checkout that was considered and refused."""

    root: Path
    reason: str
    detail: str


@dataclass(frozen=True)
class SeedCandidate:
    """A checkout whose index is eligible to seed the destination worktree."""

    root: Path
    """The candidate checkout's top level."""

    index_path: Path
    """The candidate's `.archex/index.db`."""

    commit_hash: str
    """The revision the candidate's index reflects."""

    indexed_at: float
    """When the candidate's index was last measured against its tree."""

    same_commit: bool
    """Whether the candidate index's revision equals the destination's HEAD."""


@dataclass(frozen=True)
class SeedSelection:
    """Outcome of looking for a seed source for one destination worktree."""

    candidate: SeedCandidate | None
    reason: str
    """`eligible` when a candidate was selected, else the refusal disposition."""

    detail: str = ""
    rejections: tuple[SeedRejection, ...] = field(default_factory=tuple)

    @property
    def eligible(self) -> bool:
        return self.candidate is not None


#: Environment variables that redirect Git away from the directory it is
#: invoked in. They are removed from every invocation here because this
#: module's whole guarantee is that a *directory* reports its own
#: repository: an ambient `GIT_DIR`/`GIT_WORK_TREE` pair makes two
#: unrelated checkouts resolve to one common directory, which would defeat
#: the candidate-side identity check. archex inherits these when it runs
#: from inside a Git hook or an env-exporting wrapper.
_GIT_LOCATION_ENV_VARS = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
)


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _GIT_LOCATION_ENV_VARS}


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run a `git` command, returning None when it cannot be run at all."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
            env=_git_env(),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("git %s failed in %s: %s", " ".join(args), cwd, exc)
        return None


def _rev_parse_values(path: Path) -> list[str] | None:
    """Read `_REV_PARSE_FLAGS` for one directory, or None if Git refuses.

    One batched call is the fast path. `git rev-parse` separates its answers
    with newlines and does not escape a newline *inside* a path, so a value
    that contains one makes positional parsing ambiguous; that case falls
    back to asking for each value separately, where the whole of stdout is
    the value.
    """
    batched = _run_git(["rev-parse", "--path-format=absolute", *_REV_PARSE_FLAGS], path)
    if batched is None or batched.returncode != 0:
        return None
    lines = batched.stdout.splitlines()
    if len(lines) == len(_REV_PARSE_FLAGS):
        return lines

    values: list[str] = []
    for flag in _REV_PARSE_FLAGS:
        single = _run_git(["rev-parse", "--path-format=absolute", flag], path)
        if single is None or single.returncode != 0:
            return None
        values.append(single.stdout.removesuffix("\n"))
    return values


def resolve_checkout_identity(path: Path) -> CheckoutIdentity | None:
    """Resolve one directory's Git identity, or None if it is not a work tree.

    Returns None for a path that is not inside a repository, for a bare
    repository, and for anything Git refuses to answer for — every one of
    which is a reason not to treat the path as a seed source or destination.
    `--path-format=absolute` is passed because `--git-common-dir` is
    otherwise reported relative to the current directory.
    """
    if not path.is_dir():
        return None
    values = _rev_parse_values(path)
    if values is None:
        return None
    common_dir, git_dir, toplevel, is_bare, is_work_tree = values
    if is_bare.strip() != "false" or is_work_tree.strip() != "true" or not toplevel:
        return None
    return CheckoutIdentity(
        root=Path(toplevel).resolve(),
        git_dir=Path(git_dir).resolve(),
        common_dir=Path(common_dir).resolve(),
    )


def _parse_worktree_list(stdout: str, *, nul_terminated: bool) -> list[Path]:
    """Extract checkout paths from `git worktree list --porcelain` output.

    Both framings are records of attribute entries with an empty entry
    between records: newline-terminated lines separated by a blank line,
    or — with `-z` — NUL-terminated entries separated by an empty one. The
    NUL framing is preferred because Git does **not** escape a newline
    inside a worktree path, which makes the line-oriented output of a
    worktree whose path contains one ambiguous.

    Bare and prunable records are dropped: a bare repository has no tree to
    read an index from, and a prunable record points at a directory that is
    gone or unreadable. Paths are returned unresolved — Git echoes an
    injected `gitdir` pointer verbatim, so resolution belongs with the
    per-candidate verification that also checks identity.
    """
    paths: list[Path] = []
    current: str | None = None
    usable = True
    for raw in stdout.split("\0" if nul_terminated else "\n"):
        # A path may legitimately end in whitespace, so only the newline
        # framing's own line ending is removed.
        entry = raw if nul_terminated else raw.rstrip("\r")
        if entry.startswith("worktree "):
            if current is not None and usable:
                paths.append(Path(current))
            current = entry[len("worktree ") :]
            usable = True
        elif entry == "bare" or entry.startswith("prunable"):
            usable = False
    if current is not None and usable:
        paths.append(Path(current))
    return paths


def _read_store_metadata(db_path: Path, keys: tuple[str, ...]) -> dict[str, str | None] | None:
    """Read metadata keys from an index database without opening it for writing.

    `IndexStore`'s constructor creates and migrates schema, which would
    write to a checkout this process does not own, so a read-only URI
    connection is used instead. Returns None when the database cannot be
    read or carries no `metadata` table.
    """
    uri = f"file:{quote(str(db_path))}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        logger.debug("could not open seed candidate %s read-only: %s", db_path, exc)
        return None
    try:
        placeholders = ",".join("?" for _ in keys)
        rows = conn.execute(
            f"SELECT key, value FROM metadata WHERE key IN ({placeholders})",  # noqa: S608
            keys,
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("could not read metadata from seed candidate %s: %s", db_path, exc)
        return None
    finally:
        conn.close()
    stored = {str(key): (None if value is None else str(value)) for key, value in rows}
    return {key: stored.get(key) for key in keys}


def _unsupported_index_config(index_config: IndexConfig) -> str | None:
    """Return why `index_config` cannot be served by a seed, or None.

    Vector and SPLADE state is not transferable by this path: keeping
    embeddings correct across a delta needs the embedder pipeline ordinary
    delta indexing owns, and copying either without that would be a silent
    incompatibility. Both are off by default.
    """
    if index_config.vector:
        return "vector retrieval is enabled"
    if index_config.splade:
        return "SPLADE retrieval is enabled"
    return None


def _reject_symlinked(path: Path, label: str) -> SeedRejection | None:
    if path.is_symlink():
        return SeedRejection(root=path, reason="symlinked_path", detail=f"{label} is a symlink")
    return None


#: Hard ceiling on a candidate index's file size. A seed exists to be
#: cheaper than a full index; copying an arbitrarily large database (twice,
#: since publication copies the staged file again) is not. Mirrors the
#: ceiling `archex.index.artifact` puts on a decompressed artifact, and is
#: far above any index this format legitimately carries.
MAX_SEED_INDEX_BYTES = 1024**3

#: Hard ceiling on how many checkouts are interrogated. Each costs a `git`
#: subprocess and a SQLite open, so an unbounded list would let a
#: repository with hundreds of worktrees make seeding slower than the full
#: index it replaces. The remainder is refused by name, never dropped
#: silently.
MAX_SEED_CANDIDATES = 32


def _index_is_provenanced(checkout_root: Path, commit_hash: str) -> bool:
    """Whether a checkout's index carries archex's own cache marker for itself.

    A candidate index is otherwise validated entirely from metadata stored
    *inside* the candidate database — which means whoever supplies the
    database supplies the evidence for accepting it. `.archex/index.db` and
    `.archex/settings.toml` are ordinary files, so a published repository
    can commit both, and a clone plus `git worktree add` would then install
    that content as a real index.

    The cache marker closes that door because it is keyed on
    `sha256("<resolved absolute path>@<commit>")`: a legitimate index has
    one because archex wrote it after building the index in that directory,
    and committed repository content cannot forge one for a clone directory
    nobody knows in advance. This is the same binding
    `CacheManager.get` already requires before reusing a project-layout
    index.
    """
    cache = CacheManager(cache_dir=str(checkout_root / ".archex"), project_layout=True)
    key = cache.cache_key(RepoSource(local_path=str(checkout_root)), head_override=commit_hash)
    return cache.marker_matches(key)


def _evaluate_candidate(
    candidate_path: Path,
    *,
    destination: CheckoutIdentity,
    destination_head: str,
    index_config: IndexConfig,
) -> tuple[SeedCandidate | None, SeedRejection | None]:
    """Verify one listed checkout and decide whether its index may seed us."""
    resolved = candidate_path.resolve()
    if resolved == destination.root:
        return None, None

    root_rejection = _reject_symlinked(candidate_path, "candidate root")
    if root_rejection is not None:
        return None, root_rejection

    identity = resolve_checkout_identity(resolved)
    if identity is None:
        return None, SeedRejection(
            root=resolved,
            reason="not_a_work_tree",
            detail="git rev-parse did not report a usable work tree",
        )
    if identity.common_dir != destination.common_dir:
        return None, SeedRejection(
            root=resolved,
            reason="different_repository",
            detail=(
                f"common directory {identity.common_dir} does not match {destination.common_dir}"
            ),
        )
    if identity.root != resolved:
        return None, SeedRejection(
            root=resolved,
            reason="not_a_checkout_root",
            detail=f"git reports its top level as {identity.root}",
        )

    project_dir = resolved / ".archex"
    index_path = project_dir / "index.db"
    for path, label in ((project_dir, "project directory"), (index_path, "index database")):
        rejection = _reject_symlinked(path, label)
        if rejection is not None:
            return None, rejection
    if not index_path.is_file():
        return None, SeedRejection(
            root=resolved, reason="no_index", detail=f"{index_path} does not exist"
        )
    size_bytes = index_path.stat().st_size
    if size_bytes > MAX_SEED_INDEX_BYTES:
        return None, SeedRejection(
            root=resolved,
            reason="index_too_large",
            detail=f"{size_bytes} bytes exceeds the {MAX_SEED_INDEX_BYTES} byte seed limit",
        )

    metadata = _read_store_metadata(index_path, _REQUIRED_METADATA_KEYS)
    if metadata is None:
        return None, SeedRejection(
            root=resolved, reason="unreadable_index", detail=f"could not read {index_path}"
        )
    if metadata.get("schema_version") != CURRENT_SCHEMA_VERSION:
        return None, SeedRejection(
            root=resolved,
            reason="incompatible_schema",
            detail=(
                f"index schema {metadata.get('schema_version')!r} != {CURRENT_SCHEMA_VERSION!r}"
            ),
        )
    if metadata.get("needs_reindex") == "true":
        return None, SeedRejection(
            root=resolved, reason="needs_reindex", detail="candidate index requires a re-index"
        )
    mismatch = index_config_metadata_mismatch(metadata, index_config)
    if mismatch is not None:
        return None, SeedRejection(
            root=resolved,
            reason="incompatible_index_config",
            detail=f"{mismatch} differs from the destination's index config",
        )
    commit_hash = metadata.get("commit_hash")
    if not commit_hash:
        return None, SeedRejection(
            root=resolved, reason="no_revision", detail="candidate index records no commit_hash"
        )

    try:
        indexed_at = float(metadata.get("indexed_at") or 0.0)
    except ValueError:
        indexed_at = 0.0

    if not _index_is_provenanced(resolved, commit_hash):
        return None, SeedRejection(
            root=resolved,
            reason="unprovenanced_index",
            detail=(
                "candidate index carries no cache marker written for "
                f"{resolved} at {commit_hash[:8]}"
            ),
        )

    return (
        SeedCandidate(
            root=resolved,
            index_path=index_path,
            commit_hash=commit_hash,
            indexed_at=indexed_at,
            same_commit=commit_hash == destination_head,
        ),
        None,
    )


def select_worktree_seed(
    repo_root: Path,
    *,
    destination_head: str,
    index_config: IndexConfig,
) -> SeedSelection:
    """Select an eligible seed source for the linked worktree at `repo_root`.

    `destination_head` is the destination's resolved HEAD; it only ranks
    candidates (an index already at the destination's revision needs the
    smallest delta) and is never used to establish identity.
    """
    unsupported = _unsupported_index_config(index_config)
    if unsupported is not None:
        return SeedSelection(candidate=None, reason="unsupported_index_config", detail=unsupported)

    destination = resolve_checkout_identity(repo_root)
    if destination is None:
        return SeedSelection(
            candidate=None,
            reason="not_a_work_tree",
            detail=f"{repo_root} is not a usable Git work tree",
        )
    if not destination.is_linked_worktree:
        return SeedSelection(
            candidate=None,
            reason="not_a_linked_worktree",
            detail=f"git directory {destination.git_dir} is the repository's common directory",
        )

    # `-z` (Git 2.36+) is the unambiguous framing; older Git rejects the
    # flag, in which case the line-oriented output is parsed instead.
    nul_terminated = True
    listing = _run_git(["worktree", "list", "--porcelain", "-z"], destination.root)
    if listing is None or listing.returncode != 0:
        nul_terminated = False
        listing = _run_git(["worktree", "list", "--porcelain"], destination.root)
    if listing is None or listing.returncode != 0:
        return SeedSelection(
            candidate=None,
            reason="worktree_list_failed",
            detail=(listing.stderr.strip() if listing is not None else "git could not be run"),
        )

    candidates: list[SeedCandidate] = []
    rejections: list[SeedRejection] = []
    listed = _parse_worktree_list(listing.stdout, nul_terminated=nul_terminated)
    for skipped in listed[MAX_SEED_CANDIDATES:]:
        rejections.append(
            SeedRejection(
                root=skipped,
                reason="candidate_limit_reached",
                detail=f"more than {MAX_SEED_CANDIDATES} checkouts share this repository",
            )
        )
    for candidate_path in listed[:MAX_SEED_CANDIDATES]:
        try:
            candidate, rejection = _evaluate_candidate(
                candidate_path,
                destination=destination,
                destination_head=destination_head,
                index_config=index_config,
            )
        except (OSError, sqlite3.Error) as exc:
            # A candidate is another checkout this process does not own: its
            # files can be removed or replaced while it is being examined,
            # and reading one must never be able to fail the indexing
            # command that merely asked whether a seed exists.
            candidate = None
            rejection = SeedRejection(
                root=candidate_path,
                reason="candidate_unreadable",
                detail=f"{type(exc).__name__}: {exc}",
            )
        if candidate is not None:
            candidates.append(candidate)
        elif rejection is not None:
            rejections.append(rejection)

    if not candidates:
        return SeedSelection(
            candidate=None,
            reason="no_eligible_seed",
            detail=f"{len(rejections)} candidate checkout(s) refused",
            rejections=tuple(rejections),
        )

    # An index already at the destination's revision needs the smallest
    # delta; among equals, the most recently measured one; path breaks ties
    # so selection is deterministic across runs.
    best = min(
        candidates,
        key=lambda candidate: (
            not candidate.same_commit,
            -candidate.indexed_at,
            str(candidate.root),
        ),
    )
    return SeedSelection(
        candidate=best,
        reason="eligible",
        detail=f"seed source {best.root} at {best.commit_hash[:8]}",
        rejections=tuple(rejections),
    )


#: Name of the sibling lock that serializes seeding for one destination.
SEED_LOCK_FILENAME = "index-seed.lock"

#: Prefix of a staging directory inside the destination project directory.
#: Staging lives beside the destination database so publication is a rename
#: on the same filesystem; a leftover directory means a seeding process died
#: and is removed by the next seeder while holding the lock.
_STAGING_PREFIX = ".seed-"

#: How long a seeding attempt waits for the seed lock. Seeding is an
#: optimization over a full index that costs seconds, so a contended lock
#: means another process is already doing this work and this one should get
#: on with ordinary indexing instead of queueing behind a whole copy.
_SEED_LOCK_TIMEOUT_SECONDS = 0.5

_SEED_DIAGNOSTICS = StateFileDiagnostics(
    lock_error="worktree-seed-lock-error",
    lock_timeout="worktree-seed-lock-timeout",
    write_error="worktree-seed-write-error",
)


@dataclass(frozen=True)
class WorktreeSeedResult:
    """Outcome of one seeding attempt for one destination worktree."""

    installed: bool
    """Whether the destination now holds an index published from a seed."""

    reason: str
    """`seeded` when installed, else the disposition that refused seeding."""

    detail: str = ""
    source_root: Path | None = None
    sync_strategy: str | None = None
    """`clean` or `delta` when installed; None otherwise."""

    files_changed: int = 0
    seed_time_ms: float = 0.0
    rejections: tuple[SeedRejection, ...] = field(default_factory=tuple)


def _snapshot_index(source_db: Path, dest_db: Path) -> None:
    """Copy a candidate index into `dest_db` through SQLite's backup API.

    A filesystem copy of a WAL-mode database without its write-ahead log can
    omit committed transactions, and copying the log is forbidden — the
    destination must never receive `-wal`/`-shm` state. `backup()` over a
    read-only connection instead produces one consistent snapshot that
    already includes whatever the log holds, while opening the source for
    reading only. (SQLite may materialize empty `-wal`/`-shm` sidecars
    beside the *source* for any WAL-mode reader; the source's own commands
    do the same, and its database bytes are untouched.)
    """
    uri = f"file:{quote(str(source_db))}?mode=ro"
    reader = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        writer = sqlite3.connect(dest_db)
        try:
            reader.backup(writer)
        finally:
            writer.close()
    finally:
        reader.close()


def _staged_copy_rejection(
    staged_db: Path,
    *,
    candidate: SeedCandidate,
    index_config: IndexConfig,
) -> str | None:
    """Re-validate the snapshot that will actually be installed, or None if sound.

    Eligibility was decided against the candidate's database *by path*, and
    the snapshot is taken from that path again later: between the two, the
    file can be replaced — by another process publishing its own index over
    it, or by anything else able to write there. Re-checking the staged
    bytes makes the validated object and the installed object the same
    object.

    The schema-object check is separate from that: archex's schema declares
    no view and no trigger, and both are the SQLite constructs that can
    carry SQL expressions evaluated during ordinary reads and schema
    migration. A snapshot that declares one was not written by this
    program, whatever its metadata claims.
    """
    metadata = _read_store_metadata(staged_db, _REQUIRED_METADATA_KEYS)
    if metadata is None:
        return "staged snapshot could not be read"
    if metadata.get("schema_version") != CURRENT_SCHEMA_VERSION:
        return f"staged snapshot schema {metadata.get('schema_version')!r} is not supported"
    if metadata.get("needs_reindex") == "true":
        return "staged snapshot requires a re-index"
    if metadata.get("commit_hash") != candidate.commit_hash:
        return (
            f"staged snapshot revision {metadata.get('commit_hash')!r} is not the "
            f"{candidate.commit_hash!r} that was validated"
        )
    mismatch = index_config_metadata_mismatch(metadata, index_config)
    if mismatch is not None:
        return f"staged snapshot {mismatch} does not match the destination's index config"

    conn = sqlite3.connect(f"file:{quote(str(staged_db))}?mode=ro", uri=True, timeout=5.0)
    try:
        unexpected = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'trigger')"
        ).fetchall()
    except sqlite3.Error as exc:
        return f"staged snapshot schema could not be read: {exc}"
    finally:
        conn.close()
    if unexpected:
        names = ", ".join(f"{row[0]} {row[1]}" for row in unexpected)
        return f"staged snapshot declares unexpected schema objects: {names}"
    return None


def _remove_stale_staging(project_dir: Path) -> None:
    """Delete staging directories left by a seeding process that died.

    Only ever called while holding the seed lock, so any staging directory
    present belongs to a process that is no longer running.
    """
    for path in project_dir.glob(f"{_STAGING_PREFIX}*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)


def _stamp_destination_identity(
    store: IndexStore,
    *,
    repo_root: Path,
    source_identity: str,
    destination_head: str,
    config: Config,
    index_config: IndexConfig,
) -> None:
    """Make a seeded store describe the destination rather than its source.

    A copied index still carries the source checkout's identity metadata.
    Left that way the destination's own cache lookup would reject it and
    re-index from scratch on the very next command, so the seed would buy
    nothing. The artifact-import path needs exactly the same treatment, so
    the fields live in `archex.index.adopt`.
    """
    stamp_project_index_identity(
        store,
        repo_root=repo_root,
        source_identity=source_identity,
        commit_hash=destination_head,
        config=config,
        index_config=index_config,
    )


@dataclass(frozen=True)
class _StagedSync:
    """Result of delta-syncing a staged seed against the destination tree."""

    strategy: str
    """`clean`, `delta`, or `too_stale` (nothing publishable)."""

    files_changed: int


def _sync_staged_seed(
    staged_db: Path,
    *,
    repo_root: Path,
    source_identity: str,
    destination_head: str,
    config: Config,
    index_config: IndexConfig,
) -> _StagedSync:
    """Delta-sync a staged seed to the destination tree.

    `strategy` is `clean` or `delta` when the staged copy is publishable,
    and `too_stale` when the destination has drifted at or past
    `config.delta_threshold` — the same knob ordinary delta indexing and
    artifact import use. Past that point the staged copy is worth less than
    a fresh build, so nothing is published and the caller falls back to full
    indexing; `files_changed` is still reported so the disposition can say
    how far the tree had moved.
    """
    from archex.acquire import discover_files
    from archex.index.delta import apply_delta, compute_working_tree_delta
    from archex.index.graph import DependencyGraph

    store = IndexStore(staged_db)
    try:
        manifest = compute_working_tree_delta(repo_root, store, config)
        manifest.base_commit = store.get_metadata("commit_hash") or manifest.base_commit
        manifest.current_commit = destination_head

        strategy = "clean"
        files_changed = len(manifest.changes)
        if manifest.changes:
            total_files = len(
                discover_files(
                    repo_root,
                    languages=config.languages,
                    max_file_size=config.max_file_size,
                ).files
            )
            change_ratio = files_changed / total_files if total_files > 0 else 1.0
            if change_ratio >= config.delta_threshold:
                logger.info(
                    "Worktree seed is %.0f%% stale (delta_threshold=%.0f%%) — "
                    "falling back to a full index.",
                    change_ratio * 100,
                    config.delta_threshold * 100,
                )
                return _StagedSync(strategy="too_stale", files_changed=files_changed)
            graph = DependencyGraph.from_edges(store.get_edges())
            apply_delta(store, graph, manifest, repo_root, config, index_config)
            strategy = "delta"

        _stamp_destination_identity(
            store,
            repo_root=repo_root,
            source_identity=source_identity,
            destination_head=destination_head,
            config=config,
            index_config=index_config,
        )
        return _StagedSync(strategy=strategy, files_changed=files_changed)
    finally:
        store.close()


def seed_worktree_index(
    repo_root: Path,
    *,
    cache: CacheManager,
    cache_key: str,
    source_identity: str,
    config: Config,
    index_config: IndexConfig,
) -> WorktreeSeedResult:
    """Bootstrap a linked worktree's index from a compatible same-repository seed.

    Never raises for a seeding failure: seeding is an optimization over the
    ordinary index path, so every refusal and every recoverable error
    returns a named disposition with `installed=False` and the caller
    proceeds to index normally. Publication happens once, through
    `CacheManager.put`, which copies into a sibling temp file, renames it
    over the destination database, and writes the `index.meta` validity
    marker last — so an interrupted seed leaves either the previous state or
    nothing at all, never a half-installed index.
    """
    started = time.perf_counter()
    destination_db = cache.db_path(cache_key)
    project_dir = destination_db.parent

    if destination_db.exists():
        return WorktreeSeedResult(
            installed=False,
            reason="destination_index_present",
            detail=f"{destination_db} already exists",
        )
    if project_dir.is_symlink() or not project_dir.is_dir():
        return WorktreeSeedResult(
            installed=False,
            reason="destination_unusable",
            detail=f"{project_dir} is not a usable project directory",
        )

    destination_head = CacheManager.git_head(str(repo_root))
    if not destination_head:
        return WorktreeSeedResult(
            installed=False,
            reason="no_destination_revision",
            detail=f"could not resolve HEAD for {repo_root}",
        )

    try:
        selection = select_worktree_seed(
            repo_root, destination_head=destination_head, index_config=index_config
        )
    except (OSError, sqlite3.Error, ArchexError) as exc:
        # Selection reads directories and databases this process does not
        # own. This function promises never to fail the indexing command
        # that asked it for a seed, so the promise has to cover looking for
        # one as well as installing it.
        logger.warning(
            "Looking for a worktree seed in %s failed (%s) — indexing normally.",
            repo_root,
            exc,
        )
        return WorktreeSeedResult(
            installed=False,
            reason="seed_failed",
            detail=f"{type(exc).__name__}: {exc}",
            seed_time_ms=round((time.perf_counter() - started) * 1000, 1),
        )
    candidate = selection.candidate
    if candidate is None:
        return WorktreeSeedResult(
            installed=False,
            reason=selection.reason,
            detail=selection.detail,
            rejections=selection.rejections,
        )

    lock = ExclusiveLock(
        project_dir / SEED_LOCK_FILENAME,
        timeout=_SEED_LOCK_TIMEOUT_SECONDS,
        cwd=repo_root,
        diagnostics=_SEED_DIAGNOSTICS,
    )
    with lock as acquired:
        if not acquired:
            return WorktreeSeedResult(
                installed=False,
                reason="seed_in_progress",
                detail="another process holds the worktree seed lock",
                source_root=candidate.root,
                rejections=selection.rejections,
            )
        if destination_db.exists():
            return WorktreeSeedResult(
                installed=False,
                reason="destination_index_present",
                detail=f"{destination_db} was created while waiting for the seed lock",
                source_root=candidate.root,
            )

        _remove_stale_staging(project_dir)
        staging = project_dir / f"{_STAGING_PREFIX}{os.getpid()}-{time.time_ns()}"
        try:
            staging.mkdir(parents=True)
            staged_db = staging / "index.db"
            _snapshot_index(candidate.index_path, staged_db)
            staged_rejection = _staged_copy_rejection(
                staged_db, candidate=candidate, index_config=index_config
            )
            if staged_rejection is not None:
                return WorktreeSeedResult(
                    installed=False,
                    reason="staged_copy_rejected",
                    detail=staged_rejection,
                    source_root=candidate.root,
                    seed_time_ms=round((time.perf_counter() - started) * 1000, 1),
                    rejections=selection.rejections,
                )
            synced = _sync_staged_seed(
                staged_db,
                repo_root=repo_root,
                source_identity=source_identity,
                destination_head=destination_head,
                config=config,
                index_config=index_config,
            )
            if synced.strategy == "too_stale":
                return WorktreeSeedResult(
                    installed=False,
                    reason="large_delta",
                    detail=(
                        f"{synced.files_changed} file(s) changed, at or past "
                        f"delta_threshold {config.delta_threshold}"
                    ),
                    source_root=candidate.root,
                    files_changed=synced.files_changed,
                    seed_time_ms=round((time.perf_counter() - started) * 1000, 1),
                    rejections=selection.rejections,
                )
            cache.put(
                cache_key,
                staged_db,
                resolved_commit=destination_head,
                source_identity=source_identity,
            )
        except (OSError, sqlite3.Error, ArchexError) as exc:
            logger.warning(
                "Worktree seeding from %s failed (%s) — falling back to a full index.",
                candidate.root,
                exc,
            )
            return WorktreeSeedResult(
                installed=False,
                reason="seed_failed",
                detail=f"{type(exc).__name__}: {exc}",
                source_root=candidate.root,
                seed_time_ms=round((time.perf_counter() - started) * 1000, 1),
                rejections=selection.rejections,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # Installing another checkout's index is a cross-directory data import,
    # so it is recorded here rather than only through the caller's optional
    # timing object — a library or MCP caller that passes none would
    # otherwise have it happen with no trace at all.
    logger.info(
        "Seeded %s from %s at %s (%s sync, %d file(s) changed, %.0fms)",
        repo_root,
        candidate.root,
        candidate.commit_hash[:8],
        synced.strategy,
        synced.files_changed,
        (time.perf_counter() - started) * 1000,
    )
    return WorktreeSeedResult(
        installed=True,
        reason="seeded",
        detail=f"seeded from {candidate.root} at {candidate.commit_hash[:8]}",
        source_root=candidate.root,
        sync_strategy=synced.strategy,
        files_changed=synced.files_changed,
        seed_time_ms=round((time.perf_counter() - started) * 1000, 1),
        rejections=selection.rejections,
    )
