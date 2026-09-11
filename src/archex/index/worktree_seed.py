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

Selection stops at eligibility. Copying, delta synchronization, and
installation are a separate concern.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from archex.cache import CacheManager
from archex.index.compat import INDEX_CONFIG_METADATA_KEYS, index_config_metadata_mismatch
from archex.index.store import CURRENT_SCHEMA_VERSION
from archex.models import RepoSource

if TYPE_CHECKING:
    from archex.models import IndexConfig

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
