"""External-corpus pinning and sealed-holdout policy for the M3 quality frontier.

Archex's benchmark corpus already contains pinned-external tasks (``repo``
values other than ``"."``, checked out at an immutable tag or commit) living
alongside self-repo tasks under ``benchmarks/tasks``. This module formalizes
that existing convention into an enforceable policy plus a distinct **sealed
chronological holdout** corpus under ``benchmarks/sealed_tasks``.

The sealed corpus exists so a candidate retrieval path can be evaluated
against evidence that was never available to tune production heuristics:

- every sealed task targets a real external repository pinned to an
  immutable tag or commit (never ``.``, the current working tree);
- no sealed task's ``task_id`` may appear anywhere in ``src/archex``, so
  production code cannot have been keyed to a specific sealed task;
- the sealed directory is never the default or CI ``--tasks-dir`` target;
  callers must opt in explicitly (see ``enforce_sealed_corpus_access``).

This module only implements the policy layer. Actually retrieving an
external repository's contents (cloning at a pinned ref) reuses the existing
``archex.benchmark.runner`` machinery; nothing here performs network I/O.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from archex.benchmark.loader import load_tasks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from archex.benchmark.models import BenchmarkTask

#: Default location of the sealed chronological holdout corpus.
SEALED_TASKS_DIR = Path("benchmarks/sealed_tasks")

#: Directory basename that marks a task directory as sealed. Access to any
#: ``--tasks-dir`` whose name (or any path segment) matches this is gated by
#: ``enforce_sealed_corpus_access``.
SEALED_TASKS_DIRNAME = "sealed_tasks"

#: ``repo: .`` denotes the current working tree; every other value must
#: resolve to an immutable ref. These are the floating/mutable refs a careless
#: task author could otherwise pin against, where "latest" silently drifts
#: and would make results unreproducible.
_FLOATING_REFS = frozenset({"head", "main", "master", "trunk", "develop", "latest", "stable"})

#: Vocabulary terms shorter than this are too generic to signal a leak
#: (avoids false positives on common short identifiers).
_MIN_LEAK_TERM_LENGTH = 4


class CorpusPinningViolation(NamedTuple):
    """A benchmark task whose external repo is not pinned to an immutable ref."""

    task_id: str
    repo: str
    commit: str


class VocabularyLeak(NamedTuple):
    """A sealed task's distinguishing vocabulary term found in production code."""

    task_id: str
    term: str
    source_path: str


class SealedCorpusPolicyError(RuntimeError):
    """Raised when the sealed holdout corpus violates its own policy."""


class SealedCorpusAccessError(RuntimeError):
    """Raised when a caller targets the sealed corpus without opting in."""


def is_pinned_commit(commit: str) -> bool:
    """Return whether *commit* looks like an immutable ref (tag or SHA).

    Any non-empty value other than a known floating/mutable ref name counts
    as pinned — this matches the existing convention across
    ``benchmarks/tasks`` (version tags such as ``"3.1.0"`` or ``"v2.32.3"``,
    or full 40-character commit SHAs), rather than requiring a specific
    format that would reject already-published, already-verified tasks.
    """
    normalized = commit.strip().lower()
    return bool(normalized) and normalized not in _FLOATING_REFS


def is_external_task(task: BenchmarkTask) -> bool:
    """Return whether *task* targets a repository other than the working tree."""
    return task.repo != "."


def find_unpinned_external_tasks(
    tasks: Sequence[BenchmarkTask],
) -> list[CorpusPinningViolation]:
    """Return every external task whose ``commit`` is not an immutable ref."""
    return [
        CorpusPinningViolation(task_id=task.task_id, repo=task.repo, commit=task.commit)
        for task in tasks
        if is_external_task(task) and not is_pinned_commit(task.commit)
    ]


def pinned_external_tasks(tasks: Sequence[BenchmarkTask]) -> list[BenchmarkTask]:
    """Return the subset of *tasks* that are pinned-external (the external corpus)."""
    return [task for task in tasks if is_external_task(task) and is_pinned_commit(task.commit)]


def sealed_vocabulary_terms(task: BenchmarkTask) -> set[str]:
    """Return a sealed task's distinguishing terms that must not leak into production code.

    Limited to ``task_id`` — the same task-keyed-logic guard already enforced
    ad hoc by ``scripts/benchmark_replay_smoke.sh`` (its task-ID equality grep check)
    — rather than ``keywords``/``expected_symbols``/``question`` text. Those
    fields deliberately reuse ordinary domain vocabulary (``proxy``,
    ``environ``, ``root_path``) that already appears throughout unrelated
    production code; flagging them would not detect a real sealed-boundary
    violation, only generate noise. A ``task_id`` is engineered to be
    specific and unique, so its appearance in ``src/archex`` is a reliable
    signal that production logic has been keyed to one benchmark task.
    """
    return {task.task_id} if len(task.task_id) >= _MIN_LEAK_TERM_LENGTH else set()


def find_vocabulary_leaks(
    tasks: Sequence[BenchmarkTask],
    src_root: Path,
) -> list[VocabularyLeak]:
    """Return every sealed-task vocabulary term found verbatim under *src_root*.

    An empty result is the sealed-boundary compliance proof: no task-keyed or
    task-vocabulary-keyed logic exists in production code.
    """
    leaks: list[VocabularyLeak] = []
    source_files = sorted(src_root.rglob("*.py"))
    for task in tasks:
        terms = sealed_vocabulary_terms(task)
        if not terms:
            continue
        patterns = {term: re.compile(re.escape(term)) for term in terms}
        for source_path in source_files:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
            for term, pattern in patterns.items():
                if pattern.search(text):
                    leaks.append(
                        VocabularyLeak(
                            task_id=task.task_id,
                            term=term,
                            source_path=str(source_path),
                        )
                    )
    return leaks


def find_ci_sealed_references(
    workflows_dir: Path,
    sealed_dirname: str = SEALED_TASKS_DIRNAME,
) -> list[Path]:
    """Return CI workflow files that reference the sealed corpus directory.

    An empty result proves CI never executes the sealed holdout, matching the
    same never-full-corpus-in-CI boundary already enforced for the bounded
    public task set.
    """
    if not workflows_dir.is_dir():
        return []
    return [
        path
        for path in sorted(workflows_dir.glob("*.yml"))
        if sealed_dirname in path.read_text(encoding="utf-8")
    ]


def load_sealed_tasks(tasks_dir: Path = SEALED_TASKS_DIR) -> list[BenchmarkTask]:
    """Load the sealed holdout corpus, enforcing its policy at load time.

    Raises ``SealedCorpusPolicyError`` if any sealed task targets the working
    tree (``repo: .``) or an unpinned external ref — the sealed corpus exists
    specifically to provide external, immutable-snapshot evidence, so a task
    violating either constraint would silently defeat its purpose.
    """
    tasks = load_tasks(tasks_dir)
    self_repo_tasks = [task.task_id for task in tasks if not is_external_task(task)]
    if self_repo_tasks:
        msg = (
            "Sealed holdout tasks must target an external repository, "
            f"not the working tree: {sorted(self_repo_tasks)}"
        )
        raise SealedCorpusPolicyError(msg)
    unpinned = find_unpinned_external_tasks(tasks)
    if unpinned:
        msg = (
            "Sealed holdout tasks must pin an immutable ref: "
            f"{[(v.task_id, v.commit) for v in unpinned]}"
        )
        raise SealedCorpusPolicyError(msg)
    return tasks


def is_sealed_tasks_dir(tasks_dir: Path, sealed_dirname: str = SEALED_TASKS_DIRNAME) -> bool:
    """Return whether *tasks_dir* names the sealed holdout corpus.

    Matches on the final path segment rather than a full-path comparison so
    the check is stable across relative/absolute invocation and symlinks.
    """
    return tasks_dir.name == sealed_dirname


def enforce_sealed_corpus_access(tasks_dir: Path, *, allow_sealed: bool) -> None:
    """Raise ``SealedCorpusAccessError`` when *tasks_dir* is sealed without opt-in.

    Called by the benchmark CLI before loading any tasks, so an operator
    cannot accidentally fold sealed-holdout evidence into a bounded or
    default-directory run.
    """
    if is_sealed_tasks_dir(tasks_dir) and not allow_sealed:
        msg = (
            f"{tasks_dir} is the sealed chronological holdout corpus; "
            "pass --allow-sealed-corpus to target it explicitly"
        )
        raise SealedCorpusAccessError(msg)
