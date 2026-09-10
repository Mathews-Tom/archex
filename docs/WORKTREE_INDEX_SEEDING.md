# Worktree Index Seeding

A linked Git worktree (`git worktree add`) starts with no `.archex/index.db`.
Left alone it pays a full cold index for a tree that is usually a few commits
away from a checkout that is already indexed on the same machine — the exact
cost an agent or stacked-PR workflow pays every time it opens a new worktree.

`archex index`, `archex init`, and `archex query` therefore offer a fresh
linked worktree a **seed**: a compatible index from a checkout sharing the
same Git *common directory*, copied locally, synchronized against the
destination tree, and published atomically. Nothing about the retrieval
result changes — a seeded index and an independently built one are the same
index (see [Equivalence](#equivalence)).

Seeding never applies to a clone. It is strictly same-repository, same
machine, and same filesystem. For a fresh clone or a teammate's machine, use
the [portable index artifact](PORTABLE_INDEX_ARTIFACT.md), which is validated
and versioned for travel.

## What the CLI reports

```console
$ cd /path/to/linked-worktree
$ archex init --no-index
$ archex index
Indexed repository: /path/to/linked-worktree
Index path:         /path/to/linked-worktree/.archex/index.db
Commit:             344c97f9f21619d34182751072c74fb69ff8f70b
Strategy:           seeded
Worktree seed:      seeded from /path/to/main-checkout (delta, 1 file(s) synchronized in 1615.9 ms)
Files indexed:      1176
Chunks indexed:     16664
```

`archex index --format json` adds `seed_disposition`, `seed_source`,
`seed_strategy`, `seed_files_changed`, and `seed_time_ms`. Those keys appear
**only when a seed was actually considered** — an ordinary checkout, which can
never be seeded, reports nothing at all rather than a permanently-null
section.

A refusal is reported as explicitly as a success:

```console
Strategy:           full
Worktree seed:      not used (no_eligible_seed); indexed normally
```

## Eligibility

A checkout may seed this worktree only if all of the following hold.

| Check | Rule |
|---|---|
| Destination shape | The destination is a linked worktree: its `--git-dir` differs from its `--git-common-dir`. |
| Same repository | The candidate's own `--git-common-dir` resolves to the destination's. |
| Candidate shape | The candidate reports itself as a non-bare work tree whose top level is the candidate path. |
| No symlinks | Neither the candidate root, its `.archex`, nor its `index.db` may be a symlink. |
| Schema | The candidate index's `schema_version` equals this build's. |
| Health | The candidate index is not flagged `needs_reindex` and records a `commit_hash`. |
| Index config | Chunker, chunker revision, quantization, and all four evidence-provider lists match the destination's config. |
| Retrieval config | `vector` and `splade` must be disabled (see [Refused configurations](#refused-configurations)). |
| Provenance | The candidate carries archex's own `.archex/index.meta` marker, written for that checkout's absolute path at the index's recorded revision. |
| Size | The candidate index is at most 1 GiB. |
| Candidate count | At most 32 checkouts are interrogated; the rest are refused by name. |

Among eligible candidates, one already at the destination's revision wins
(its delta is smallest), then the most recently measured, then path order so
selection is deterministic.

### Why identity is proven twice

`git worktree list --porcelain` is a starting point, not evidence. A stale or
hand-edited `.git/worktrees/<name>/gitdir` makes it report a checkout that
belongs to a **different repository**, and seeding from that would serve
another repository's code as this one's. Every candidate is therefore
re-interrogated from its own directory, and only the common directory it
reports itself decides eligibility.

Only `git rev-parse` is ever run inside a candidate. `rev-parse` invokes no
hook, no `core.fsmonitor` program, and no filter or diff driver, so identity
is established without executing repository-configured code. The commands
that can — `git status`, `git ls-files` — run only against the destination
working tree, which the caller already owns.

### Why the index's own metadata is not enough

Everything a candidate declares about itself — schema version, revision,
chunker, provider lists — is stored *inside* the candidate database, so
whoever supplies that database supplies the evidence for accepting it. And
`.archex/index.db` and `.archex/settings.toml` are ordinary files: a
published repository can commit both, and a clone followed by
`git worktree add` would then install committed content as this worktree's
real index.

The cache marker closes that door. `index.meta` records
`sha256("<resolved absolute path>@<commit>")`, so a legitimate index has one
because archex wrote it after building the index *in that directory*, and
committed content cannot forge one for a clone directory nobody knows in
advance. It is the same binding `CacheManager.get` already requires before
reusing a project-layout index. A candidate without a matching marker is
refused as `unprovenanced_index`.

The snapshot is then re-validated after it is taken: eligibility is decided
against a *path*, and the file at that path can be replaced before the copy
runs, so the staged bytes' schema version, revision, health flag, and index
config are checked again, and a snapshot declaring a view or trigger —
constructs archex's schema never creates — is refused outright. The
validated object and the installed object are therefore the same object.

### Shapes that are refused structurally

Submodules, bare repositories, and `git init --separate-git-dir` checkouts
are not refused by name. In all three shapes Git reports the Git directory
and the common directory as *equal*, so they never satisfy the
linked-worktree test in the first place:

| Shape | `.git` | `--git-dir` vs `--git-common-dir` |
|---|---|---|
| Ordinary checkout | directory | equal |
| Linked worktree | file → `<common>/worktrees/<name>` | **differ** |
| Submodule | file → `<super>/.git/modules/<path>` | equal |
| `--separate-git-dir` | file → external directory | equal |
| Bare repository | absent | no work tree at all |

### Refused configurations

Seeding refuses a destination whose index config enables `vector` or
`splade`. Keeping embedding state correct across a delta needs the embedder
pipeline that ordinary delta indexing owns, and SPLADE tables have no
validated transfer story; copying either without that would be exactly the
silent incompatibility this path exists to avoid. Both are disabled by
default and in `.archex/settings.toml`.

## What is copied

Only index content, and only through SQLite's backup API over a **read-only**
connection to the candidate database.

A filesystem copy of a WAL-mode database without its write-ahead log can omit
committed transactions, and copying the log is forbidden — the destination
must never receive `-wal`/`-shm` state. The backup produces one consistent
snapshot that already includes whatever the log holds, while never opening the
source for writing.

Nothing else is read at all: no lock file, no WAL/SHM sidecar, no status
snapshot, no post-edit state, no session ledger, no metrics or trace
directory, and no `settings.toml`. The destination keeps its own
configuration. (SQLite may materialize empty `-wal`/`-shm` sidecars beside the
*source* database, as it does for any WAL-mode reader, including the source
checkout's own commands. The source's database bytes are never modified.)

## Installation order

```text
stage  ->  snapshot  ->  delta-sync in staging  ->  publish  ->  marker
```

Everything expensive happens before publication. The snapshot lands in a
staging directory beside the destination database, is synchronized there
against the destination working tree, and is only then published through the
same `CacheManager.put` the ordinary full-index path uses: a copy into a
sibling temp file, a rename over the destination database, and the
`index.meta` validity marker written **last**.

A crash or failure at any earlier point therefore leaves the destination
exactly as it was, and staging left behind by a process that died is
reclaimed by the next seeder. The one window that is not atomic is between
the database rename and the marker write: an interruption there leaves a
complete database with no marker, which the cache treats as absent (so it
is never served) but which seeding will not overwrite either — that
worktree then indexes normally until `archex reset --force` clears it.

Concurrent seeders serialize on a short-timeout `flock` on
`.archex/index-seed.lock`. A contended lock is a disposition, not a queue:
seeding saves seconds, so waiting behind another process's whole copy would
defeat the point.

## Synchronization and the staleness fallback

Synchronization reuses the same content-hash delta machinery ordinary delta
indexing and artifact import use — `compute_working_tree_delta()` against the
destination tree, then `apply_delta()`:

1. Nothing changed → `seed_strategy: "clean"`.
2. Change ratio below `config.delta_threshold` (default `0.5`) → targeted
   `apply_delta()`, `seed_strategy: "delta"`.
3. Change ratio at or above the threshold → **nothing is published**. The
   disposition is `large_delta` with the changed-file count, and ordinary
   full indexing runs in its place. Past that point a targeted delta costs
   more than a fresh build.

A seeded store finishes by describing the destination — revision, source
identity, working-tree signature, generation id — and publishing
`index.meta` for the destination's own cache key. That is the same end state
a full index leaves, and it is what makes the seed *stick*: a copied index
still carrying its source's identity would be discarded by the destination's
own cache lookup on the very next command.

## Dispositions

Every outcome is named. `seeded` is the only one that installs anything;
every other falls through to ordinary indexing.

| Disposition | Meaning |
|---|---|
| `seeded` | A seed was installed and synchronized. |
| `seed_discarded` | A seed was installed but the resolution declined to reuse it (the tree moved under it); the reported strategy is what indexing actually did. |
| `not_a_linked_worktree` | The destination is an ordinary, submodule, or separate-git-dir checkout. |
| `not_a_work_tree` | Git reported no usable work tree. |
| `unsupported_index_config` | `vector` or `splade` is enabled. |
| `no_destination_revision` | The destination's HEAD could not be resolved. |
| `destination_unusable` | The destination's `.archex` is a symlink or not a directory. |
| `worktree_list_failed` | `git worktree list` failed. |
| `no_eligible_seed` | Candidates existed but all were refused; each refusal carries its own reason (`different_repository`, `not_a_checkout_root`, `not_a_work_tree`, `symlinked_path`, `no_index`, `index_too_large`, `unreadable_index`, `incompatible_schema`, `needs_reindex`, `incompatible_index_config`, `no_revision`, `unprovenanced_index`, `candidate_limit_reached`). |
| `large_delta` | A seed was found but the destination had drifted past `delta_threshold`. |
| `staged_copy_rejected` | The snapshot that would have been installed no longer matched what was validated, or declared schema objects archex does not write. |
| `seed_in_progress` | Another process held the seed lock. |
| `destination_index_present` | The destination already had an index. |
| `seed_failed` | A recoverable error occurred; the destination was left untouched. |

## Turning it off

```toml
# .archex/settings.toml
[index]
worktree_seed = false
```

or per invocation:

```console
$ ARCHEX_WORKTREE_SEED=0 archex index
```

Either makes every new worktree index from scratch.

## Equivalence

Measured on archex's own repository (1,176 indexed files, 16,664 chunks) with
a `0.29.0` wheel installed into a fresh virtualenv, driving real linked
worktrees created with `git worktree add --detach`:

| Arm | Strategy | Wall time | Index phase | Seed phase |
|---|---|---|---|---|
| Seeded, 3 files drifted | `seeded` / `delta` | 2.89 s | 2,326 ms | 2,234 ms |
| Seeded, 18 files drifted | `seeded` / `delta` | 4.09 s | 3,438 ms | 3,337 ms |
| Seeded, tree unchanged | `seeded` / `clean` | 2.11 s | 1,501 ms | 1,445 ms |
| Full index (`ARCHEX_WORKTREE_SEED=0`) | `full` | 8.86–15.61 s | 8,289–14,996 ms | — |

A ready-to-query worktree in **2.1–4.1 s** against **8.9–15.6 s** for a full
index of the same tree — the full-index range is filesystem-cache variance
across runs, measured on the same machine and wheel. A second
`archex index` in a seeded worktree reports `cached` in 74 ms, so no parse
happens again.

What "the same index" means, precisely, because the two comparisons differ:

**Against the same worktree delta-indexed in place — exact.** Seeding one
worktree from a checkout indexed at an earlier commit produces a store
byte-identical to letting that checkout index itself forward to the same
tree: identical chunk rows *including content* (16,664), identical edges
*including evidence* (3,659), identical `file_states` (1,194), and identical
metadata apart from `indexed_at` and `source_identity`. This is the
load-bearing equivalence, because a seeded worktree is exactly a worktree
whose index arrived by delta.

**Against a fresh full build — exact when the seed synced `clean`.** The
`clean` arm and an independent full index agree on chunk ids, `file_states`,
`generation_id`
(`371237cc7b6efab2b1de623b96105ba996fb05420aed3b29fee2df21b6c23c7b`), and on
the files `archex query` returns. `archex status --cached` reports `fresh` in
both.

**Against a fresh full build after a `delta` sync — same corpus, possibly
different tail ranking.** Corpus, chunk ids, counts and `generation_id` still
agree, but `apply_delta` records different edge *evidence* for the files it
reparses than a full parse does (`[]` where a full build records
`["resolved import …"]`), and it does not re-add every unresolved-import
edge. Structural scoring reads those edges, so a query *can* return a
different file at the bottom of a bounded result set — observed once, with
18 files drifted; with 3 files drifted the seeded and full builds returned
identical rows. This is a property of delta indexing, not of seeding: an
ordinary `archex index` after a one-line edit produces the same divergence
from a fresh full build, which is why the meaningful comparison is the first
one above. The only metadata difference
is `delta_applied`, which honestly records that the store reached its state
through a delta.

Reproduce with:

```console
$ git worktree add --detach /tmp/wt-seeded <commit>
$ cd /tmp/wt-seeded && archex init --no-index && archex index --format json

$ git worktree add --detach /tmp/wt-clean <commit>
$ cd /tmp/wt-clean && archex init --no-index \
    && ARCHEX_WORKTREE_SEED=0 archex index --format json
```
