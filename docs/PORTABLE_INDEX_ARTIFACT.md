# Portable Index Artifact

`archex index --export-artifact <path>` writes a compacted, compressed,
versioned snapshot of a repo's `.archex/index.db`. A teammate (or CI) on a
fresh clone runs `archex init --from-artifact <path>` to bootstrap their
local index from that snapshot instead of paying a full cold-start reindex,
then delta-syncs to their current working tree.

This mirrors the highest-leverage team-workflow feature observed in
`DeusData/codebase-memory-mcp`'s committed `.codebase-memory/graph.db.zst`
artifact, adapted to archex's SQLite + FTS5 index and stdlib-only dependency
policy (`lzma`, not `zstd`).

## Format

An artifact is **not** a bare `.xz` file — it is a small custom frame around
an LZMA-compressed payload, so a reader can validate compatibility before
paying the cost of decompressing a potentially large index:

```text
+-----------------------------+
| MAGIC (12 bytes)            |  b"ARCHEXIDXv1\n"
+-----------------------------+
| header_len (4 bytes, u32be) |
+-----------------------------+
| header (UTF-8 JSON)         |
+-----------------------------+
| payload (LZMA stream)       |  VACUUM INTO-compacted SQLite database,
|                             |  FTS5 tables stripped
+-----------------------------+
```

### Header fields

| Field | Meaning |
|---|---|
| `format_version` | Container format version (framing + payload shape). |
| `created_by_version` | The `archex` version that produced the artifact. |
| `compat_min_version` / `compat_max_version` | Inclusive `archex` version range declared able to import this artifact. |
| `index_revision` | The `commit_hash` the index reflects at export time — the base revision `init --from-artifact` delta-syncs from. |
| `schema_version` | The store's SQLite schema version at export time (informational). |
| `chunk_count` / `file_count` | Corpus size at export time (informational, shown in CLI output). |
| `created_at` | Unix timestamp of export. |

The header is read and validated in full **before** the payload is
decompressed. An artifact whose `format_version` or `archex` compat range
this build does not support raises `ArtifactVersionError` immediately —
never a silent or partial import.

### What gets stripped

`chunks_fts` and `symbols_fts` (both FTS5 virtual tables) are dropped from
the compacted copy before compression and rebuilt locally on import. They
are fully derivable from the `chunks` table; shipping them would duplicate
the corpus text in the artifact and bake in a build-specific FTS5 binary
shape that has no reason to travel between machines. `VACUUM INTO` compacts
the copy; dropping the FTS5 tables and running a second `VACUUM` reclaims
the space they occupied.

## Compatibility ranges

`ARTIFACT_FORMAT_VERSION` (in `src/archex/index/artifact.py`) versions the
container itself. `ARTIFACT_MIN_COMPAT_VERSION` / `ARTIFACT_MAX_COMPAT_VERSION`
declare which `archex` versions this build's format is compatible with.
Both checks run on import:

1. The artifact's `format_version` must equal this build's
   `ARTIFACT_FORMAT_VERSION` — a future breaking format change bumps this
   constant, and older archex builds refuse the new artifact shape outright.
2. The *running* `archex.__version__` must fall within the artifact's own
   `compat_min_version..compat_max_version` range — an artifact exported by
   a version outside what this build declares itself compatible with is
   rejected.

Version comparison is a minimal dotted-integer-tuple comparator (not a full
PEP 440/semver parser) — sufficient for archex's plain `MAJOR.MINOR.PATCH`
versioning and avoids adding a dependency for it.

## Delta-sync after import and the staleness fallback

`archex init --from-artifact` does not leave the imported store pinned at
the artifact's export-time revision. Immediately after import, it runs
`sync_imported_artifact()`:

1. `compute_working_tree_delta()` compares the artifact's imported
   `file_states` (content hashes) against the current working tree —
   whatever commit that happens to be at, since the comparison is
   content-hash-based rather than a git commit diff. This is what makes it
   correct even when the artifact's recorded revision is unreachable in a
   shallow clone.
2. If nothing changed, syncing is a no-op (`strategy: "clean"`).
3. If the change ratio is **below** `config.delta_threshold` (default
   `0.5`, the same knob the ordinary delta-indexing path already exposes),
   a targeted `apply_delta()` updates only the affected chunks, edges, and
   FTS rows — proportional to the delta, not the repo.
4. If the change ratio is **at or above** `config.delta_threshold`, the
   imported store is discarded and an ordinary full re-index runs in its
   place instead, logged as a loud warning (`strategy: "full_reindex"`).
   Past that point a targeted delta costs more than starting fresh, and a
   loud, deterministic fallback is safer than silently syncing an index
   that is technically "delta-applied" but has drifted so far it no longer
   resembles the artifact it started from.

`archex init --from-artifact` reports the chosen strategy, files changed,
and sync time.

## `.gitattributes` management

Committing a binary artifact into git risks merge conflicts every time two
branches both regenerate it. `archex index --export-artifact <path>`
auto-manages a `merge=ours -diff` entry in `.gitattributes` for the
artifact's repo-relative path whenever it is exported *inside* the repo (an
artifact exported elsewhere, e.g. `/tmp`, has no repo-relative path to
attach a strategy to, and is skipped):

```gitattributes
# archex portable index artifact — never diff/merge-conflict
.archex-artifact.xz merge=ours -diff
```

This is written to disk only — never staged or committed automatically.
Committing it (alongside the artifact itself) is a deliberate, separate
decision.

`merge=ours` names a custom merge driver git must be told about via the
`merge.ours.driver` git config key — deliberately **local, non-shareable**
config, by git's own design (a committed config that redefines a merge
driver could otherwise let a malicious repo silently turn every merge into
a no-op). Export best-effort registers it on the exporting machine
(`git config merge.ours.driver true`), but every other clone must run that
same one-line command once:

```console
$ git config merge.ours.driver true
```

Without it, `merge=ours` is a no-op attribute and git falls back to its
normal (conflict-prone) binary merge behavior for the artifact path.

## Wall-time evidence

Measured on archex's own repository as the benchmark fixture — the same
self-referential `repo: "."`, pinned-commit-pair methodology
`benchmarks/delta_tasks/archex_delta_large.yaml` already uses for the M2
delta-indexing benchmark (`base_commit` → `delta_commit`, 174 files, 2995
chunks, 15 files changed in the delta):

| Metric | Value |
|---|---|
| Raw index size | 4,780,032 bytes |
| Artifact size (compressed) | ~595,000 bytes |
| Compression ratio | ~8.0x |
| Import + delta-sync wall time | ~690–730 ms |
| Full re-index wall time | ~1,740–1,850 ms |
| Speedup | ~2.5x |
| Correctness | Synced store and an independent full re-index of the same final commit produce identical file/chunk counts (174 files, 2995 chunks) |

Two independent runs produced consistent results (2.54x speedup both
times; compression ratio 8.03–8.04x). Reproduce via `prepare_repo(".",
base_commit)` (from `archex.benchmark.delta_strategies`, the same helper
the M2 delta benchmark uses) plus `export_artifact` /
`import_artifact` / `sync_imported_artifact` timed around a checkout to
`delta_commit`, compared against an independent `_full_index` call at the
same commit.

## CLI usage

```console
$ archex index --export-artifact .archex-artifact.xz
Indexed repository: /path/to/repo
...
Artifact exported:  .archex-artifact.xz
Artifact revision:  a1b2c3d4...
Artifact size:      123456 bytes
Updated .gitattributes: /path/to/repo/.gitattributes

$ archex init --from-artifact .archex-artifact.xz
Initialized archex project at /path/to/repo/.archex
...
Imported index artifact: .archex-artifact.xz
Artifact revision:       a1b2c3d4...
Artifact corpus:         174 files, 2995 chunks
Delta-sync strategy:     delta
Files changed since export: 15
Sync time:               687.0 ms
```

Export never runs automatically — it is an explicit, opt-in flag on
`archex index`, and the resulting file is never committed automatically.
