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

## CLI usage

```console
$ archex index --export-artifact .archex-artifact.xz
Indexed repository: /path/to/repo
...
Artifact exported:  .archex-artifact.xz
Artifact revision:  a1b2c3d4...
Artifact size:      123456 bytes

$ archex init --from-artifact .archex-artifact.xz
```

Export never runs automatically — it is an explicit, opt-in flag on
`archex index`, and the resulting file is never committed automatically.
