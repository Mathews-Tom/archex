# Spike pre-registrations

A pre-registration fixes a spike's hypothesis, primary metric, decision margins, clustering unit, and kill criterion **before** any data exists, and merges before its first run. Commit order is the proof: if the pre-registration commit is not an ancestor of the evidence commit, the result is not pre-registered.

That proof only works for tracked files. This directory exists so it keeps working — `/.docs/` is ignored by both the global excludes file and this repository's `.gitignore`, so a pre-registration written there is silently untrackable and `git add` reports success while committing nothing.

## Where things live

| | Location |
| --- | --- |
| Template for new pre-registrations | `TEMPLATE.md`, here |
| New pre-registrations | here, as `<spike-id>.md` |
| S0 — external replication gate (Gate A) | Historical record identified by each S0 evidence artifact's `preregistration_commit` |

S0's pre-registration remains identified by the immutable `preregistration` path and `preregistration_commit` stored in both evidence artifacts. Those commit references, the artifacts, and `tests/benchmark/test_s0_replication_artifacts.py` preserve provenance without requiring a local working-copy document in a fresh clone.

## Rules

- Copy `TEMPLATE.md`, complete every required field, and merge it before the first data-generating run.
- Never revise a pre-registration after data exists. Append to its `Post-hoc changes` section instead, dated, and label every number the change touches as exploratory.
- Never widen a margin or band after seeing a result. `src/archex/benchmark/replication.py` enforces the machine-checkable half of this for replication artifacts; the rest is discipline.
