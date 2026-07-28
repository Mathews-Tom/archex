# Spike pre-registrations

A pre-registration fixes a spike's hypothesis, primary metric, decision margins, clustering unit, and kill criterion **before** any data exists, and merges before its first run. Commit order is the proof: if the pre-registration commit is not an ancestor of the evidence commit, the result is not pre-registered.

That proof only works for tracked files. This directory exists so it keeps working — `.docs/` is ignored by the global excludes file, so a pre-registration written there is silently untrackable and `git add` reports success while committing nothing.

## Where things live

| | Location |
| --- | --- |
| Template for new pre-registrations | `TEMPLATE.md`, here |
| New pre-registrations | here, as `<spike-id>.md` |
| S0 — external replication gate (Gate A) | `.docs/spikes/S0-replication-gate.md` |

S0 stays at its historical path deliberately. It is a closed record: `GATE-A.md`, both evidence artifacts, and `tests/benchmark/test_s0_replication_artifacts.py` all reference that path and the commit it merged in, and rewriting a completed study's provenance to tidy a directory would weaken the record it exists to protect. The file is already tracked, so appending to its post-hoc section still works.

## Rules

- Copy `TEMPLATE.md`, complete every required field, and merge it before the first data-generating run.
- Never revise a pre-registration after data exists. Append to its `Post-hoc changes` section instead, dated, and label every number the change touches as exploratory.
- Never widen a margin or band after seeing a result. `src/archex/benchmark/replication.py` enforces the machine-checkable half of this for replication artifacts; the rest is discipline.
