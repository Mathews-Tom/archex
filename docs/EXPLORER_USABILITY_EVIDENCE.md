# Explorer orientation usability evidence

M5 (`.docs/DEVELOPMENT_PLAN.md` §4) requires "usability evidence measures time to first correct
file/symbol rather than self-reported satisfaction" for the local explorer's new-contributor
orientation use case (`.docs/2026-07-10-codeflow-archex-enhancement-analysis.md`, "Adoption
wedge": *New contributor -- Where do I start? -- Bounded repository projection -- Time to first
correct file/symbol*).

## What this evidence is, and is not

This is an **automated, deterministic proxy**, not a live human trial. No human subjects were
recruited or observed to produce these numbers, and this evidence does not claim otherwise. What
it measures honestly: real wall-clock elapsed time, over real HTTP requests, against a real
running `ExplorerServer` (`archex.explorer.server`), to reach a page whose rendered content
contains an objectively correct answer -- defined by the fixture's own documented graph shape,
not by the script that measures it.

A live new-contributor study (recruit unfamiliar developers, time their real navigation, compare
against a self-reported-satisfaction control) remains valuable follow-on work but is out of scope
for an autonomous delivery with no access to human subjects. Reporting this proxy honestly is
preferable to fabricating a satisfaction score or skipping the acceptance row.

## Protocol

Run:

```text
uv run python scripts/m5_explorer_usability_evidence.py
```

The script:

1. Copies `tests/fixtures/impact_diff` into a scratch directory and initializes a git repository
   (mirrors `tests/conftest.py`'s `_init_fixture_repo` and `tests/test_report_artifact.py`'s
   `_edit_hub` helper).
2. Edits `hub.py` (`value * 2` -> `value * 3`), producing one changed file against `HEAD`.
3. Builds a real `AnalysisArtifactV1` (`archex report diff`'s underlying builder) and a real
   `ArchGraph` (`archex graph export`'s underlying builder) from the edited repository.
4. Starts a real `ExplorerServer` loaded with both artifacts.
5. Times two navigation paths over real HTTP requests against the running server:
   - **Diff Review** (`GET /view/diff`): does the rendered page name `hub.py` as the changed
     file?
   - **Target Neighborhood** (`GET /view/neighborhood?node=file:hub.py`): does searching for
     `hub.py` surface all four of its real importers?

## Scenario ground truth

`tests/fixtures/impact_diff`'s own documented graph shape (`tests/conftest.py`): `hub.py` is
imported by four files (`leaf.py`, `consumer_a.py`, `consumer_b.py`, `consumer_c.py`) and is
transitively reachable from the entry point `main.py` -- a deliberate hub for diff-scoped risk
classification tests. A contributor who just cloned this repository and asks "what changed, and
what does it affect?" should be able to answer "`hub.py`, and its four importers" from the
explorer alone, without reading source.

## Measured evidence (2026-07-24, reference dev machine)

| Navigation path | Elapsed | Correct |
| --- | ---: | :---: |
| Diff Review (`/view/diff`) | 0.005s-0.009s (3 runs) | yes |
| Target Neighborhood (`/view/neighborhood`) | 0.001s | yes |

Both navigation paths reached the objectively correct file/symbol on every run. Elapsed time is
dominated by Python HTTP request/response overhead, not by any per-request graph-index
reconstruction: `archex.explorer.server.ExplorerServer` builds its `GraphQuery` once at startup
(see `scripts/m5_explorer_projection_benchmark.py` and
`tests/explorer/test_projection_benchmark.py` for the corresponding 10k/100k-node scale evidence),
so every subsequent neighborhood lookup during one explorer session is a bounded, already-indexed
traversal.

## Reproducing this evidence

Re-run the script above; correctness is asserted by the script itself (nonzero exit on a wrong
answer), so a passing run is self-verifying. Wall-clock numbers will vary with hardware but should
remain well under any budget a human would perceive as slow (sub-second), since the scenario's
`ArchGraph`/`AnalysisArtifactV1` are small (six files).
