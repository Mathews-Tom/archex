# Explorer orientation usability evidence

M5 required "usability evidence measures time to first correct file/symbol rather than
self-reported satisfaction" for the local explorer's new-contributor orientation use case:
*New contributor — Where do I start? — Bounded repository projection — Time to first correct
file/symbol*.

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
5. Times five navigation paths, the first two over real HTTP requests against the running
   server, the next two exercising the R25 interactive additions, and the last against the
   offline static bundle with the server already stopped:
   - **Diff Review** (`GET /view/diff`): does the rendered page name `hub.py` as the changed
     file?
   - **Target Neighborhood** (`GET /view/neighborhood?node=file:hub.py`): does searching for
     `hub.py` surface all four of its real importers?
   - **Node Search** (`GET /view/search?q=hub`): does a substring query reach `hub.py` and link
     to its neighborhood, without the reader already knowing the exact node id?
   - **Inbound-only Neighborhood** (`GET /view/neighborhood?node=file:hub.py&direction=in`):
     does the page report every importer, mark each row inbound, and contain no outbound row?
   - **Static export** (`export_explorer_site` then read from disk): does the offline bundle's
     per-node page carry the same four importers with inbound orientation, does its node index
     list `hub.py`, and does it contain no `<script>` and no session token?

## Scenario ground truth

`tests/fixtures/impact_diff` is a deliberate hub built for diff-scoped risk classification tests.
Read from the fixture source rather than from prose: `hub.py` is imported **directly** by five
files -- `leaf.py`, `consumer_a.py`, `consumer_b.py`, `consumer_c.py` (each `from hub import
shared_helper`), and the entry point `main.py:4` (`from hub import other_helper`). The four
`shared_helper` importers are the set the evidence script asserts, because they exercise the
shared-symbol fan-in the risk classifier keys on; `main.py` is the fifth direct importer and
appears in every inbound-edge count below. A contributor who just cloned this repository and asks
"what changed, and what does it affect?" should be able to answer "`hub.py`, and the files that
import it" from the explorer alone, without reading source.

## Measured evidence (2026-09-10, reference dev machine)

| Navigation path | Elapsed | Correct |
| --- | ---: | :---: |
| Diff Review (`/view/diff`) | 0.005s | yes |
| Target Neighborhood (`/view/neighborhood`) | 0.001s | yes |
| Node Search (`/view/search`) | 0.001s | yes |
| Inbound-only Neighborhood (`direction=in`) | 0.001s | yes |
| Static export (offline, server stopped) | 0.004s | yes |

Every navigation path reached the objectively correct file/symbol. Elapsed time for the served
paths is dominated by Python HTTP request/response overhead, not by any per-request graph-index
reconstruction: `archex.explorer.server.ExplorerServer` builds its `GraphQuery` once at startup
(see `scripts/m5_explorer_projection_benchmark.py` and
`tests/explorer/test_projection_benchmark.py` for the corresponding 10k/100k-node scale evidence),
so every subsequent neighborhood lookup during one explorer session is a bounded, already-indexed
traversal. The static-export figure is whole-bundle generation, not per-page: one call writes the
index, every view, and one pre-rendered neighborhood page per high-degree node.

## Browser verification (2026-09-10, Chromium via CDP)

The measurements above are HTTP-level. The R25 views were additionally exercised in a real
browser against a live `archex explore` server loaded with a real `AnalysisArtifactV1` and a real
exported graph (22 nodes, 20 edges) built from the same fixture, and then against the static
export over `file://`. What the browser confirmed, rather than what the HTML string contains:

- **Node search.** `/view/search?q=hub` rendered five matches -- the `hub.py` file node, its two
  interface nodes, and its two symbol nodes -- each linking to a percent-encoded neighborhood
  URL (`/view/neighborhood?node=file%3Ahub.py`). `document.querySelectorAll('script').length`
  was `0`.
- **Directional highlighting.** `/view/neighborhood?node=file:hub.py&depth=2&limit=40` rendered
  14 outbound, 5 inbound, and 1 lateral edge row, with distinct computed background colours per
  orientation (`rgb(238, 246, 255)`, `rgb(255, 246, 238)`, `rgb(246, 246, 246)`) -- so the
  distinction is visible in a browser, not just present as a class name. `consumer_a.py imports
  hub.py` rendered as inbound, which is the semantics a reviewer needs: the consumer depends on
  the seed.
- **Typed-edge filtering.** Checking the `imports` box and submitting the real form navigated to
  `?node=file%3Ahub.py&direction=both&depth=2&limit=40&edge_type=imports`, authenticated by the
  session cookie with no token in the URL, and rendered 6 `imports` rows with the node table
  narrowed from 14 to 6 and the note "Filtered to imports: 14 of this neighborhood's edges
  hidden. The filter narrows the bounded traversal above, not the whole graph."
- **Direction control.** `direction=in` rendered exactly the five importers, every row inbound,
  with the select element reflecting `in`.
- **Rejection paths.** An unresolvable node rendered "No graph node matches 'does-not-exist'" and
  a non-matching search rendered "No node matches zzz-nothing." with no table at all -- neither
  fabricated a result.
- **Offline export.** Opening `file:///.../index.html` resolved all six relative nav links,
  reported zero `<script>` elements and zero remote `src`/`href` values, and the per-node page
  for `hub.py` showed 5 inbound and 4 outbound rows with zero `<form>` elements and the note
  explaining that the interactive controls need the local server.
- **Clean install.** The bundle produced by a wheel installed into a fresh virtualenv was
  byte-identical to the source-tree bundle (`diff -r`), and an oversized artifact was refused by
  the installed CLI with "above the explorer's 33554432-byte limit".

## Two static-HTML surfaces, deliberately distinct

archex renders the same `AnalysisArtifactV1` to static HTML in two places, and they are not
interchangeable:

| Surface | Produced by | Shape | Row caps |
| --- | --- | --- | --- |
| Single-page diff report | `archex report diff --format html` (`src/archex/report/render_html.py`) | One self-contained document: provenance table, summary, a Mermaid structure block embedded as `<pre>` text, and diff tables. Includes `file://`/editor links built from `source_root`. | `MAX_HTML_ROWS = 50` |
| Explorer bundle | `archex explore ... --export DIR` (`src/archex/explorer/export.py`) | A directory: diff review, module map, node index, per-node dependency neighborhoods, context receipt, and index health, cross-linked by relative filename. No editor links, no Mermaid block. | 100 per diff table, 200 module rows, 1000 index rows, 200 node pages |

The caps differ because the budgets differ: a single page a reviewer opens standalone needs a
tighter ceiling than a multi-page bundle whose tables are one view among several. The explorer
bundle's tables come from `archex.explorer.viewmodel`, which the loopback server also uses, so the
served and exported explorer views cannot drift from each other. They can drift from
`report diff --format html`, which is why both surfaces are named here rather than left implicit.

## Reproducing this evidence

Re-run the script above; correctness is asserted by the script itself (nonzero exit on a wrong
answer), so a passing run is self-verifying. Wall-clock numbers will vary with hardware but should
remain well under any budget a human would perceive as slow (sub-second), since the scenario's
`ArchGraph`/`AnalysisArtifactV1` are small (six files).
