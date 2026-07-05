# M15 — HTML→JS/CSS cross-language graph edges: benchmark decision

Milestone: Cross-language edges (HTML to JS/CSS), conditional on demonstrated
recall improvement. This records the required-file-recall comparison and the
resulting merge decision, per the "measure then claim" gate.

## What was actually wired vs. what was already there

Before this milestone's investigation, `HtmlAdapter.resolve_import` (shipped
in the M12 HTML STRUCTURED adapter) already flowed into the main dependency
graph generically — `DependencyGraph.from_parsed_files()` and
`index/delta.py`'s edge machinery treat every adapter's resolved imports the
same way regardless of language tier, so HTML→JS/CSS `IMPORTS` edges were
already produced whenever an HTML reference resolved cleanly. No change to
`src/archex/index/graph.py` or `src/archex/index/delta.py` was needed or
made; both were verified (via the fixture in this benchmark and a dedicated
regression test) to already handle cross-language edges correctly once given
a correct resolution.

The actual defect was in resolution itself: `HtmlAdapter`'s local-path
resolver fell back to the same extension-stripped, dotted "module key" that
`build_file_map()` uses for Python-style imports. Two files that share a
directory and basename across extensions (e.g. `app.js` / `app.css`, a
common static-asset pattern) collapse onto the same dotted key, so
`build_file_map()` could only retain one of the two — the other reference
silently resolved to nothing (or, before a first fix attempt, to the wrong
sibling file). This is a real, demonstrated correctness defect: a wrong or
missing edge, not merely an absent one.

The fix: `build_file_map()` now also registers each file's literal
repo-relative path (extension preserved) as its own lookup key, via
`setdefault` so it can never overwrite an existing dotted-key entry —
purely additive, zero risk to existing Python/Go/Rust/etc. import
resolution. `HtmlAdapter` resolves local references through this literal
path key first (via the new shared `resolve_path_reference` helper in
`structured.py`), falling back to the dotted key only if no exact file
exists. See `src/archex/parse/imports.py`, `src/archex/parse/adapters/structured.py`,
`src/archex/parse/adapters/html.py`.

## Fixture and method

`tests/fixtures/polyglot_html_js_css/{kiosk,checkout,catalog,support}/` —
four isolated mini-repos, each one HTML page plus its two local JS/CSS
assets plus ~1KB of unrelated vendored filler (to keep the corpus above the
trivial "everything fits under any budget" size). `kiosk` and `checkout`
deliberately give their JS/CSS assets the same basename (`roster.js`/
`roster.css`, `purchase.js`/`purchase.css`) to exercise the collision
defect; `catalog` and `support` use distinct basenames as a non-colliding
regression control. Page prose and asset identifiers were checked against
each task's actual query text via a real FTS5 `porter unicode61` index
(matching the product's own BM25 tokenizer) to remove accidental lexical
overlap that would let BM25 alone "solve" a task independent of the graph
edge — see `benchmarks/tasks/polyglot_html_js_css/*.yaml`.

Each task's `question` asks what script/stylesheet its own page loads, with
`expected_files` = {page, its JS, its CSS} as required-file-recall ground
truth, at `token_budget: 400` (tight enough that `archex_query` cannot
trivially return the whole mini-repo).

Benchmark support: `src/archex/benchmark/runner.py` gained
`_prepare_local_fixture_repo`, letting a task's `repo:` field point at an
in-repo fixture directory (copied to a temp dir and `git init`'d, offline,
no network) instead of requiring `"."` or a GitHub slug — needed for
`archex benchmark run --tasks-dir` to run these fixture-backed tasks at all.

## Results

`archex_query`, `required_file_recall`, `token_budget=400`:

| task | pre-fix | post-fix | delta |
|---|---|---|---|
| polyglot_kiosk_html_assets (colliding basename) | 0.667 | 1.000 | **+0.333** |
| polyglot_checkout_html_assets (colliding basename) | 0.667 | 1.000 | **+0.333** |
| polyglot_catalog_html_assets (control, no collision) | 1.000 | 1.000 | 0.000 |
| polyglot_support_html_assets (control, no collision) | 0.667 | 0.667 | 0.000 |
| **mean** | **0.750** | **0.917** | **+0.167** |

Reproduce: `pre_fix/` was captured on the commit before the
`build_file_map`/`HtmlAdapter` fix; `post_fix/` on the commit with the fix
applied — same fixture, same tasks, same budget, only the code under test
differs.

```
uv run pytest tests/index/test_graph.py -k cross_language -v
uv run archex benchmark run --tasks-dir benchmarks/tasks/polyglot_html_js_css \
  --strategy archex_query --output <dir>
uv run archex benchmark gate --input <post_fix_dir> --baseline pre_fix
```

`archex benchmark gate --input post_fix --baseline pre_fix` → **quality gate
passed** (no regression on any metric).

## Decision

**Improvement measured. Ship it.** The fix raises required-file-recall on
the two collision-exercising tasks from 0.667 to 1.000 with zero regression
on the two non-colliding controls (support's 0.667 is an unrelated,
pre-existing budget/ranking characteristic, unchanged by this fix — its
`assets/case-widget.js` simply doesn't clear the graph-expansion score
threshold at this budget, independent of collision handling). The stack
proceeds to merge normally.
