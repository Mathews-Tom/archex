# Orientation profile comparison (R24)

Measures whether the opt-in compact onboarding profile costs fewer context tokens **without** making the files a task needs harder to reach. Deterministic, offline, model-free and agent-free: no hosted call, no agent loop, no spend.

## Protocol

- **Population.** The 16 self-repository tasks already in `benchmarks/tasks/` (`repo: "."`), listed by id in `manifest.yaml` so the population cannot drift when tasks are added. They are reused, not invented: each already declares the exact `expected_files` a correct answer must reach, which is the reachability oracle this measurement needs. 58 expected files in total.
- **Subject.** One `ArchGraph` for both profiles, so the comparison cannot be confounded by a different corpus.
- **Profiles.** `full` at `max_files=40` (the shipped default) against `compact` at 400, 900, and 1500 tokens.
- **Metrics.** Context tokens; exact-path hits, locatable and unlocated expected files; completeness; mean locator breadth; a declared exploration-call model; files named out of the whole indexed corpus; and self-reported omissions. Definitions are printed in the report and documented in `src/archex/benchmark/orientation.py`.

### Reading the omission columns

Two different things get called "omission", and this report keeps them apart deliberately:

- **Files named / not named** is the only figure comparable across profiles. It counts indexed files whose exact path appears in the view, measured identically for both, against the same 1182-file denominator. An enumerating profile names more files; a clustering profile names fewer and points at prefixes instead.
- **Self-reported omissions** counts what each profile declares in its own receipt, in the unit each section actually counts — files, rows, or folded directories — never summed across units. A folded directory still has a listed parent prefix, so its files keep a locator; a truncated file in the full profile does not. The per-section lists in the report carry the unit on every line.

The earlier framing of this comparison ("full omits 2810 items, compact omits 61") was withdrawn: it charged the full profile's reading order against the entire corpus while charging the compact profile's against only its 16 hub candidates, and it summed files, rows and directories into one number. The reading-order section is now excluded from omission accounting for both profiles, because a ranked top-N is not "omitting" the tail in the sense a truncated enumeration is.

## Regenerating

Two provenance facts are separate and both recorded in the report. The **graph revision** is the commit whose tree the measured corpus comes from; it must be a commit on `main`, both so a reader can reproduce it forever and so the report's own files cannot change the corpus it reports on. The **archex version** that exported the graph is recorded next to it. The renderers themselves come from the working tree you run the command in.

```bash
git worktree add /tmp/r24-graph <commit-on-main>
(cd /tmp/r24-graph && archex graph export . --output /tmp/r24-graph.json)
archex benchmark orientation --graph /tmp/r24-graph.json --format markdown \
  --output benchmarks/orientation/R24_ORIENTATION.md
archex benchmark orientation --graph /tmp/r24-graph.json --format json \
  --output benchmarks/evidence/r24-compact-orientation.json
git worktree remove /tmp/r24-graph --force
```

Both artifacts must come from the same graph; the JSON re-renders to the markdown byte-for-byte, which is what makes the pair auditable and is asserted by `tests/benchmark/test_orientation.py`. The checked-in pair was produced from `555157ddd1199cd9a02690511c9004bcbd3ad577` on `main`, a commit that will not be rewritten.

## Findings on this repository

Against the shipped `full` profile at `max_files=40` (2859 tokens):

- **Context cost.** `compact` at 900 tokens renders in 830 — a 71.0% reduction. At 400 tokens it renders in 376, an 86.9% reduction. The analysis document forecast 10–30%.
- **Completeness.** `1.0000` for every profile: no expected file becomes unreachable at any budget measured. The forecast of unchanged required-file recall holds.
- **Locator specificity.** Mean locator breadth falls from 198 files per handle (`full`) to 103 (`compact` at 900) — the adaptive clusters point at roughly half as much material for the same completeness. At 400 tokens the reading order is dropped entirely and breadth rises to 210, slightly worse than `full`: that budget buys cheapness with coarser locators, which is why the default is 900.
- **Enumeration versus summary, stated plainly.** `full` names 127 of 1182 indexed files exactly; `compact` at 900 names 17. The compact profile is a summary and reaches the same completeness through prefixes, not through naming more files. Anyone who needs enumeration should keep using `full`.
- **Omission honesty.** `full` declares nothing at all — 0 self-reported omissions — while silently truncating 524 public interfaces, 680 complexity hotspots and 375 test files (1579 files across three sections, reconstructed here because the tool does not report them). `compact` at 900 declares 61 items across 4 sections in its own receipt, each carrying its unit.
- **Modeled exploration calls are higher, not lower: 46 (`compact` at 900) against 32 (`full`).** This is reported rather than buried because it is the one forecast this measurement does not support. The cause is the model, not a defect: it charges one unit per distinct locator directory, so a view with finer prefixes is charged more than one naming a single 260-file directory — even though the coarse call returns far more material to read. Locator breadth is the counterweight and moves the other way. No agent-observed call reduction is claimed anywhere: R20's telemetry is whole-session (mean 11.16 tool calls in the Archex arm, mean 170,394 total tokens) with no orientation-phase segmentation, so there is no baseline to claim against.

## What this does not measure

- Any agent behaviour. The exploration-call figure is a declared model over path reachability, not an observation of a model or an agent.
- Retrieval quality. No retrieval lane, ranking, or default changed, and `benchmarks/product_loop/` and `benchmarks/headtohead/results/` are untouched.
- Whether the compact profile should become a default. It is opt-in; promotion needs separate evidence and authorization.
