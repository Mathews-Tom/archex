# Orientation profile comparison (R24)

Deterministic, offline, model-free and agent-free: every number below is
recomputed from one graph artifact and a frozen task population, so this
report regenerates byte-for-byte from the same inputs.

## Inputs

- Repository: `archex`
- Graph revision (corpus measured): `555157ddd1199cd9a02690511c9004bcbd3ad577`
- Graph exported by archex `0.29.0`
- Indexed files: 1182
- Graph: 17473 nodes, 20024 edges
- Tasks: 16 (frozen in the manifest)

## Profiles

| Profile | Budget | Context tokens | Exact paths | Locatable | Unlocated | Completeness | Mean locator breadth | Modeled calls | Files named | Files not named | Self-reported omissions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | max_files=40 | 2859 | 14/58 | 44/58 | 0/58 | 1.0000 | 197.5 | 32 | 127/1182 | 1055/1182 | 0 in 0 section(s) |
| `compact` | 400 tokens | 376 | 0/58 | 58/58 | 0/58 | 1.0000 | 209.6 | 41 | 2/1182 | 1180/1182 | 94 in 7 section(s) |
| `compact` | 900 tokens | 830 | 12/58 | 46/58 | 0/58 | 1.0000 | 103.3 | 46 | 17/1182 | 1165/1182 | 61 in 4 section(s) |
| `compact` | 1500 tokens | 877 | 12/58 | 46/58 | 0/58 | 1.0000 | 103.3 | 46 | 22/1182 | 1160/1182 | 56 in 3 section(s) |

## Omissions per profile

### `full` (max_files=40)

- `public_interfaces`: 524 of 609 files omitted (section_cap, **silent**)
- `complexity_hotspots`: 680 of 757 files omitted (section_cap, **silent**)
- `test_surface`: 375 of 417 files omitted (section_cap, **silent**)

### `compact` (token_budget=400)

- `directory_clusters`: 21 of 39 directories omitted (folded_directories, reported by the tool)
- `directory_clusters`: 8 of 18 rows omitted (token_budget, reported by the tool)
- `reading_order`: 2 of 16 items omitted (section_cap, reported by the tool)
- `reading_order`: 14 of 14 rows omitted (token_budget, reported by the tool)
- `test_surface`: 33 of 41 directories omitted (folded_directories, reported by the tool)
- `test_surface`: 8 of 8 rows omitted (token_budget, reported by the tool)
- `configuration_surface`: 8 of 8 rows omitted (token_budget, reported by the tool)

### `compact` (token_budget=900)

- `directory_clusters`: 21 of 39 directories omitted (folded_directories, reported by the tool)
- `reading_order`: 2 of 16 items omitted (section_cap, reported by the tool)
- `test_surface`: 33 of 41 directories omitted (folded_directories, reported by the tool)
- `configuration_surface`: 5 of 8 rows omitted (token_budget, reported by the tool)

### `compact` (token_budget=1500)

- `directory_clusters`: 21 of 39 directories omitted (folded_directories, reported by the tool)
- `reading_order`: 2 of 16 items omitted (section_cap, reported by the tool)
- `test_surface`: 33 of 41 directories omitted (folded_directories, reported by the tool)

## Metric definitions

- **Exact paths** — expected files named by their exact repository-relative path in the view.
- **Locatable** — not named exactly, but an ancestor directory is named, so one directory listing or path-scoped query reaches them.
- **Unlocated** — neither; a repository-wide search is the only route left.
- **Mean locator breadth** — indexed files the matched handle stands for (1 for an exact path, the corpus size for an unlocated file). Lower is better; it prevents buying completeness with uselessly coarse prefixes.
- **Modeled calls** — a declared model, not an observation: one call for the view, one per distinct matched locator directory, one per unlocated file. It charges one unit per locator directory regardless of how much that directory contains, so it penalises a view with finer prefixes; read it together with locator breadth, which moves the other way. R20's telemetry is whole-session and records no orientation-phase call counts, so no agent-observed call reduction is claimed here.
- **Files named / not named** — indexed files whose exact path the view carries, measured identically for both profiles. This is the only omission figure comparable across profiles: an enumerating profile names more files, a clustering profile names fewer and points at prefixes instead.
- **Self-reported omissions** — items the profile itself declares in its own receipt, in the unit each section counts (files, rows, or folded directories — never summed across units). The full profile declares nothing, which is the difference this column exists to show; the per-section lists above give the silent counts the harness had to reconstruct for it.
