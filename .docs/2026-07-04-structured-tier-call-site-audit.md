# STRUCTURED tier call-site audit

Date: 2026-07-04
Milestone: M11 STRUCTURED tier scaffolding

## Audit method

Full-repo call-site audit was cross-checked with these searches:

- `LanguageTier\.(FULL|CHUNK_ONLY)` across `.`
- `LanguageTier\.(FULL|CHUNK_ONLY|UNKNOWN)|CHUNK_ONLY_LANGUAGE_IDS|get_language_tier\(` across `src/archex;tests`
- `CHUNK_ONLY_LANGUAGE_IDS|LANGUAGE_SUPPORT|tier ==|support\.tier|\.tier ==|LanguageStats\(` across `src/archex;tests`
- `languages|LanguageStats|tier|chunk_only|full grammars` across `src/archex/cli;src/archex/doctor.py;src/archex/serve;tests/cli;tests/serve`

The exact grep evidence for the first search is attached in the PR-1 handoff/review notes. The audit is source-based; non-code hits in comments, docs, and benchmark prose were excluded from required production decisions.

## Production decisions

| File | Current tier dependency | STRUCTURED decision | PR |
| --- | --- | --- | --- |
| `src/archex/models.py` | `LanguageTier` enum only exposes `FULL`, `CHUNK_ONLY`, `UNKNOWN`. | Add `STRUCTURED = "structured"` between `FULL` and `CHUNK_ONLY`. No behavior branch changes. | PR-1 |
| `src/archex/languages.py` | `_FULL`, `_CHUNK`, `CHUNK_ONLY_LANGUAGE_IDS`, and `get_language_tier()` encode tier registry metadata. | Add `_STRUCTURED`, expose `STRUCTURED_LANGUAGE_IDS`, keep `CHUNK_ONLY_LANGUAGE_IDS` strictly chunk-only, and let `get_language_tier()` return `STRUCTURED` from registry metadata. | PR-2 |
| `src/archex/parse/adapters/__init__.py` | Default adapter registration iterates only `CHUNK_ONLY_LANGUAGE_IDS`; STRUCTURED languages must not be silently registered as chunk-only or omitted once a concrete language flips tiers. | Keep chunk-only registration scoped to `CHUNK_ONLY_LANGUAGE_IDS`. Add a distinct STRUCTURED registration loop through `STRUCTURED_LANGUAGE_IDS` and `make_structured_adapter()` in PR-3 so future concrete STRUCTURED languages enter the same parser registry path without chunk-only routing. | PR-2/PR-3 |
| `src/archex/doctor.py` | Grammar details/text initialize and render only `full` and `chunk_only` buckets; missing grammar severity only treats FULL as fatal. | Add explicit `structured` counts in dictionaries, JSON details, and text rendering. Missing STRUCTURED grammars remain non-fatal like CHUNK_ONLY because STRUCTURED is outline/reference tier, not programming-symbol tier. | PR-2 |
| `src/archex/serve/profile.py` | `LanguageStats.tier` delegates to `get_language_tier()`. | No production branch change needed; add regression coverage proving STRUCTURED propagates and is not defaulted to UNKNOWN. | PR-2 |
| `src/archex/index/graph.py` | No `LanguageTier` branch. It builds file nodes from `ParsedFile`, symbol nodes from `ParsedFile.symbols`, and file edges from resolved `ImportStatement`s. | No production branch change needed; add regression coverage proving a STRUCTURED parsed file with no symbols and a native resolved reference is handled as a file edge with zero symbol nodes. | PR-2 |
| `src/archex/cli/*` | No `LanguageTier` branch. CLI language output surfaces counts/status from lower layers. | No production branch change needed. Doctor CLI receives structured grammar text through `doctor.py`; profile/index/status language counts remain tier-agnostic. | PR-2 |
| `src/archex/parse/adapters/chunk_only.py` | Generic chunk-only adapter returns chunk ranges and no imports/symbols. | Leave unchanged. Add a separate `structured.py` base that preserves the no-symbol invariant while allowing native references through `parse_imports()` and `resolve_import()`. | PR-3 |

## Non-production references

| File | Decision |
| --- | --- |
| `tests/parse/test_language_coverage.py` | Existing FULL/CHUNK_ONLY regression tests stay intact; PR-2 adds STRUCTURED-specific coverage without weakening these assertions. |
| `tests/cli/test_doctor.py` | Add structured grammar JSON/text regression coverage. |
| `tests/serve/test_profile.py` | Add structured tier propagation regression coverage. |
| `tests/index/test_graph.py` | Add structured native-reference graph handling coverage. |
| `tests/parse/adapters/test_structured.py` | Add shared structured base adapter coverage in PR-3. |

## Explicit non-decisions

- No built-in language is flipped to `STRUCTURED` in M11. HTML/XML/YAML/Markdown/CSS are M12/M13.
- No new `SymbolKind` is added. STRUCTURED adapters must not produce programming symbols.
- No new `EdgeKind` is added in M11. Native cross-file references are represented through the existing resolved-import edge path until M15 decides cross-language graph semantics.
