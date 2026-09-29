## archex (installed, repository already indexed)

`archex` is a local code-context CLI for this repository. Its index is built; do not run `archex init` or `archex index`.

Choose by question:

- Exact identifier, literal string, regex, or every occurrence: use `grep`. It is exact and complete.
- Where something is implemented, how a flow works, or an unfamiliar subsystem: run `archex scout . "<question>" --budget 1000 --format json` first, then fetch the handles it returns with `archex symbol . '<handle>'`.
- Body of a known symbol: `archex symbol . 'symbol:<path>::<Name>#<kind>'` instead of reading the whole file.
- Symbols by name: `archex symbols . <name>`. Outline of one file: `archex outline . <path>`.
- Before changing an exported symbol or a widely imported file: `archex impact . --changed-file <path>`.
- A larger ranked bundle: `archex query . "<question>" --format markdown`.

Every scout and query result ends with a receipt. If `context_complete_reason` is `low_query_match` or `no_candidates`, rephrase with the code's own identifiers (`query_terms_unmatched` lists the words that matched nothing) or use `grep`.

`archex status .` reports whether the index still matches the working tree; after your edits it will not. archex output is context selection, not proof: confirm with `read` or `grep` before editing.
