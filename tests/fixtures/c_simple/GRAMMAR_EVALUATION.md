# C grammar evaluation

**Grammar:** `tree-sitter/tree-sitter-c`, the official `tree-sitter` GitHub org
grammar, resolved via the bundled `tree-sitter-language-pack==0.13.0` fallback
under the pack name `"c"` (`src/archex/parse/engine.py`'s
`_try_language_pack`). No new dependency — `c` already resolves through this
path today at `CHUNK_ONLY` tier. C is one of the oldest and most heavily used
tree-sitter grammars (same maturity tier as the bundled Python/Java/Go
grammars, not a third-party or single-purpose fork).

**Method:** probed the grammar directly with
`tree_sitter_language_pack.get_language("c")` + `tree_sitter.Parser`, walking
`root_node` for `has_error`/`is_error`/`is_missing` nodes across idiomatic C,
per the grammar-vetting procedure (never trust a clean happy-path snippet;
test real edge cases; test adjacent declarations for boundary corruption, not
isolated ones). Every field name used by the adapter (`declarator`,
`parameters`, `type`, `name`, `body`, `path`) was confirmed via
`child_by_field_name` before being relied on in code, not assumed from the
grammar's public docs.

## Idioms probed (all `has_error == False`, zero `ERROR`/missing nodes, unless noted)

- Function definitions: plain (`int add(int a, int b) { ... }`), `static`
  (internal-linkage), pointer-returning (`void *make_thing(int size) { ... }`,
  wraps the `function_declarator` in a `pointer_declarator`).
- Function prototypes (`declaration` nodes without a body, e.g.
  `int point_distance_squared(const Point *a, const Point *b);`) — the
  dominant shape of `.h` files. Distinguished from function-*pointer*
  variable declarations (`int (*fp)(int, int);`) by checking that the
  innermost `function_declarator`'s own `declarator` field is a plain
  `identifier`, not a `parenthesized_declarator` — verified both shapes
  explicitly since the two are structurally close.
- Structs: named (`struct Point { ... };`), anonymous-typedef
  (`typedef struct { ... } Size;`), named-typedef
  (`typedef struct Rect { ... } Rect;`), forward declaration (`struct Fwd;` —
  `struct_specifier` with no `body` field, correctly excluded from symbol
  extraction since it carries no member information), and the K&R combined
  struct-definition-plus-variable form (`struct Point { ... } origin;`, a
  `declaration` node whose `type` field is the struct_specifier).
- `#include` in both forms: quoted (`#include "point.h"`, a `string_literal`
  `path` field) and angle-bracket (`#include <stdlib.h>`, a
  `system_lib_string` `path` field) — both exercised adjacently in the same
  file (`list.c`, `platform.c`, `main.c`).
- Preprocessor-conditional-wrapped declarations at the top level
  (`#ifdef`/`#else`/`#endif` guarding a function definition or prototype) and
  `extern "C" { ... }` linkage blocks — both extremely common in real C
  headers and BOTH nest their contents as children of the wrapper node
  (`preproc_ifdef`/`preproc_if`/`preproc_elif`/`preproc_else`, or
  `linkage_specification`'s `body` field) rather than flattening to top-level
  siblings of `translation_unit`. The adapter recurses through these wrapper
  types explicitly (`_toplevel_declarations`); a naive one-level
  `root.children` scan would silently miss every declaration inside a
  `#ifdef` guard or an `extern "C"` block — i.e. most real-world `.h` files.

## Result: PASS, with one documented, non-blocking gap

Every construct this milestone's adapter needs — function definitions,
function prototypes, struct definitions (bare, typedef-anonymous,
typedef-named), and both include forms — parses with zero `ERROR` nodes and
independently correct `start_point`/`end_point` boundaries, including when
two declarations of different kinds sit directly adjacent with no blank line
(`platform.h`'s two `platform_sleep_ms` prototypes). Per the vetting
procedure's disqualifying-failure-mode test, no cascading corruption was
found: `tests/fixtures/c_simple/platform.h` (`#ifdef __cplusplus extern "C"
{ #endif ... #ifdef __cplusplus } #endif`, the standard C++-interop header
idiom) triggers a single **contained** `is_missing` `#endif` token inside the
outer `preproc_ifdef`'s own bookkeeping — this is intrinsic to any grammar
that parses C's token stream without running the preprocessor (the real
closing `}` for `extern "C" {` lives inside a *different* `#ifdef __cplusplus`
block than the opening one, so the grammar cannot statically pair them).
Verified this does **not** cascade: all three sibling function prototypes
inside that same `platform.h` file (`platform_sleep_ms` x2, `platform_name`)
retain exactly correct, single-line, mutually independent boundaries
(confirmed by direct node-boundary diff, not just visual inspection) despite
`root_node.has_error == True` for that one file. This is the same
"tolerable, contained" error class documented for Scala's `given` blocks, not
the Groovy-class cascading failure. Full-tier promotion proceeds.

**Documented gap (non-blocking):** the `extern "C"` + `#ifdef __cplusplus`
idiom above always produces a contained parser diagnostic on the specific
line pairing the outer `#ifdef`/`#endif` around the wrapper's closing brace,
purely due to preprocessor-conditional brace matching being undecidable
without preprocessing. It has no effect on symbol or import extraction
(verified above) and requires no adapter workaround.

## Design decisions this evaluation drove

- **Function prototypes count as `FUNCTION` symbols, same as definitions.**
  `chunk_node_types` for the prior `CHUNK_ONLY` tier only names
  `function_definition`, but a header file's public API is almost entirely
  prototype `declaration` nodes with no body — extracting only
  `function_definition` would make `archex outline` return zero functions for
  every `.h` file, failing this milestone's acceptance criterion outright.
  Both shapes reuse the same declarator-unwrapping helper
  (`_unwrap_function_declarator`) so a function's name/signature/visibility
  logic is identical regardless of whether it has a body.
- **`struct_specifier` -> `SymbolKind.TYPE`**, matching Go's precedent
  (`type_spec` -> `TYPE` unless it is an `interface_type`). C structs have no
  interface/trait analogue, so there is no second `SymbolKind` in play.
- **Struct naming prefers the `typedef` alias over the tag name.**
  `typedef struct Rect { ... } Rect;` and `typedef struct { ... } Size;` are
  both far more common in idiomatic C than the bare `struct Foo { ... };`
  form; using the typedef's `declarator` name (falling back to the
  `struct_specifier`'s own `name` field when there is no enclosing typedef)
  means callers see the name they actually write at use sites (`Size`, not an
  invented tag).
- **Structs without a `body` field are not symbols.** A forward declaration
  (`struct Fwd;`) or a bare type reference inside another declaration's
  `type` field (e.g. `struct Point origin;`) carries no member information
  and is excluded — this also prevents `struct ListNode *next;` (a
  self-referential field reference inside `struct ListNode`'s own body) from
  producing a second, bodyless symbol.
- **Visibility is AST-derived (`static` storage class), not name-derived.**
  Unlike Go's uppercase-letter convention, C's public/private distinction is
  the `static` keyword (internal linkage) vs. its absence (external linkage)
  — a real, standardized C semantic, not a project convention. Because this
  cannot be recomputed from a bare `Symbol.name`, `classify_visibility`
  trusts the value set during extraction, the same pattern TypeScript uses
  for its export-based visibility (`return symbol.visibility`).
- **Types (structs) default to `PUBLIC`.** C has no per-type visibility
  keyword; a struct's actual visibility is a function of which files
  `#include` its declaring header, which is an include-graph property, not a
  per-symbol one.
- **Angle-bracket includes never resolve to a local file.**
  `#include <...>` names a system/standard-library search path, not a
  project-relative path; treating it as always-external is the same
  conservative choice Go's resolver makes for stdlib imports.
- **Quoted includes resolve relative-to-including-file first, then by
  basename anywhere in the project** as a fallback for the common case where
  a project's headers live on a compiler `-I` search path rather than
  strictly alongside the including file — a build-system detail this adapter
  has no visibility into.
- **Recursion through `preproc_ifdef`/`preproc_if`/`preproc_elif`/
  `preproc_else`/`extern "C"` is not optional.** Both idioms nest their
  declarations as children of the wrapper node rather than flattening to
  top-level siblings (confirmed above); the same recursive walk is shared by
  symbol extraction and `#include` parsing so a conditionally-compiled
  `#include` is not missed either.

## Fixture layout

C has no package/namespace/directory convention (unlike PHP/Java/Scala), so
the fixture tree is flat, mirroring `go_simple`'s convention (also chosen as
this milestone's closest-effort reference adapter).

| File | Constructs covered |
|---|---|
| `point.h` | Named-and-typedef'd struct (`Point`), anonymous-typedef struct (`Size`), two function prototypes, include guard |
| `point.c` | Function definition returning a struct by value, `static` (private-visibility) helper function, quoted include |
| `list.h` | Self-referential struct (`struct ListNode *next` inside its own body — proves no false bodyless duplicate), pointer-returning function prototype, cross-file quoted include (`point.h`) |
| `list.c` | Pointer-returning function definition, `struct ListNode *`-typed parameter, quoted + angle-bracket includes together |
| `platform.h` | The `extern "C"` + `#ifdef` edge case above: two mutually-exclusive prototypes of the same name under `#ifdef _WIN32`/`#else`, contained (non-cascading) `is_missing` diagnostic |
| `platform.c` | Matching function definitions for the `#ifdef`-guarded prototypes, a `#ifdef` guard *inside* a function body (not adapter-visible, proves the body content doesn't leak into the symbol boundary) |
| `main.c` | Entry point (`int main(void)`), quoted includes across all three sibling headers plus one angle-bracket include |
