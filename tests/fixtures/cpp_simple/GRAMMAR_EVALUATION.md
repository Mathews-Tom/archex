# C++ grammar evaluation

**Grammar:** `tree-sitter/tree-sitter-cpp`, the official `tree-sitter` GitHub
org grammar, resolved via the bundled `tree-sitter-language-pack==0.13.0`
fallback under the pack name `"cpp"` (`src/archex/parse/engine.py`'s
`_try_language_pack`). No new dependency -- `cpp` already resolves through
this path today at `CHUNK_ONLY` tier. Same maturity tier as the bundled C/
Java/Go grammars, not a third-party or single-purpose fork.

**Method:** probed the grammar directly with
`tree_sitter_language_pack.get_language("cpp")` + `tree_sitter.Parser`,
walking `root_node` for `has_error`/`is_error`/`is_missing` nodes across
idiomatic C++, per the grammar-vetting procedure (never trust a clean
happy-path snippet; test real edge cases; test adjacent declarations for
boundary corruption, not isolated ones) -- with specific attention to the two
constructs this milestone is named for and flagged as highest-risk (report
Â§8 objection 4): **templates** and **overload resolution**. Every field name
used by the adapter (`declarator`, `parameters`, `type`, `name`, `body`,
`path`, `scope`) was confirmed via `child_by_field_name` before being relied
on in code, not assumed from the grammar's public docs.

## Idioms probed (all `has_error == False`, zero `ERROR`/missing nodes, unless noted)

- **Overloaded free functions and methods**, same name, different parameter
  lists (`int add(int,int)` / `double add(double,double)` /
  `int add(int,int,int)`; `void move(int,int)` / `void move(double,double)`
  both as in-class prototypes and out-of-class definitions) -- each overload
  parses as an independent `function_definition`/`declaration` with its own
  `function_declarator` and `parameter_list`, siblings with fully independent
  boundaries. No grammar-level merging or corruption at any arity.
- **Namespaces**: single (`namespace geo { ... }`), nested the classic way
  (`namespace geo { namespace shapes { ... } } }`), and the C++17
  nested-namespace-definition form (`namespace geo::shapes { ... }` --
  `namespace_definition`'s `name` field is a `nested_namespace_specifier`
  whose text is the full `::`-joined path), and anonymous (`namespace { ... }`
  -- `name` field is `None`).
- **Classes and structs**: `class_specifier` and `struct_specifier` are
  distinct node types (trivial CLASS-vs-TYPE kind mapping); nested inside
  another class/struct they surface as a `field_declaration` whose `type`
  field is the nested `class_specifier`/`struct_specifier` and whose
  `declarator` field is `None` (distinguishing a nested *type* from a K&R
  combined struct-definition-plus-*variable*, where `declarator` is present);
  forward declarations (`struct Fwd;`, `class Fwd2;`) have no `body` field
  and are correctly excludable, same as C's precedent.
- **Templates -- functions, classes, and member templates**:
  `template <...> ...` wraps exactly one declaration
  (`template_parameter_list` then the templated `function_definition` or
  `class_specifier`) as a `template_declaration` node, at both namespace
  scope and inside a class body (a templated method inside a non-template or
  template class). Unwrapping this single-child wrapper needs no different
  handling from a plain declaration once unwrapped.
- **Explicit template specialization**: `template <> class Pair<int> { ... }`
  parses cleanly; the specialization's `class_specifier` `name` field is a
  `template_type` node (`type_identifier` "Pair" + `template_argument_list`
  "<int>"), whose **full text** ("Pair<int>") is genuinely distinct from the
  primary template's plain "Pair" -- a real, adapter-constructed qualified-
  name difference, not something deferred to disambiguation.
- **Header/impl split via out-of-class qualified definitions**:
  `Point::Point(...)`, `Point::~Point()`, `Point::operator+(...)`,
  `Point::getX()`, `int geo::Point::getX() const`,
  `void geo::inner::Foo::bar()`, and the member-template form
  `T Box<T>::get() const` all parse as a `function_definition` whose
  `declarator`'s inner declarator is a `qualified_identifier` node. Its
  `scope`/`name` fields are themselves right-recursively nested
  (`geo::inner::Foo::bar` = `scope: geo`, `name: (inner::Foo::bar)`, ...), so
  the adapter uses the qualified_identifier's **full raw text**,
  `rpartition("::")`-split once into `(scope_prefix, tail_name)`, rather than
  walking the field recursion manually -- correct regardless of nesting depth
  and identical whether the `.cpp` file re-opens the same namespace as the
  header (relative scope, e.g. plain `Point::getX`) or spells the full
  namespace path out (`geo::Point::getX`).
- **Constructors, destructors, and operator overloads** (both declaration-
  only and inline-bodied) are structurally distinguishable by their
  declarator's inner-declarator node type: `identifier` matching the
  enclosing class's own name (constructor), `destructor_name` (`~Point`),
  `operator_name` (`operator+`), or `field_identifier` (an ordinary method) --
  confirmed for both the bodyless `declaration` form (in-class prototype /
  out-of-class-declared-nowhere-else case) and the bodied `function_definition`
  form (inline-in-class or out-of-class-defined).
- **Data members**: `int x_, y_;` (one `field_declaration`, multiple
  `field_identifier` children -- all must be extracted, not just the first);
  pointer (`Foo* self_;`) and reference (`int& ref_;`) members -- the two
  wrapper node types differ in field exposure (`pointer_declarator` exposes a
  `declarator` field; `reference_declarator` does not, and its identifier
  must be found positionally instead, confirmed by direct probing after an
  initial implementation silently dropped reference members).
- **`#include`** in both forms (quoted and angle-bracket), identical `path`
  field shape to C's grammar (`string_literal` vs `system_lib_string`).
- **`#pragma once`** parses as an inert `preproc_call` sibling (no wrapping),
  confirming it needs no special-casing in the top-level flattening walk,
  unlike `#ifndef`/`#define`/`#endif` guards which -- like C -- wrap their
  contents in a `preproc_ifdef` node.
- **`extern "C" { ... }`** (braced) and **`extern "C" void f();`** (unbraced,
  single-declaration) linkage-specification forms both parse cleanly; the
  unbraced form's `body` field is the single wrapped `declaration` directly,
  not a `declaration_list` -- the flattening helper branches on `body`'s node
  type to handle both shapes uniformly.

## Result: PASS, with one documented, non-blocking gap

Every construct this milestone names in its acceptance criteria --
overloaded functions/methods, template declarations and specializations,
namespaces, classes, structs, and header/impl-split definitions -- parses
with zero `ERROR` nodes and independently correct `start_point`/`end_point`
boundaries. Per the vetting procedure's disqualifying-failure-mode test, no
cascading corruption was found anywhere probed.

**Documented gap (non-blocking), same class as C's `extern "C"` + `#ifdef`
finding:** `tests/fixtures/cpp_simple/platform.hpp`'s
`#ifdef __cplusplus extern "C" { #endif ... #ifdef __cplusplus } #endif`
idiom (the standard C-interop header guard) triggers a single **contained**
`is_missing` `#endif` token inside the outer `preproc_ifdef`'s own
bookkeeping -- intrinsic to any grammar parsing C++'s token stream without
running the preprocessor (the real closing `}` for `extern "C" {` lives
inside a *different* `#ifdef __cplusplus` block than the opening one, so the
grammar cannot statically pair them). Verified non-cascading directly: all
three sibling declarations inside that file (`platform_sleep_ms` x2 under
`#ifdef _WIN32`/`#else`, `platform_name`) retain exactly correct,
independent, single-line boundaries despite `root_node.has_error == True`.

**Additional idiom probed but excluded from the fixture corpus:**
`extern template class std::vector<int>;` (an explicit-instantiation
*declaration*, an advanced/rare technique for controlling template
instantiation across translation units) produces a single contained
`is_missing` diagnostic on its own line and does not cascade to sibling
declarations either -- verified with a direct probe, not fixture-embedded,
since it is not idiomatic enough to warrant permanent regression coverage
for this milestone.

## Design decisions this evaluation drove

- **`class_specifier` -> `SymbolKind.CLASS`, `struct_specifier` ->
  `SymbolKind.TYPE`**, matching C's struct-as-TYPE precedent -- these are two
  distinct grammar node types, so the mapping is exact, not name-sniffed.
- **Function/method prototypes count as symbols, same as definitions** (same
  rationale as C: a header's public API is almost entirely bodyless
  declarations).
- **No parameter-type signature is embedded in `qualified_name`.** Overloads
  intentionally share one `qualified_name` (e.g. two `move` overloads both
  resolve to `geo.Point.move`) and are distinguished by their own, always-
  distinct `signature` field plus line range -- collision at the final
  `symbol_id` layer is resolved by the existing, already-tested
  `pipeline/chunker.py::_disambiguate_symbol_ids` mechanism (`@2`, `@3`, ...
  sorted by `start_line`), the same mechanism the Scala adapter's own
  `GRAMMAR_EVALUATION.md` documents relying on for its analogous
  companion-object collision. Embedding raw parameter-type text into
  `qualified_name` was considered and rejected: C++ types routinely contain
  `::` (`std::vector<int>`) which, once normalized to this codebase's `.`
  qualified-name convention, would land a stray `.` *after* the true
  class/method boundary and corrupt `archex.precision._get_parent_qname`'s
  shared last-dot parent-lookup (used by every language's `file_outline`
  nesting) into splitting mid-signature instead of at the class/method
  boundary. Template specialization names (`Pair<int>` vs `Pair`) get a
  genuinely distinct `qualified_name` for a different, safer reason: the
  distinguishing text sits *inside* the type's own single name segment
  (before any `.` a member of that type would add), never after it, so it
  cannot corrupt the rightmost-dot split.
- **Out-of-class member definitions resolve their parent via the
  `qualified_identifier`'s own scope text**, `rpartition("::")`-split once
  and `.`-joined onto whatever namespace scope the adapter is currently
  recursing through. This correctly unifies the "re-open the same namespace"
  and "spell the namespace out in full" styles into the same parent
  qualified name. **Known, documented limitation:** an out-of-class
  definition of a *template* class's member (`template <typename T> T
  Box<T>::get() const { ... }`) carries the literal placeholder text
  `Box<T>` as its scope, which will not match the header-declared primary
  template's plain `Box` qualified name -- the member still extracts as a
  correctly named, non-colliding symbol, but nests as a top-level entry
  rather than under `Box` in the outline tree. Real template definitions
  are overwhelmingly header-only (the idiomatic C++ style this milestone's
  fixtures follow throughout) precisely to avoid the explicit-instantiation
  machinery this pattern requires, so this is a deliberately accepted,
  narrow gap rather than a fixture-covered one.
- **An anonymous namespace's direct members default to `PRIVATE`**
  (internal linkage), the modern-C++-idiomatic equivalent of a top-level
  `static` -- without cascading into the access-specifier-governed
  visibility of a class's own members if that class happens to be declared
  inside the anonymous namespace (the two are different, real C++ semantics:
  translation-unit linkage vs. class-encapsulation access).
- **Default member access is a real, grammar-independent C++ semantic**:
  `private` until the first `public`/`protected`/`private` label for
  `class`, `public` for `struct` -- tracked as adapter state while walking a
  body's flattened declaration list, the same class of AST-independent fact
  C's `static`-based visibility check documents.
- **A field_declaration may declare multiple data members**
  (`int x_, y_;`) and must yield one symbol per `field_identifier` child, not
  just the first.
- **Angle-bracket includes never resolve to a local file; quoted includes
  resolve relative-to-including-file first, then by basename** -- identical
  policy to C's resolver, and deliberately extension-agnostic so a `.cpp`
  file's `#include` of a C-tier `.h` header resolves correctly too (`archex`
  builds one repo-wide `file_map` across all languages).
- **`using`/`using namespace`/`typedef`/`alias_declaration` are out of scope
  for import resolution.** They are intra-translation-unit name aliasing, not
  file references -- `#include` remains the only cross-file dependency
  mechanism this adapter models, matching C's include-only precedent.
- **`enum`/`enum class` extraction is out of scope for this milestone.** Not
  named in the milestone's acceptance criteria (functions, classes, structs,
  namespaces) and not declared in the prior `CHUNK_ONLY` tier's
  `chunk_node_types` for `cpp` (`function_definition`, `class_specifier`,
  `struct_specifier`, `namespace_definition`, `preproc_include`) --
  deliberately excluded to keep this adapter's scope matched to what was
  evaluated and tested, not gold-plated.

## Fixture layout

| File | Constructs covered |
|---|---|
| `point.hpp` / `point.cpp` | Namespace `geo`; class with constructors, destructor, overloaded `move(int,int)`/`move(double,double)`, operator overload; header declares (prototypes), impl defines out-of-class (header/impl split); private data members |
| `shapes.hpp` / `shapes.cpp` | C++17 nested-namespace syntax (`geo::shapes`); `struct` with default-public data members; overloaded free functions `area(int,int)`/`area(double,double)`, header/impl split for free functions |
| `container.hpp` | Header-only template class with overloaded template methods (`add(T)` / `add(const T&, int)`) |
| `pair.hpp` | Header-only template class (`Pair<T, U = T>`) plus an explicit specialization (`Pair<int>`) -- distinct qualified names by construction |
| `list.hpp` / `list.cpp` | Self-referential `struct` (`ListNode* next` inside its own body -- proves no false bodyless duplicate); pointer-returning method; cross-file quoted include (`point.hpp`) plus angle-bracket includes (`<cstddef>`, `<cstdlib>`) |
| `platform.hpp` / `platform.cpp` | The `extern "C"` + `#ifdef __cplusplus` C-interop guard edge case above (contained, non-cascading diagnostic); `#ifdef`/`#else`-guarded duplicate-signature prototypes (mirrors C's `platform.h` precedent) |
| `main.cpp` | Entry point (`int main()`), quoted includes across every sibling header plus one angle-bracket include |
