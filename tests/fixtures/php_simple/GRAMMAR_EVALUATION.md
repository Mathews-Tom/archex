# PHP grammar evaluation

**Grammar:** `tree-sitter/tree-sitter-php` (official, `tree-sitter` GitHub org), resolved via the
bundled `tree-sitter-language-pack==0.13.0` fallback under the pack name `"php"`
(`src/archex/parse/engine.py`'s `_try_language_pack`). No new dependency — `php` already resolves
through this path today at `CHUNK_ONLY` tier.

**Method:** probed the grammar directly with `tree_sitter_language_pack.get_language("php")` +
`tree_sitter.Parser`, walking `root_node` for `ERROR`/`is_error` nodes across idiomatic PHP,
per the grammar-vetting procedure (never trust a clean happy-path snippet; test real edge cases;
test adjacent declarations for boundary corruption, not isolated ones).

## Idioms probed (all `has_error == False`, zero `ERROR` nodes)

- Semicolon-style namespace (`namespace App\Models;`) and brace-style namespace
  (`namespace App\Legacy { ... }`) with multiple adjacent declarations inside the brace body.
- `use` import forms: simple (`use App\Contracts\Arrayable;`), aliased
  (`use App\Contracts\Arrayable as Arr;`), grouped
  (`use App\Traits\{HasTimestamps, HasSlug};`), `use function ...`, `use const ...`.
- `interface`, `trait`, `class` (including `abstract`, `final`), `enum` (backed, with methods and
  `match` expressions) declared adjacently in one file — no cross-declaration boundary bleed.
- Trait composition inside a class body, including conflict resolution
  (`use A, B { A::method insteadof B; }`).
- Constructor property promotion (PHP 8.0+: `__construct(private readonly int $id, ...)`),
  `readonly` properties (8.1+), typed properties, nullable types, static properties/methods,
  class constants.
- Adjacent top-level declarations of different kinds back-to-back (`class`, `interface`, `trait`)
  to specifically probe the cascading-corruption failure mode called out in the vetting procedure
  (a parse failure in one declaration silently swallowing the next sibling's boundary). Not
  observed — every declaration's `start_point`/`end_point` stayed independent across all probes.

## Result: PASS, no GAP

The official grammar is mature and handles every construct required by this milestone (and
several PHP 8.x additions beyond it, e.g. enums and property promotion) without a single `ERROR`
node. This is unlike the Groovy precedent (#371) where four candidate grammars all corrupted
chunk boundaries on idiomatic input — no such failure mode reproduces here on any probed PHP
idiom. Full-tier promotion proceeds; see `../../parse/adapters/test_php.py` for the adapter test
suite exercising these fixtures.

## Fixture layout

Mirrors the existing `java_simple`/`csharp_simple` convention: one namespace-rooted directory
tree under this directory, each file exercising a distinct construct so the adapter's symbol
extraction and cross-file import resolution are both covered end to end.

| File | Constructs covered |
|---|---|
| `Contracts/Arrayable.php` | Interface |
| `Traits/HasTimestamps.php` | Trait, nullable typed property, method |
| `Models/User.php` | Class implementing an interface, trait use, constructor property promotion, static factory method, class constant, mixed visibility |
| `Models/Status.php` | Backed enum, enum method, `match` expression |
| `Services/UserService.php` | Grouped `use`, aliased `use`, `use function`, static property, private constant, constructor property promotion |
| `Helpers/functions.php` | Top-level namespaced functions (no enclosing class) |
| `Legacy/BraceNamespace.php` | Brace-style namespace with adjacent interface + final class |
| `index.php` | Script-style entry point wiring the above together |
