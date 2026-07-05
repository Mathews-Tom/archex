# Ruby grammar evaluation

**Grammar:** bundled Ruby grammar resolved by `tree_sitter_language_pack.get_language("ruby")` and Archex's existing `tree-sitter-language-pack==0.13.0` fallback. No new dependency is required; Ruby already resolves through this grammar at `CHUNK_ONLY` tier.

**Method:** probed the grammar directly with `tree_sitter_language_pack.get_language("ruby")` + `tree_sitter.Parser`, walking every parsed fixture `root_node` for `has_error`, `ERROR`, and `is_error` nodes. The fixture set intentionally checks adjacent declarations, nested modules, singleton methods, mixin calls, visibility markers, constants, external `require`, and relative `require_relative` forms because the Groovy halt precedent was boundary corruption on idiomatic declarations, not grammar absence.

## Idioms probed (all `has_error == False`, zero `ERROR` nodes)

- Top-level `require` and `require_relative` calls with string arguments.
- Nested modules (`module StoreFront`, `module StoreFront::Mixins`, `module ClassMethods`) and adjacent class declarations in one namespace body.
- Classes under nested modules, inheritance (`class User < ApplicationRecord`), constants, initializer methods, instance methods, singleton methods (`def self.find_by_email`, `def self.slugify`, `def self.included`).
- Ruby mixin idioms: `include`, `extend`, and an `included(base)` hook extending nested `ClassMethods`.
- Visibility markers (`protected`, `private`) affecting subsequent methods without corrupting method boundaries.
- Attribute helper calls (`attr_reader :email, :role`), keyword arguments, hash literals, instance variables, chained calls, regex literals, and block-free collection calls.
- Adjacent declarations of the same kind (`Admin` and `Guest` classes) inside `StoreFront::Legacy`, verifying one declaration does not swallow the next sibling's boundary.

## Result: PASS, no GAP

The bundled Ruby grammar handles every construct required by this milestone without a single error node. Module, class, method, singleton method, mixin, import, and visibility constructs all parse into stable named nodes with independent `start_point`/`end_point` ranges. Full-tier promotion proceeds; the adapter test suite exercises these fixtures.

## Fixture layout

Mirrors the existing `java_simple`/`php_simple` convention: a small namespace-rooted Ruby tree under this directory, with each file exercising a distinct construct so symbol extraction and cross-file import resolution are covered end to end.

| File | Constructs covered |
|---|---|
| `app.rb` | Script-style entry point, `require`, multiple `require_relative` imports, full fixture wiring |
| `store_front/auditable.rb` | Nested modules, mixin instance method |
| `support/slugger.rb` | Module constant, singleton method |
| `models/user.rb` | Class, inheritance, include/extend mixins, constants, attr reader, initializer, singleton method, public/protected/private methods |
| `services/user_service.rb` | External and relative imports, class service object, constants, methods using imported model |
| `mixins/trackable.rb` | Nested modules, `included` hook singleton method, nested `ClassMethods` module, mixin instance method |
| `legacy/admin.rb` | Adjacent classes in one namespace body for boundary-corruption probing |
