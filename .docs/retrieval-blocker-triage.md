# G2 Retrieval Blocker Triage

Source evidence: `uv run archex benchmark readiness --input .archex/benchmark-current --tasks-dir benchmarks/tasks --strategy archex_query --format markdown` and `uv run archex benchmark triage --input .archex/benchmark-current --tasks-dir benchmarks/tasks --strategy archex_query --format markdown` on 2026-06-10.

Baseline for this work: `archex_query` recall `0.819`, precision `0.480`, F1 `0.589`, token efficiency `0.704`, p95 `2059 ms`; zero-recall tasks `0`.

| Task | Classification | Evidence | Expected movement |
|---|---|---|---|
| `express_middleware` | semantic gap + path-alignment miss | Expected router files are present, but broad middleware vocabulary admits application/request/response/view utilities; raw grepped has full recall. | Framework synonym and route/middleware path evidence should rank `lib/router/*` above generic Express support files. |
| `django_orm_queries` | large-repo ambiguity | Full recall with 18 extra files; large Django ORM/backends surface and low MRR indicate broad query terms across many SQL-adjacent modules. | ORM query construction terms should prefer `models/query.py` and `models/sql/{query,compiler}.py`; selectivity diagnostics should identify expansion noise separately. |
| `fastapi_dependency_injection` | semantic gap + expansion noise | Seeds hit all three oracle files; six security/exception files arrive through expansion. | Dependency-injection terms should keep `fastapi/dependencies/*` and `routing.py` dominant; expansion diagnostics should make the noisy import-target additions attributable. |
| `archex_project_init` | path-alignment miss + expansion noise | Oracle files are returned, but low precision comes from status/test/cache/delta/reporting extras; CLI lifecycle vocabulary is not treated as command/path evidence early enough. | Lifecycle ranking should prioritize `cli/init_cmd.py`, `cli/main.py`, `project.py`, and `config.py`; low-signal tests and lifecycle neighbors should be demoted. |
| `archex_project_status` | path-alignment miss + expansion noise | Oracle files are returned, but very broad seed and expansion sets include cache/index/query/test files; p95/token baseline is already acceptable. | Status/fresh/stale/dirty/corrupt terms should focus `status.py`, `project.py`, `index/delta.py`, and `cli/status_cmd.py`. |
| `pydantic_validators` | semantic gap + expansion noise | Oracle files are present; `_generate_schema`, decorators, fields, metadata, and utility modules dilute precision. | Validator/decorator synonym normalization should lift `_validators.py`, `functional_validators.py`, and `_validate_call.py`; diagnostics should expose expansion sources. |
| `click_decorators` | semantic gap + expansion noise | Oracle files are present; expansion adds compatibility, globals, formatting, termui, exceptions, parser, completion. | Decorator/parameter/command lifecycle terms should rank `decorators.py`, `core.py`, and `types.py` above support modules. |
| `django_middleware` | semantic gap + expansion noise | Oracle files are present; middleware package breadth pulls many concrete middleware and HTTP/cache/log helpers. | Middleware lifecycle terms should prefer handler chain files and `middleware/common.py`; expansion diagnostics should identify generic import-target/helper additions. |
| `archex_project_config_resolution` | path-alignment miss + large-repo ambiguity | Oracle files are returned with many parser/index/test/comparison extras; query mixes settings/runtime/config vocabulary across the repo. | Config/settings/runtime terms should prioritize `project.py`, `config.py`, `models.py`, and `cache.py`, with lifecycle path evidence over incidental parser/config mentions. |
| `archex_project_index` | path-alignment miss + expansion noise | Expected files are present, but many tests/reporting/graph/index internals are packed; token efficiency is low (`0.429`). | Index/build/refresh lifecycle ranking should keep CLI/API/cache/project/config files before tests and low-signal internals. |

## Failure mechanisms

- Semantic gap: framework terms such as middleware, validators, decorators, ORM query construction, dependency injection, hooks, route registration, and lifecycle do not yet consistently expand to code-level identifiers and path terms that match implementation files.
- Path-alignment miss: self-repo lifecycle questions need command names (`init`, `status`, `index`, `reset`), project-state concepts (`fresh`, `stale`, `dirty`, `corrupt`), and config path concepts treated as first-class ranking evidence.
- Expansion noise: several blockers already include all oracle files in seeds; precision falls when import-target/importer expansion admits low-signal helpers, tests, or broad support modules.
- Large-repo ambiguity: Django ORM and self config/index questions have many true-adjacent files; ranking must reduce tail files without trading away recall.
- Oracle/task-spec issue: none identified in the top ten. Each top blocker has returned oracle files and actionable ranking/packing noise rather than an invalid expected set.
