# Benchmark-quality control characterization

## Scope and identity

- Strategy: `archex_query` control; default behavior unchanged.
- Command: `uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/benchmark-quality-control --strategy archex_query --no-progress`
- Candidate baseline manifest: `.archex/baselines/benchmark-quality-control/manifest.json`.
- Candidate baseline coverage: 64 tasks × 3 emitted strategies (`raw_files`, `raw_ripgrep`, `archex_query`). It is unpromoted.
- Source identity, task-manifest digest, retrieval configuration, and generated timestamp are recorded by the manifest.
- Repeat command: `uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/benchmark-quality-repeat --strategy archex_query --no-progress`.
- Stability: the repeat wrote 64 reports; all 64 `archex_query` rows matched the control exactly for recall, precision, F1, MRR, token efficiency, completion-adjusted token efficiency, and required-file recall.

## Control result

The run wrote 64 reports. The approved absolute floors (`recall >= 0.60`, `precision >= 0.20`, `F1 >= 0.30`, `MRR >= 0.55`, product-default token efficiency >= 0.08) produce 44 metric violations across 23 tasks.

| Family | Tasks | Tasks with absolute violation | Total misses | Mean required-file recall | Mean MRR | Mean precision | Mean completion-adjusted token efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| self-repo | 24 | 4 | 0 | 0.863 | 0.979 | 0.504 | 0.697 |
| external-comprehension | 19 | 5 | 0 | 0.947 | 0.825 | 0.536 | 0.738 |
| localization | 21 | 14 | 1 | 0.952 | 0.667 | 0.242 | 0.653 |

Localization is evaluated as a separate user-critical family. Its low precision is not offset by its unweighted aggregate.

## Evidence-backed task classification

Classification is derived from the ordered `archex_query.result_files` and report metrics. `absent required file` means required-file recall is nonzero but below one; `ranked below first` means required-file recall is nonzero and MRR is below one; `excessive unrelated files` means precision is below 0.20; `negative completion-adjusted token` means completion-adjusted token efficiency is below zero.

| Task | Classification |
| --- | --- |
| `archex_adapter_registry` | absent required file |
| `archex_delta_index_lifecycle` | absent required file |
| `archex_project_config_resolution` | absent required file |
| `archex_project_index` | absent required file |
| `archex_project_init` | absent required file |
| `archex_project_status` | absent required file |
| `archex_query_cache_lifecycle` | absent required file |
| `archex_vector_cache_lifecycle` | absent required file |
| `celery_task_dispatch` | ranked below first |
| `click_decorators` | absent required file |
| `django_middleware` | absent required file; ranked below first |
| `express_error_handling` | ranked below first |
| `fastapi_dependency_injection` | absent required file |
| `fastapi_routing` | ranked below first |
| `loc_celery_task_retry` | excessive unrelated files |
| `loc_click_param_process` | ranked below first; excessive unrelated files |
| `loc_django_username_validator` | total miss; excessive unrelated files |
| `loc_express_error_middleware` | ranked below first |
| `loc_fastapi_jsonable_encoder` | ranked below first; excessive unrelated files |
| `loc_fastapi_solve_dependencies` | ranked below first |
| `loc_flask_full_dispatch` | excessive unrelated files |
| `loc_httpx_pool_transport` | ranked below first; excessive unrelated files |
| `loc_httpx_redirect_headers` | excessive unrelated files |
| `loc_mini_redis_command_from_frame` | ranked below first |
| `loc_requests_adapter_send` | ranked below first; excessive unrelated files |
| `loc_requests_redirect_auth` | ranked below first; excessive unrelated files |
| `loc_sqlalchemy_session_merge` | ranked below first; excessive unrelated files |
| `react_hooks` | ranked below first |
| `routing_mixed_chunker` | ranked below first |
| `routing_pl_scoring` | absent required file |

No task oracle has been changed. `loc_django_username_validator` identifies `django/contrib/auth/validators.py`, `ASCIIUsernameValidator`, and `UnicodeUsernameValidator` at the pinned Django 5.1.4 revision; the raw-file baseline returns that exact file. The default control instead returns five unrelated auth files and omits `validators.py`. This is a deterministic retrieval defect, not an oracle correction.

## Next decision input

The localization candidate must recover `loc_django_username_validator`, improve localization precision or MRR, preserve required-file recall in every family, keep completion-adjusted token efficiency non-negative, and stay within the documented p95 latency budget. The control remains unchanged until those criteria are measured.