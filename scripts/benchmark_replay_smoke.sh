#!/usr/bin/env bash
set -euo pipefail

uv run pytest tests/benchmark/test_strategies.py --no-cov
uv run pytest tests/benchmark/test_dogfood.py tests/benchmark/test_baseline.py tests/benchmark/test_triage.py --no-cov
uv run pytest tests/serve/ --no-cov
uv run pytest tests/index/test_rerank.py --no-cov
! grep -rE 'task_id\s*==\s*"' src/archex
