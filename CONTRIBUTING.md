# Contributing to archex

## Development Setup

```bash
git clone https://github.com/Mathews-Tom/archex.git
cd archex
uv sync --all-extras
```

## Running Tests

```bash
# Full non-slow test suite with coverage gate
uv run pytest

# With HTML coverage report
uv run pytest --cov-report=html

# Include slow tests
uv run pytest -m ""

# Run a specific test file
uv run pytest tests/test_integration.py -v
```

## Linting and Type Checking

```bash
# Lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check . --fix

# Format
uv run ruff format .

# Type check (strict mode)
uv run pyright .
```

All checks must pass before submitting a PR. CI runs lint, format check, type check, and tests on supported Python versions.

## Code Style

- **Formatter/Linter:** ruff (config in `pyproject.toml`)
- **Line length:** 100
- **Type checking:** pyright strict mode
- **Target Python:** 3.11+

## Strategic Reassessment Freeze

Until Gate A passes on R3's external replication verdict, contributors must not:

- add a new retrieval lane;
- attempt to promote a retrieval path to the default;
- add a new language tier; or
- add a new MCP tool.

The unstarted forward milestones from the prior plan — M2, M3, M4, M5, and M10 — are **SUSPENDED — pending strategic-reassessment Gate A**. Do not resume them. The local strategic-reassessment record is `.docs/strategic-reassessment/`; the governing contract and recorded lift condition are in tracked `.docs/DEVELOPMENT_PLAN.md` §3 (Gate A) and §6 (R3).

R5, MCP retrieval-gated tool disclosure, is the sole carve-out before Gate A. It may change how existing MCP tools are exposed, but must not add or remove a tool capability or change tool behavior.

## Adding a Language Adapter

Language adapters live in `src/archex/parse/adapters/`. Each adapter implements the `LanguageAdapter` protocol defined in `src/archex/parse/adapters/base.py`.

1. Create `src/archex/parse/adapters/your_language.py`
2. Implement the `LanguageAdapter` protocol (see existing adapters for reference)
3. Register the adapter in `src/archex/parse/adapters/__init__.py`
4. Add the tree-sitter grammar dependency to `pyproject.toml`
5. Add tests in `tests/test_parse/test_your_language.py`
6. Add fixture files in `tests/fixtures/your_language/`

External adapters can be registered via entry points without modifying archex core:

```toml
[project.entry-points."archex.language_adapters"]
dart = "mypackage.adapters:DartAdapter"
```

## Adding a Pattern Detector

Pattern detectors are registered via the `PatternRegistry`. See `src/archex/analysis/patterns/` for existing detectors.

1. Create your detector function with signature: `(list[ParsedFile], DependencyGraph) -> DetectedPattern | None`
2. Register via entry points:

```toml
[project.entry-points."archex.pattern_detectors"]
my_pattern = "mypackage.patterns:detect_my_pattern"
```

## Running Benchmarks

```bash
# Run all benchmark tasks
uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/e2e

# Run specific tasks
uv run archex benchmark run --tasks-dir benchmarks/tasks --filter "archex_*" --output .archex/e2e

# Check quality gate
uv run archex benchmark gate --input .archex/e2e --baseline benchmarks/dogfood_baseline.json --warn-latency-ms 3000

# Generate head-to-head report from captured results
uv run archex benchmark headtohead report --input .archex/headtohead --format markdown
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Run the full validation suite: `uv run ruff check . && uv run ruff format --check . && uv run pyright . && uv run pytest`
4. Submit a PR against `main`
5. PR title should follow conventional commits format (e.g., `feat: add Dart language adapter`)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.
