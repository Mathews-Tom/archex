# Contributing to archex

## Development Setup

```bash
git clone https://github.com/determ-ai/archex.git
cd archex
uv sync --all-extras
```

## Running Tests

```bash
# Full test suite (1274 tests)
uv run pytest

# With coverage report
uv run pytest --cov-report=html

# Skip slow tests
uv run pytest -m "not slow"

# Run a specific test file
uv run pytest tests/test_api.py -v
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

All checks must pass before submitting a PR. CI runs lint, format check, type check, and tests on Python 3.11, 3.12, and 3.13.

## Code Style

- **Formatter/Linter:** ruff (config in `pyproject.toml`)
- **Line length:** 100
- **Type checking:** pyright strict mode
- **Target Python:** 3.11+

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
uv run archex benchmark run benchmarks/tasks/ --strategies bm25

# Run specific tasks
uv run archex benchmark run benchmarks/tasks/ --filter "archex_*"

# Check quality gate
uv run archex benchmark gate benchmarks/results/latest.json

# Generate report
uv run archex benchmark report benchmarks/results/latest.json --format markdown
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Run the full validation suite: `uv run ruff check . && uv run ruff format --check . && uv run pyright . && uv run pytest`
4. Submit a PR against `main`
5. PR title should follow conventional commits format (e.g., `feat: add Dart language adapter`)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.
