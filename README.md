# archex

Architecture extraction and codebase intelligence for the agentic era.

archex is a Python library and CLI that transforms any Git repository into structured architectural intelligence and token-budget-aware code context. It serves two consumers from a single index: **human architects** receive an `ArchProfile` with module boundaries, dependency graphs, detected patterns, and interface surfaces; **AI agents** receive a `ContextBundle` with relevance-ranked, syntax-aligned code chunks assembled to fit within a specified token budget.

## Features

- **8 language adapters** — Python, TypeScript/JavaScript, Go, Rust, Java, Kotlin, C#, Swift (tree-sitter AST parsing), extensible via entry points
- **8 public APIs** — `analyze()`, `query()`, `compare()`, `file_tree()`, `file_outline()`, `search_symbols()`, `get_symbol()`, `get_symbols_batch()` + token counting utilities
- **Hybrid retrieval** — BM25 keyword search + optional vector embeddings with reciprocal rank fusion
- **Token budget assembly** — AST-aware chunking, dependency-graph expansion, greedy bin-packing with configurable `ScoringWeights`
- **Structural analysis** — module detection (Louvain), pattern recognition (extensible `PatternRegistry`), interface extraction
- **Cross-repo comparison** — 6 architectural dimensions, no LLM required
- **Delta indexing** — surgical re-index via git diff when only a few files changed; mtime-based fallback for non-git sources; configurable `delta_threshold`
- **Performance** — cache-first query (skips parse on cache hit), delta indexing, parallel parsing, parallel compare, git-aware cache keys; warm queries under 150ms
- **Extensibility** — plugin APIs for language adapters, pattern detectors, chunkers, and scoring weights via entry points and protocols
- **Security** — input validation on git URLs/branches, FTS5 query escaping, cache key validation, `allow_pickle=False` for vector persistence
- **LLM-optional** — entire structural pipeline runs without API calls; LLM enrichment is opt-in

## Installation

```bash
uv add archex
```

### Extras

| Extra                   | What it adds                                      |
| ----------------------- | ------------------------------------------------- |
| `archex[vector]`        | ONNX-based local embeddings (Nomic Code)          |
| `archex[vector-torch]`  | Torch-backed sentence-transformers                |
| `archex[voyage]`        | Voyage Code API embeddings                        |
| `archex[openai]`        | OpenAI API embeddings + LLM enrichment            |
| `archex[anthropic]`     | Anthropic API LLM enrichment                      |
| `archex[mcp]`           | MCP server for agent integration                  |
| `archex[langchain]`     | LangChain retriever integration                   |
| `archex[llamaindex]`    | LlamaIndex retriever integration                  |
| `archex[language-pack]` | Additional grammars via tree-sitter-language-pack |
| `archex[all]`           | All optional dependencies                         |

## Quick Start

### Python API

```python
from archex import analyze, query, compare
from archex.models import RepoSource

# Architectural analysis
profile = analyze(RepoSource(local_path="./my-project"))
for module in profile.module_map:
    print(f"{module.name}: {len(module.files)} files")
for pattern in profile.pattern_catalog:
    print(f"[{pattern.confidence:.0%}] {pattern.name}")

# Implementation context for an agent
bundle = query(
    RepoSource(local_path="./my-project"),
    "How does authentication work?",
    token_budget=8192,
)
print(bundle.to_prompt(format="xml"))

# Query with custom scoring weights
from archex.models import ScoringWeights

bundle = query(
    RepoSource(local_path="./my-project"),
    "database connection pooling",
    scoring_weights=ScoringWeights(relevance=0.8, structural=0.1, type_coverage=0.1),
)

# Cross-repo comparison
result = compare(
    RepoSource(local_path="./project-a"),
    RepoSource(local_path="./project-b"),
    dimensions=["error_handling", "api_surface"],
)
```

### Surgical Lookups

```python
from archex.api import file_tree, file_outline, search_symbols, get_symbol
from archex.models import RepoSource

source = RepoSource(local_path="./my-project")

# Browse repository structure (~2,000 tokens vs 200,000+ for raw listing)
tree = file_tree(source, max_depth=3, language="python")

# Get symbol outline for a single file (~180 tokens vs 4,800 for full file)
outline = file_outline(source, "src/auth/middleware.py")

# Search symbols by name across the codebase
matches = search_symbols(source, "authenticate", kind="function", limit=10)

# Retrieve full source for a specific symbol by stable ID
symbol = get_symbol(source, "src/auth/middleware.py::authenticate#function")

# Batch retrieval for multiple symbols
from archex.api import get_symbols_batch
symbols = get_symbols_batch(source, [
    "src/auth/middleware.py::authenticate#function",
    "src/models/user.py::User#class",
])
```

### Token Counting

```python
from archex.api import get_file_token_count, get_files_token_count, get_repo_total_tokens
from archex.models import RepoSource

source = RepoSource(local_path="./my-project")

# Count tokens before deciding what to include in context
total = get_repo_total_tokens(source)
auth_tokens = get_file_token_count(source, "src/auth/middleware.py")
batch_tokens = get_files_token_count(source, ["src/auth/middleware.py", "src/models/user.py"])
```

### CLI

```bash
# Analyze a local repo or remote URL
archex analyze ./my-project --format json
archex analyze https://github.com/org/repo --format markdown -l python --timing

# Query for implementation context
archex query ./my-project "How does auth work?" --budget 8192 --format xml
archex query ./my-project "connection pooling" --strategy hybrid --timing

# Browse and search
archex tree ./my-project --depth 3 -l python
archex outline ./my-project src/auth/middleware.py
archex symbols ./my-project "authenticate" --kind function --limit 10
archex symbol ./my-project "src/auth/middleware.py::authenticate#function"

# Compare two repositories
archex compare ./project-a ./project-b --dimensions error_handling,api_surface --format markdown

# Manage the analysis cache
archex cache list
archex cache clean --max-age 168
archex cache info

# Benchmark retrieval strategies
archex benchmark run tasks.yaml --strategies bm25,hybrid --output results.json
archex benchmark report results.json --format markdown
archex benchmark delta ./my-project
```

## Agent Integration

archex is designed to be called by coding agents. Three integration paths, ordered by depth:

### Shell Out (any agent)

Any agent that can execute shell commands (Cursor, Claude Code, Copilot, custom frameworks) can use archex directly:

```bash
# Agent needs to understand a foreign codebase — one call, structured output
archex query https://github.com/encode/httpx "connection pooling" --budget 8000 --format xml

# Agent needs the lay of the land before implementation
archex tree https://github.com/encode/httpx --depth 3

# Agent needs a specific function's source
archex symbol ./repo "src/pool.py::ConnectionPool#class"
```

### MCP Server (Claude Code / Claude Desktop)

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "archex": {
      "command": "archex",
      "args": ["mcp"]
    }
  }
}
```

The agent discovers 8 tools automatically:

| Tool                | What it does                                               |
| ------------------- | ---------------------------------------------------------- |
| `analyze_repo`      | Full architectural profile (modules, patterns, interfaces) |
| `query_repo`        | Token-budget context assembly for a natural language query |
| `compare_repos`     | Structural comparison across architectural dimensions      |
| `get_file_tree`     | Annotated directory listing with language/symbol counts    |
| `get_file_outline`  | Symbol hierarchy for a single file (signatures only)       |
| `search_symbols`    | Fuzzy symbol search by name, kind, and language            |
| `get_symbol`        | Full source code for a symbol by its stable ID             |
| `get_symbols_batch` | Batch retrieval of multiple symbols in one call            |

### Python API (agent frameworks)

```python
# LangChain retriever
from archex.integrations.langchain import ArchexRetriever
retriever = ArchexRetriever(source=RepoSource(local_path="./repo"))

# LlamaIndex retriever
from archex.integrations.llamaindex import ArchexRetriever
retriever = ArchexRetriever(source=RepoSource(local_path="./repo"))

# Direct usage in any framework
from archex import query
from archex.models import RepoSource

bundle = query(
    RepoSource(url="https://github.com/encode/httpx"),
    "How does connection pooling work?",
    token_budget=8000,
)
agent_context = bundle.to_prompt(format="xml")
```

### Token Efficiency

archex replaces multi-file exploration loops with single-call retrieval:

| Task                    | Without archex                                       | With archex                      | Savings |
| ----------------------- | ---------------------------------------------------- | -------------------------------- | ------- |
| Understand a subsystem  | ~45,000 tokens (10+ file reads, backtracking)        | ~7,500 tokens (1 `query()` call) | 83%     |
| Get a symbol's source   | ~3,200 tokens (read entire file)                     | ~340 tokens (`get_symbol()`)     | 89%     |
| File structure overview | ~4,800 tokens (read full file)                       | ~180 tokens (`file_outline()`)   | 96%     |
| Repository navigation   | ~200,000+ tokens (directory listing + file skimming) | ~2,000 tokens (`file_tree()`)    | 99%     |

## CLI Reference

### `archex analyze <source>`

Analyze a repository and produce an architecture profile.

| Option              | Default | Description                       |
| ------------------- | ------- | --------------------------------- |
| `--format`          | `json`  | Output format: `json`, `markdown` |
| `-l` / `--language` | all     | Filter by language (repeatable)   |
| `--timing`          | off     | Print timing breakdown to stderr  |

### `archex query <source> <question>`

Query a repository and return a context bundle.

| Option              | Default | Description                              |
| ------------------- | ------- | ---------------------------------------- |
| `--budget`          | `8192`  | Token budget for context assembly        |
| `--format`          | `xml`   | Output format: `xml`, `json`, `markdown` |
| `-l` / `--language` | all     | Filter by language (repeatable)          |
| `--strategy`        | `bm25`  | Retrieval strategy: `bm25`, `hybrid`     |
| `--timing`          | off     | Print timing breakdown to stderr         |

### `archex compare <source_a> <source_b>`

Compare two repositories across architectural dimensions.

| Option              | Default | Description                       |
| ------------------- | ------- | --------------------------------- |
| `--dimensions`      | all 6   | Comma-separated dimension list    |
| `--format`          | `json`  | Output format: `json`, `markdown` |
| `-l` / `--language` | all     | Filter by language (repeatable)   |
| `--timing`          | off     | Print timing breakdown to stderr  |

Supported dimensions: `api_surface`, `concurrency`, `configuration`, `error_handling`, `state_management`, `testing`.

### `archex tree <source>`

Display the annotated file tree of a repository.

| Option              | Default | Description                 |
| ------------------- | ------- | --------------------------- |
| `--depth`           | `5`     | Maximum directory depth     |
| `-l` / `--language` | all     | Filter to specific language |
| `--json`            | off     | Output as JSON              |
| `--timing`          | off     | Print timing breakdown      |

### `archex outline <source> <file_path>`

Display the symbol outline for a single file (signatures, hierarchy, no source bodies).

| Option     | Default | Description            |
| ---------- | ------- | ---------------------- |
| `--json`   | off     | Output as JSON         |
| `--timing` | off     | Print timing breakdown |

### `archex symbols <source> <query>`

Search symbols by name across a repository.

| Option              | Default | Description                               |
| ------------------- | ------- | ----------------------------------------- |
| `--kind`            | all     | Filter by kind: `function`, `class`, etc. |
| `-l` / `--language` | all     | Filter to specific language               |
| `--limit`           | `20`    | Maximum results                           |
| `--json`            | off     | Output as JSON                            |
| `--timing`          | off     | Print timing breakdown                    |

### `archex symbol <source> <symbol_id>`

Retrieve the full source code for a symbol by its stable ID (e.g., `src/auth.py::login#function`).

| Option     | Default | Description            |
| ---------- | ------- | ---------------------- |
| `--json`   | off     | Output as JSON         |
| `--timing` | off     | Print timing breakdown |

### `archex mcp`

Start the MCP (Model Context Protocol) server for agent integration.

Tools exposed: `analyze_repo`, `query_repo`, `compare_repos`, `get_file_tree`, `get_file_outline`, `search_symbols`, `get_symbol`, `get_symbols_batch`.

### `archex cache <subcommand>`

Manage the local analysis cache.

| Subcommand | Options                              | Description            |
| ---------- | ------------------------------------ | ---------------------- |
| `list`     | `--cache-dir`                        | List cached entries    |
| `clean`    | `--max-age N` (hours), `--cache-dir` | Remove expired entries |
| `info`     | `--cache-dir`                        | Show cache summary     |

### `archex benchmark <subcommand>`

Benchmark retrieval strategies against real repositories.

| Subcommand | Description                                       |
| ---------- | ------------------------------------------------- |
| `run`      | Run benchmarks across strategies                  |
| `report`   | Generate formatted reports from results           |
| `delta`    | Delta indexing benchmarks (speedup + correctness) |
| `baseline` | Manage baselines for regression detection         |
| `gate`     | Check results against quality thresholds          |
| `validate` | Validate benchmark task definitions               |

## Extensibility

archex supports plugin APIs via Python entry points and protocols.

### Custom Language Adapters

Register a language adapter in your package's `pyproject.toml`:

```toml
[project.entry-points."archex.language_adapters"]
dart = "mypackage.adapters:DartAdapter"
```

The adapter class must implement the `LanguageAdapter` protocol (see `archex.parse.adapters.base`).

### Custom Pattern Detectors

Register a pattern detector function:

```toml
[project.entry-points."archex.pattern_detectors"]
my_pattern = "mypackage.patterns:detect_my_pattern"
```

Detector signature: `(list[ParsedFile], DependencyGraph) -> DetectedPattern | None`.

### Custom Chunkers

Implement the `Chunker` protocol and pass it to `query()`:

```python
from archex.index.chunker import Chunker

class MyChunker:
    def chunk_file(self, parsed_file, source): ...
    def chunk_files(self, parsed_files, sources): ...

bundle = query(source, question, chunker=MyChunker())
```

### Custom Scoring Weights

Adjust the retrieval ranking formula:

```python
from archex.models import ScoringWeights

# Boost relevance, lower structural weight
weights = ScoringWeights(relevance=0.8, structural=0.1, type_coverage=0.1)
bundle = query(source, question, scoring_weights=weights)
```

## Configuration

archex reads configuration from `~/.archex/config.toml` and `ARCHEX_*` environment variables.

```toml
# ~/.archex/config.toml
[default]
languages = ["python", "typescript"]
cache = true
cache_dir = "~/.archex/cache"
parallel = true
max_file_size = 10000000
delta_threshold = 0.5
```

| Field             | Default            | Description                                                                 |
| ----------------- | ------------------ | --------------------------------------------------------------------------- |
| `languages`       | all supported      | Languages to parse (list of strings)                                        |
| `depth`           | `3`                | Module detection depth                                                      |
| `enrich`          | `false`            | Enable LLM enrichment (requires provider config)                            |
| `cache`           | `true`             | Enable caching of parse/index results                                       |
| `cache_dir`       | `~/.archex/cache`  | Cache storage directory                                                     |
| `max_file_size`   | `10000000` (10 MB) | Skip files larger than this (bytes)                                         |
| `parallel`        | `true`             | Enable parallel parsing and comparison                                      |
| `delta_threshold` | `0.5`              | If more than this fraction of files changed, full re-index instead of delta |

Environment variables override file config: `ARCHEX_CACHE_DIR`, `ARCHEX_PARALLEL`, `ARCHEX_MAX_FILE_SIZE`.

## Language Support

| Language                    | Extensions                   | Symbols                                                                      | Import Resolution                                   |
| --------------------------- | ---------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------- |
| **Python**                  | `.py`                        | Functions, classes, methods, types, constants, decorators                    | `import`, `from...import`, relative imports         |
| **TypeScript / JavaScript** | `.ts`, `.tsx`, `.js`, `.jsx` | Functions, classes, methods, types, interfaces, enums, constants             | ES modules, CommonJS, type-only imports, re-exports |
| **Go**                      | `.go`                        | Functions, methods (pointer/value receivers), structs, interfaces, constants | Package imports with alias support                  |
| **Rust**                    | `.rs`                        | Functions, structs, enums, traits, impl blocks, constants, macros            | `use` statements, `pub`/`pub(crate)` visibility     |
| **Java**                    | `.java`                      | Classes, interfaces, enums, methods, fields, annotations                     | Package imports, static imports, wildcard imports   |
| **Kotlin**                  | `.kt`, `.kts`                | Classes, objects, functions, properties, extensions, companions              | Package imports, alias imports                      |
| **C#**                      | `.cs`                        | Classes, structs, interfaces, enums, methods, properties, events             | `using` directives, namespace-qualified resolution  |
| **Swift**                   | `.swift`                     | Classes, structs, enums, protocols, actors, extensions, functions            | `import` declarations                               |

Adapters are extensible via Python entry points — add a new language without modifying archex core.

## Development

```bash
git clone https://github.com/determ-ai/archex.git
cd archex
uv sync --all-extras

# Run tests (1274 tests, 92% coverage)
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check (strict mode)
uv run pyright
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
