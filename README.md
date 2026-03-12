# archex

[![CI](https://github.com/determ-ai/archex/actions/workflows/ci.yml/badge.svg)](https://github.com/determ-ai/archex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/archex)](https://pypi.org/project/archex/)
[![Python](https://img.shields.io/pypi/pyversions/archex)](https://pypi.org/project/archex/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/determ-ai/archex/graph/badge.svg)](https://codecov.io/gh/determ-ai/archex)

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
- **Pipeline observability** — opt-in `PipelineTrace` with step-level timing for retrieve, expand, score, assemble stages
- **Unified artifact pipeline** — `produce_artifacts()` single entry point for parse, import-resolve, chunk, edge-build
- **Query normalization** — camelCase/snake_case splitting, bigram compound generation, architecture-intent synonym expansion
- **Quality gates** — CI-embeddable threshold checks for recall, precision, F1, MRR with latency warnings
- **Expansion gating** — weak BM25 seeds (below 10% of max) don't trigger graph expansion; score-relative file cutoff removes noise
- **25-task benchmark corpus** — 6 self-referential + 19 external repos across Python, Go, Rust, JS/TS, covering 5 difficulty categories
- **LLM-optional** — entire structural pipeline runs without API calls; LLM enrichment is opt-in

## Installation

```bash
# CLI tool (system-wide)
uv tool install archex

# Project dependency
uv add archex
```

### Extras

The core package handles all 8 languages, structural analysis, and BM25 retrieval with zero API calls. Extras add optional capabilities:

**Agent integration:**

```bash
uv tool install "archex[mcp]"        # MCP server (Claude Code / Claude Desktop)
uv add "archex[langchain]"           # LangChain retriever
uv add "archex[llamaindex]"          # LlamaIndex retriever
uv add "archex[lsap]"               # LSP enrichment (Python 3.12+, lsp-client)
```

**Hybrid retrieval** (vector embeddings + BM25):

```bash
uv add "archex[vector]"              # ONNX local embeddings (Nomic Code) — no GPU required
uv add "archex[vector-torch]"        # Torch-backed sentence-transformers — GPU-accelerated
```

**LLM enrichment** (optional — enriches architectural decisions with LLM reasoning):

```bash
uv add "archex[openai]"              # OpenAI (gpt-4.1)
uv add "archex[anthropic]"           # Anthropic (claude-sonnet-4-20250514)
```

**Other:**

```bash
uv add "archex[language-pack]"       # Fallback tree-sitter grammars
uv add "archex[all]"                 # Everything
```

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

### When to Use archex

**archex gives AI agents structural priors about codebases they've never seen.**

When an agent encounters a new codebase, it has two options: explore file-by-file (expensive, slow, high risk of missing context) or receive a pre-computed structural map (cheap, fast, complete). archex is the pre-computed map.

| Capability                        | archex                                        | archex + LSAP                                | Claude Code                        | LSP                          |
| --------------------------------- | --------------------------------------------- | -------------------------------------------- | ---------------------------------- | ---------------------------- |
| Cold-start codebase understanding | **Yes** — pre-computed structural map         | **Yes** — structural + semantic              | Slow — sequential file exploration | No — requires active session |
| Semantic type resolution          | No — syntactic (tree-sitter)                  | **Yes** — LSP hover, references, definitions | Via LLM reasoning                  | **Yes** — compiler-grade     |
| Token-budget context assembly     | **Yes** — ranked, packed, dependency-expanded | **Yes** — enriched with type context         | No — agent manually selects        | No — not designed for this   |
| Cross-repo structural comparison  | **Yes** — 6 dimensions, no LLM                | **Yes**                                      | No                                 | No                           |
| Real-time editing support         | No                                            | No                                           | No                                 | **Yes**                      |
| Offline / CI-embeddable           | **Yes**                                       | Partially — needs running language server    | No                                 | Partially                    |
| Works with any agent framework    | **Yes** — CLI, MCP, Python API                | **Yes** — async Python API                   | Claude-specific                    | Editor-specific              |

### LSAP Enrichment (opt-in)

archex symbols can be enriched with LSP type information via `archex[lsap]`. This is opt-in — the core pipeline remains syntactic-only.

```python
from lsp_client import PyrightClient
from archex.api import get_symbol
from archex.integrations.lsap import LSAPEnrichedLookup
from archex.models import RepoSource

# Caller manages the language server lifecycle.
async with PyrightClient("./my-project") as client:
    lookup = LSAPEnrichedLookup(lsp_client=client)
    symbol = get_symbol(RepoSource(local_path="./my-project"), "src/repo.py::get_user#method")
    enriched = await lookup.enrich_symbol(symbol)
    print(enriched.lsap_enrichment.hover.type_signature)
    print(f"{enriched.lsap_enrichment.reference_count} references")
```

### MCP Co-Tool Configuration

archex and an LSP MCP server can run as sibling tools, giving agents both structural context and precision type operations:

```json
{
  "mcpServers": {
    "archex": {
      "command": "archex",
      "args": ["mcp"]
    },
    "lsp": {
      "command": "lsp-cli",
      "args": ["--language", "python", "--root", "./my-project"]
    }
  }
}
```

**Workflow:** archex for structural context (modules, patterns, ranked chunks) → LSP MCP for precision follow-ups (type resolution, find references, go-to-definition).

**Concrete workflows:**

- **Before agent explores new repo**: `archex query ./repo "auth system" --budget 8K` → context bundle → feed to agent's first prompt
- **Before a code review**: `archex analyze ./pr-branch` → architectural impact summary
- **CI gate**: `archex compare ./main ./feature-branch --dimensions api_surface,error_handling` → detect architectural drift

### Benchmarks

#### Retrieval Quality

`query()` retrieval quality measured against human-annotated expected files across 25 benchmark tasks (6 self-referential archex tasks + 19 external open-source repos). Token budget: 8,192. Strategy: BM25.

**Perfect recall (1.00):**

| Task                    | Recall | Precision |    F1 |  MRR |
| ----------------------- | -----: | --------: | ----: | ---: |
| httpx_pooling           |   1.00 |      0.38 |  0.55 | 1.00 |
| mini_redis_async        |   1.00 |      0.50 |  0.67 | 0.50 |
| archex_pattern_detection|   1.00 |      0.25 |  0.40 | 1.00 |
| gin_routing             |   1.00 |      0.50 |  0.67 | 1.00 |
| go_gin_middleware        |   1.00 |      0.38 |  0.55 | 0.50 |
| requests_sessions       |   1.00 |      0.43 |  0.60 | 1.00 |

**High recall (0.67):**

| Task                    | Recall | Precision |    F1 |  MRR |
| ----------------------- | -----: | --------: | ----: | ---: |
| archex_adapter_registry |   0.67 |      0.25 |  0.36 | 1.00 |
| archex_graph_expansion  |   0.67 |      0.33 |  0.44 | 1.00 |
| archex_scoring          |   0.67 |      0.25 |  0.36 | 1.00 |
| click_decorators        |   0.67 |      0.29 |  0.40 | 1.00 |
| express_error_handling  |   0.67 |      0.25 |  0.36 | 1.00 |
| fastapi_routing         |   0.67 |      0.40 |  0.50 | 1.00 |
| flask_blueprints        |   0.67 |      0.40 |  0.50 | 0.50 |
| pydantic_validators     |   0.67 |      0.25 |  0.36 | 0.17 |

**Moderate recall (0.33-0.50):**

| Task                         | Recall | Precision |    F1 |  MRR |
| ---------------------------- | -----: | --------: | ----: | ---: |
| pytest_fixtures              |   0.50 |      0.14 |  0.22 | 1.00 |
| archex_delta_indexing        |   0.33 |      0.12 |  0.18 | 1.00 |
| celery_task_dispatch         |   0.33 |      0.14 |  0.20 | 0.25 |
| django_middleware            |   0.33 |      0.17 |  0.22 | 1.00 |
| express_middleware           |   0.33 |      0.12 |  0.18 | 0.25 |
| fastapi_dependency_injection |   0.33 |      0.25 |  0.29 | 1.00 |
| react_hooks                  |   0.33 |      0.12 |  0.18 | 0.14 |
| rust_tokio_runtime           |   0.33 |      0.14 |  0.20 | 0.33 |
| sqlalchemy_sessions          |   0.33 |      0.14 |  0.20 | 0.50 |
| archex_query_pipeline        |   0.00 |      0.00 |  0.00 | 0.00 |

**Zero recall:**

| Task                         | Recall | Precision |    F1 |  MRR |
| ---------------------------- | -----: | --------: | ----: | ---: |
| django_orm_queries           |   0.00 |      0.00 |  0.00 | 0.00 |

**Summary (BM25, 25 tasks): Mean Recall: 0.58, Mean F1: 0.34, Mean MRR: 0.69, Mean nDCG: 0.52, Mean MAP: 0.40. Avg token savings: 40.2%.**

**By category:**

| Category           | Tasks | Recall | Precision |   F1 |  MRR |
| ------------------ | ----: | -----: | --------: | ---: | ---: |
| external-framework |     9 |   0.80 |      0.36 | 0.50 | 0.80 |
| architecture-broad |     3 |   0.67 |      0.26 | 0.38 | 0.83 |
| self               |     6 |   0.56 |      0.20 | 0.29 | 0.83 |
| framework-semantic |     2 |   0.33 |      0.19 | 0.23 | 0.62 |
| external-large     |     5 |   0.27 |      0.11 | 0.16 | 0.25 |

> **How to read this:** Recall measures what fraction of expected files appear in the context bundle. Precision measures what fraction of returned chunks are from expected files (low by design — archex includes dependency-expanded context beyond the strict expected set). MRR measures how early the first relevant file appears in the ranked results (1.0 = first position). Raw results in [`benchmarks/results/`](benchmarks/results/).
>
> **Honest assessment:** archex excels at mid-size framework repos (80% recall for external-framework tasks) where BM25 keyword matching aligns well with file content and query vocabulary. Self-referential tasks and architecture-broad queries also perform well (MRR 0.83). The primary weakness is very large codebases (django, react, sqlalchemy) where BM25 alone cannot disambiguate generic vocabulary across hundreds of files — external-large recall drops to 27%. Precision remains structurally low because the 8K token budget packs dependency-expanded chunks beyond the strict expected set.

#### Token Efficiency

Token savings measured across 10 open-source repositories (from [`showcase_results.log`](showcase_results.log)) spanning Python, JavaScript, TypeScript, Go, and Rust — from 35 files to 2,332 files:

**`query()` — context retrieval (the operation agents use most):**

| Repository | Language | Files | Repo Tokens | query() Raw | query() Output | query() Savings |
| ---------- | -------- | ----: | ----------: | ----------: | -------------: | --------------: |
| flask      | Python   |    80 |        182K |         68K |          8,158 |           88.0% |
| requests   | Python   |    35 |        131K |        104K |          7,823 |           92.5% |
| fastapi    | Python   |   941 |        774K |        137K |          7,964 |           94.2% |
| django     | Python   | 2,332 |      7,196K |        179K |          8,128 |           95.5% |
| express    | JS       |   141 |        136K |         23K |          7,925 |           66.1% |
| got        | TS       |    77 |        198K |         72K |          7,959 |           88.9% |
| gin        | Go       |    99 |        190K |         92K |          5,670 |           93.9% |
| actix-web  | Rust     |   313 |        619K |        107K |          7,926 |           92.6% |
| oak        | TS       |    61 |        112K |         55K |          7,821 |           85.8% |
| httpx      | Python   |    57 |        172K |         70K |          7,975 |           88.6% |

**Median query() savings: 90.8%.** "query() Raw" = tokens an agent would read by loading every file matching the query's language filter.

> **What this measures and what it doesn't:** Token savings measures compression ratio — how much less data archex sends compared to reading raw files. It does not measure retrieval quality. See the retrieval quality table above for recall, precision, and MRR.

**Per-operation breakdown (all 10 repos):**

| Operation          | Savings Range | Median | What it replaces                                    |
| ------------------ | ------------- | -----: | --------------------------------------------------- |
| `file_tree()`      | 94.9–99.1%    |  98.3% | Reading entire repo to understand structure         |
| `analyze()`        | 81.7–99.3%    |  96.7% | Manual exploration of modules, patterns, interfaces |
| `compare()`        | 79.1–99.7%    |  97.8% | Reading both repos end-to-end                       |
| `query()`          | 66.1–95.5%    |  90.8% | Multi-file search and backtracking                  |
| `get_symbol()`     | 48.4–87.1%    |  61.7% | Reading entire file to extract one function         |
| `search_symbols()` | 85.9–96.1%    |  91.0% | Grepping + reading matching files                   |

> **Aggregate numbers:** Summing raw tokens across all 8 operations gives a per-repo "total savings" of 84–99% (median 96.8%). This aggregate inflates the picture because it includes `compare()` (which counts both repos' tokens) and treats all operations as a single atomic call. The per-operation breakdown above is more honest.

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

| Subcommand         | Description                                      |
| ------------------ | ------------------------------------------------ |
| `run`              | Run benchmarks across strategies                 |
| `report`           | Generate formatted reports from results          |
| `gate`             | Check results against quality thresholds         |
| `validate`         | Validate benchmark task definitions              |
| `baseline save`    | Save current results as golden baseline          |
| `baseline compare` | Compare current results against a saved baseline |
| `delta run`        | Run delta indexing benchmarks                    |
| `delta gate`       | Check delta results against thresholds           |
| `delta report`     | Generate delta benchmark reports                 |

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
from archex.pipeline.chunker import Chunker

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

# Run tests (1541 tests, 92% coverage, 85% minimum enforced)
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check (strict mode)
uv run pyright
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
