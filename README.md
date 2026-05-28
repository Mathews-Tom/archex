# archex

[![CI](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/archex)](https://pypi.org/project/archex/)
[![Python](https://img.shields.io/pypi/pyversions/archex)](https://pypi.org/project/archex/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/Mathews-Tom/archex/graph/badge.svg)](https://codecov.io/gh/Mathews-Tom/archex)

**Architectural intelligence for code, with no LLM bill.**

archex turns any Git repository into structured architectural intelligence and token-budget-aware code context — fully local, fully deterministic, no API keys. It's the pre-computed map your AI agent reaches for *before* it starts opening files.

- **~91% fewer tokens** sent to your agent vs. raw file loading (median across 10 OSS repos)
- **<150 ms** warm queries against cached indexes
- **8 languages** out of the box — Python, TypeScript/JavaScript, Go, Rust, Java, Kotlin, C#, Swift
- **Zero API cost** — every pipeline stage runs on your machine

> Your agent reads files. archex reads codebases. See [Why archex](docs/WHY_ARCHEX.md).

## Quick Start

```bash
# Install the CLI
uv tool install archex

# Ask a question against any local repo or GitHub URL
archex query ./my-project "How does authentication work?" --budget 8192
archex query https://github.com/encode/httpx "connection pooling" --format xml
```

That's it. No init step, no language config, no API keys.

## Choose Your Path

archex meets agents and humans where they are. Pick the integration that fits:

### 1 — CLI (any agent, any shell)

Drop archex into any agent that can run shell commands (Cursor, Claude Code, Copilot, custom):

```bash
archex query ./repo "Where is cache invalidation handled?" --budget 8192 --format xml
archex tree ./repo --depth 3
archex symbol ./repo "src/auth/middleware.py::authenticate#function"
```

### 2 — MCP server (Claude Code / Claude Desktop)

```json
{
  "mcpServers": {
    "archex": { "command": "archex", "args": ["mcp"] }
  }
}
```

Eight tools register automatically: `analyze_repo`, `query_repo`, `compare_repos`, `get_file_tree`, `get_file_outline`, `search_symbols`, `get_symbol`, `get_symbols_batch`.

### 3 — Python API (your framework)

```python
from archex import analyze, query, compare
from archex.models import RepoSource

bundle = query(
    RepoSource(local_path="./my-project"),
    "How does authentication work?",
    token_budget=8192,
)
print(bundle.to_prompt(format="xml"))
```

LangChain and LlamaIndex retrievers ship in the `[langchain]` and `[llamaindex]` extras.

## What You Get

| | |
| --- | --- |
| **Hybrid retrieval** | BM25F weighted-field search + optional vector embeddings, fused via confidence-weighted RRF and adaptive RSF behind an AvgIDF gate |
| **Intent-aware ranking** | Query intent classifier (definition, architecture, usage, debugging, CLI, general) picks per-intent scoring presets |
| **Token-budget assembly** | AST-aware chunking, deterministic dependency-graph expansion, greedy bin-packing under a configurable budget |
| **Structural analysis** | Module detection (Leiden with Louvain fallback), pattern recognition via an extensible `PatternRegistry`, interface extraction |
| **Cross-repo comparison** | Diff two repos across 6 architectural dimensions — no LLM in the loop |
| **Surgical lookups** | `file_tree`, `file_outline`, `search_symbols`, `get_symbol(s)` — narrow reads that replace whole-file loads |
| **Cross-encoder reranking** | Opt-in `jinaai/jina-reranker-v3` re-scoring of top candidates, local-only, off by default |
| **Delta indexing** | Git-diff-driven incremental reindex when only a few files changed; mtime fallback for non-git sources |
| **Cache-first queries** | Skip parse on cache hit, parallel parsing and comparison, git-aware cache keys |
| **Pipeline observability** | Opt-in `PipelineTrace` with step-level timing for retrieve, expand, score, assemble |
| **LSP enrichment** | Opt-in `[lsap]` extra layers compiler-grade type info onto archex symbols |
| **Extensible** | Language adapters, pattern detectors, chunkers, and scoring weights register via Python entry points |

Full pipeline anatomy lives in [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).

## Installation

The core package handles all 8 languages, structural analysis, and BM25 retrieval with zero API calls. Extras are opt-in:

```bash
uv tool install archex                    # CLI, system-wide
uv add archex                             # project dependency

# Agent integrations
uv tool install "archex[mcp]"             # MCP server
uv add "archex[langchain]"                # LangChain retriever
uv add "archex[llamaindex]"               # LlamaIndex retriever
uv add "archex[lsap]"                     # LSP type enrichment

# Vector retrieval (any one)
uv add "archex[vector]"                   # ONNX local embeddings (no GPU)
uv add "archex[vector-fast]"              # FastEmbed (no GPU)
uv add "archex[vector-torch]"             # sentence-transformers (GPU)

# Everything
uv add "archex[all]"                      # vector + graph + mcp + langchain + llamaindex + language-pack
```

## Usage

### Analyze a repository

```python
from archex import analyze
from archex.models import RepoSource

profile = analyze(RepoSource(local_path="./my-project"))

for module in profile.module_map:
    print(f"{module.name}: {len(module.files)} files")
for pattern in profile.pattern_catalog:
    print(f"[{pattern.confidence:.0%}] {pattern.name}")
```

### Query for context

```python
from archex import query
from archex.models import RepoSource, ScoringWeights

bundle = query(
    RepoSource(local_path="./my-project"),
    "database connection pooling",
    token_budget=8192,
    scoring_weights=ScoringWeights(relevance=0.8, structural=0.1, type_coverage=0.1),
)
print(bundle.to_prompt(format="xml"))
```

### Surgical lookups (skip whole-file reads)

```python
from archex.api import file_tree, file_outline, search_symbols, get_symbol
from archex.models import RepoSource

source = RepoSource(local_path="./my-project")

tree = file_tree(source, max_depth=3, language="python")              # ~2K tokens vs 200K+
outline = file_outline(source, "src/auth/middleware.py")              # ~180 tokens vs 4,800
matches = search_symbols(source, "authenticate", kind="function")
symbol = get_symbol(source, "src/auth/middleware.py::authenticate#function")
```

### Compare two repositories

```python
from archex import compare
from archex.models import RepoSource

result = compare(
    RepoSource(local_path="./project-a"),
    RepoSource(local_path="./project-b"),
    dimensions=["error_handling", "api_surface"],
)
```

## CLI at a Glance

```bash
archex analyze ./repo --format markdown          # architectural profile
archex query ./repo "How does auth work?"        # context bundle (defaults to cwd if omitted)
archex compare ./repo-a ./repo-b                 # cross-repo diff
archex tree ./repo --depth 3                     # annotated file tree
archex outline ./repo src/auth/middleware.py     # symbol outline for one file
archex symbols ./repo "authenticate"             # symbol search
archex symbol ./repo "src/auth.py::login#function"  # full source by stable ID

# Repo-local lifecycle (inside a checked-out repo)
archex init && archex index && archex status
archex dogfood --task archex_query_pipeline
archex reset --force

# Cache and benchmarks
archex cache list | clean --max-age 168 | info
archex benchmark run tasks.yaml --strategies bm25,hybrid
```

Run `archex --help` or any subcommand with `--help` for the full option list.

## Repo-Local Mode

For agent or maintainer workflows tied to a single checked-out repo:

```bash
cd ./my-project
archex init     # creates .archex/, adds it to .gitignore
archex index    # build or refresh
archex status   # is the index fresh? does HEAD match? is the tree dirty?
archex query "Where is cache invalidation handled?" --budget 8192
```

The entire `.archex/` directory is generated state — SQLite index, vector artifacts, dogfood reports — and stays out of source control. `archex status --strict` fails on stale or dirty state, which is useful in CI gates.

## When To Use archex

archex gives AI agents structural priors about codebases they've never seen. Pre-computed map → cheap, fast, complete. File-by-file exploration → expensive, slow, incomplete.

| Capability                        | archex                          | archex + LSAP                       | Claude Code         | LSP                |
| --------------------------------- | ------------------------------- | ----------------------------------- | ------------------- | ------------------ |
| Cold-start codebase understanding | **Yes** — pre-computed map      | **Yes** — structural + semantic     | Slow — sequential   | No — needs session |
| Semantic type resolution          | Syntactic (tree-sitter)         | **Yes** — LSP hover/refs/defs       | Via LLM reasoning   | **Yes** — compiler |
| Token-budget context assembly     | **Yes** — ranked, packed        | **Yes** — type-enriched             | Manual selection    | Not designed for it |
| Cross-repo structural comparison  | **Yes** — 6 dimensions, no LLM  | **Yes**                             | No                  | No                 |
| Offline / CI-embeddable           | **Yes**                         | Partial — needs language server     | No                  | Partial            |
| Works with any agent framework    | **Yes** — CLI, MCP, Python API  | **Yes** — async Python API          | Claude-specific     | Editor-specific    |

## Performance

**Token efficiency** (median across 10 OSS repos spanning Python, JS/TS, Go, Rust — 35 to 2,332 files):

| Operation          | Median savings | Replaces                                            |
| ------------------ | -------------: | --------------------------------------------------- |
| `file_tree()`      |          98.3% | Reading the repo to understand structure            |
| `compare()`        |          97.8% | Reading both repos end-to-end                       |
| `analyze()`        |          96.7% | Manual exploration of modules, patterns, interfaces |
| `search_symbols()` |          91.0% | Grepping + reading every matching file              |
| `query()`          |          90.8% | Multi-file search and backtracking                  |
| `get_symbol()`     |          61.7% | Reading the entire file to grab one function        |

**Retrieval quality** (BM25 baseline, 25-task benchmark corpus, 8K-token budget):

| Category           | Tasks | Recall | Precision |   F1 |  MRR |
| ------------------ | ----: | -----: | --------: | ---: | ---: |
| external-framework |     9 |   0.80 |      0.36 | 0.50 | 0.80 |
| architecture-broad |     3 |   0.67 |      0.26 | 0.38 | 0.83 |
| self-referential   |     6 |   0.56 |      0.20 | 0.29 | 0.83 |
| framework-semantic |     2 |   0.33 |      0.19 | 0.23 | 0.62 |
| external-large     |     5 |   0.27 |      0.11 | 0.16 | 0.25 |

archex excels on mid-size framework repos where keyword vocabulary aligns with code. Very large codebases (django, react, sqlalchemy) need hybrid retrieval and reranking to push past BM25's vocabulary ceiling. Full per-task tables and reproduction steps live in [docs/BENCHMARK_READINESS.md](docs/BENCHMARK_READINESS.md).

## Language Support

| Language                    | Extensions                   | Symbols                                                          |
| --------------------------- | ---------------------------- | ---------------------------------------------------------------- |
| **Python**                  | `.py`                        | Functions, classes, methods, types, constants, decorators        |
| **TypeScript / JavaScript** | `.ts`, `.tsx`, `.js`, `.jsx` | Functions, classes, methods, types, interfaces, enums, constants |
| **Go**                      | `.go`                        | Functions, methods, structs, interfaces, constants               |
| **Rust**                    | `.rs`                        | Functions, structs, enums, traits, impl blocks, macros           |
| **Java**                    | `.java`                      | Classes, interfaces, enums, methods, fields, annotations         |
| **Kotlin**                  | `.kt`, `.kts`                | Classes, objects, functions, properties, extensions              |
| **C#**                      | `.cs`                        | Classes, structs, interfaces, enums, methods, properties         |
| **Swift**                   | `.swift`                     | Classes, structs, enums, protocols, actors, extensions           |

Need another language? Register an adapter via Python entry points — no core changes required.

## Configuration

Configuration cascades from defaults through `~/.archex/config.toml`, repo-local `.archex/settings.toml`, `ARCHEX_*` environment variables, and explicit CLI/API arguments — later sources override earlier ones.

```toml
# ~/.archex/config.toml
[default]
languages = ["python", "typescript"]
cache = true
cache_dir = "~/.archex/cache"
parallel = true
delta_threshold = 0.5
```

Repo-local settings (`archex init` creates this):

```toml
# .archex/settings.toml
[project]
mode = "local"

[index]
cache_dir = ".archex"
vector = false
delta_threshold = 0.5
```

## Extending archex

archex exposes four plugin surfaces via Python entry points and protocols — language adapters, pattern detectors, chunkers, and scoring weights. Register an adapter in your own package:

```toml
[project.entry-points."archex.language_adapters"]
dart = "mypackage.adapters:DartAdapter"
```

Implement the `LanguageAdapter` protocol from `archex.parse.adapters.base` and archex picks it up automatically. The same pattern applies to `archex.pattern_detectors`. See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the full extension surface.

## Development

```bash
git clone https://github.com/Mathews-Tom/archex.git
cd archex
uv sync --all-extras

uv run pytest                    # ~1,960 tests, 85% minimum coverage
uv run ruff check . && uv run ruff format .
uv run pyright                   # strict mode
```

Contribution guidelines and the dogfood gate workflow live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Learn More

- [Why archex](docs/WHY_ARCHEX.md) — the agent token problem this solves
- [System Overview](docs/OVERVIEW.md) — what archex is and isn't
- [System Design](docs/SYSTEM_DESIGN.md) — pipeline anatomy, extensibility surfaces
- [Benchmark Readiness](docs/BENCHMARK_READINESS.md) — full retrieval quality results
- [Roadmap](docs/ROADMAP.md) — what's next

## License

Apache 2.0 — see [LICENSE](LICENSE).
