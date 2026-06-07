# archex

[![CI](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/archex)](https://pypi.org/project/archex/)
[![Python](https://img.shields.io/pypi/pyversions/archex)](https://pypi.org/project/archex/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/Mathews-Tom/archex/graph/badge.svg)](https://codecov.io/gh/Mathews-Tom/archex)

**Local codebase intelligence and token-budgeted context bundles for AI agents.**

archex indexes a repository once, then answers codebase questions with compact, structured context: ranked code chunks, symbol metadata, dependency expansion, and retrieval provenance. It does not run a generative model. The downstream agent or MCP client uses the emitted bundle to explain, edit, or navigate the code.

- **Local-first retrieval** — BM25, optional local embeddings, optional local reranking; no hosted LLM inference or API keys required.
- **Intent-aware budgets** — definition lookups stay small; broad architecture questions keep a larger budget.
- **Structured output** — XML, JSON, and Markdown bundles designed for agent prompts and MCP clients.
- **8 languages** out of the box — Python, TypeScript/JavaScript, Go, Rust, Java, Kotlin, C#, Swift.
- **Deterministic architecture views** — modules, symbols, patterns, dependency graph, file tree, and cross-repo comparisons.

> Your agent reads files. archex reads codebases. See [Why archex](docs/WHY_ARCHEX.md).

## Quick Start

```bash
# Install the CLI
uv tool install archex

# Ask a question against any local repo or GitHub URL
archex query ./my-project "How does authentication work?"
archex query https://github.com/encode/httpx "Where is connection pooling implemented?" --format xml

# Override the adaptive budget when you need a hard cap
archex query ./my-project "Explain the architecture" --budget 12000
```

No init step, language config, hosted model, or API key is required for ad-hoc queries.

## Choose Your Path

archex meets agents and humans where they are. Pick the integration that fits:

### 1 — CLI (any agent, any shell)

Drop archex into any agent that can run shell commands (Cursor, Claude Code, Copilot, custom):

```bash
archex query ./repo "Where is cache invalidation handled?" --format xml
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
from archex import analyze, query
from archex.models import RepoSource

source = RepoSource(local_path="./my-project")
profile = analyze(source)
print(f"{len(profile.module_map)} modules")
bundle = query(source, "How does authentication work?")
print(bundle.to_prompt(format="xml"))
```

LangChain and LlamaIndex retrievers ship in the `[langchain]` and `[llamaindex]` extras.

## What You Get

| | |
| --- | --- |
| **Hybrid retrieval** | BM25F weighted-field search plus optional local vector retrieval, confidence-weighted RRF, and adaptive score fusion. |
| **Intent-aware budgets** | Query intent classification picks scoring weights and a default token cap: symbol/definition lookups stay small, broad architecture queries stay larger. |
| **Context assembly** | AST-aware chunking, dependency-graph expansion, type definitions, imports, and greedy packing into XML/JSON/Markdown bundles. |
| **Honest token accounting** | Benchmark gates track returned context, raw file baselines, token efficiency, latency, and MCP envelope overhead. |
| **Structural analysis** | Module detection, pattern recognition, interface extraction, architecture graph export, deterministic onboarding, and blast-radius impact analysis. |
| **Surgical lookups** | `tree`, `outline`, `symbols`, `symbol`, and MCP equivalents for narrow reads that replace whole-file loading. |
| **Cross-repo comparison** | Deterministic comparison across API surface, state management, error handling, concurrency, testing, and configuration. |
| **Local reranking** | Opt-in `jinaai/jina-reranker-v3` cross-encoder reranking for top retrieval candidates. |
| **Repo-local mode** | `.archex/` stores generated indexes, vector artifacts, graph artifacts, cache metadata, and dogfood reports outside source control. |
| **Agent integrations** | CLI, MCP server, Python API, LangChain retriever, and LlamaIndex retriever. |

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
from archex.models import RepoSource

bundle = query(
    RepoSource(local_path="./my-project"),
    "Where is database connection pooling implemented?",
)

print(bundle.to_prompt(format="xml"))
```

`query()` returns a `ContextBundle`, not a generated explanation. Feed that bundle to your agent, MCP client, or downstream LLM. Pass `token_budget=...` when the caller needs an explicit override; otherwise archex uses the intent-routed budget.

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
archex analyze ./repo --format markdown          # architecture profile
archex query ./repo "How does auth work?"        # intent-budgeted context bundle
archex compare ./repo-a ./repo-b                 # cross-repo architectural diff
archex tree ./repo --depth 3                     # annotated file tree
archex outline ./repo src/auth/middleware.py     # symbol outline for one file
archex symbols ./repo "authenticate"             # symbol search
archex symbol ./repo "src/auth.py::login#function"  # full source by stable ID
archex explain ./repo src/auth.py                # deterministic structural explanation
archex graph export ./repo --format json         # architecture graph artifact
archex impact ./repo src/auth.py                 # deterministic blast-radius analysis
archex onboard ./repo                            # deterministic onboarding guide

# Repo-local lifecycle
archex init && archex index && archex status
archex dogfood . --all --baseline benchmarks/dogfood_baseline.json --format dogfood-delta
archex reset --force

# Cache and benchmarks
archex cache list | clean --max-age 168 | info
archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e
archex benchmark gate --input .archex/e2e --warn-latency-ms 3000
```

Run `archex --help` or any subcommand with `--help` for the full option list.

## Repo-Local Mode

For agent or maintainer workflows tied to a single checked-out repo:

```bash
cd ./my-project
archex init     # creates .archex/, adds it to .gitignore
archex index    # build or refresh
archex status   # is the index fresh? does HEAD match? is the tree dirty?
archex query "Where is cache invalidation handled?"
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

## Performance and Gates

archex optimizes for the amount of context the downstream agent must read, not recall alone. Benchmark reports track recall, precision, F1, MRR, NDCG, MAP, latency, returned tokens, raw-file baselines, and token efficiency.

Current 35-task local benchmark snapshot for the product-default `archex_query` strategy, compared with the accepted Tier 2.5 run:

| Metric | Tier 2.5 | Current | Delta |
| --- | ---: | ---: | ---: |
| Mean returned tokens | 7,110 | 5,528 | -22.3% |
| Mean recall | 0.629 | 0.694 | +0.065 |
| Mean token efficiency | 0.351 | 0.296 | -0.056 |

Fusion and rerank currently reduce mean returned tokens by roughly 11% while improving mean recall from 0.627 to 0.640. Gate thresholds are intentionally stricter than averages: a run fails if individual tasks regress recall or fall below the product-default token-efficiency floor. Treat the gate output, not the aggregate table, as the release signal.

Reproduce the full retrieval gate locally:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e
uv run archex benchmark gate --input .archex/e2e --warn-latency-ms 3000
```

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

uv run pytest                    # full test suite, 85% minimum coverage
uv run ruff check && uv run ruff format --check .
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
