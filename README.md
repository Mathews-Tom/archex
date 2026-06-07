# archex

[![CI](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/archex)](https://pypi.org/project/archex/)
[![Python](https://img.shields.io/pypi/pyversions/archex)](https://pypi.org/project/archex/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/Mathews-Tom/archex/graph/badge.svg)](https://codecov.io/gh/Mathews-Tom/archex)

**archex gives AI agents the right code context with fewer tokens.**

Agents waste context windows by crawling files one at a time. archex indexes a repository locally, ranks the relevant code, expands the dependency/type context, and emits a compact XML/JSON/Markdown bundle for the downstream agent or MCP client to explain. It does not call a hosted LLM, generate prose, or require API keys.

Latest local 35-task benchmark, product-default `archex_query` vs the previous accepted baseline:

| Product-default metric | Previous baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Mean returned tokens | 7,110 | 6,037 | -15.1% |
| Weighted raw-baseline savings | 66.2% | 71.3% | +5.1 pts |
| Mean recall | 0.629 | 0.819 | +0.190 |
| Mean token efficiency | 0.351 | 0.702 | +0.351 |
| Dogfood regressions | — | 0 | pass |

> Your agent reads files. archex reads codebases. See [Why archex](docs/WHY_ARCHEX.md).

## How archex works

```text
Repository → local index → intent classifier → hybrid retrieval → dependency/type expansion → token-budgeted bundle → agent / MCP client
```

archex optimizes the bundle the agent reads, not a generated answer. Explanation queries run outside archex: the CLI, Python API, LangChain/LlamaIndex retrievers, or MCP server emits structured context; the caller decides which model, prompt, or editing workflow consumes it.

Core properties:

- **Local first** — BM25F, optional local embeddings, optional local reranking. No hosted LLM inference.
- **Token-budgeted by design** — definition lookups stay small; broad architecture questions get larger budgets; explicit `--budget` still overrides.
- **Structured for agents** — XML, JSON, and Markdown bundles include ranked chunks, symbol metadata, imports, type context, dependency expansion, and provenance.
- **Language-aware** — Python, TypeScript/JavaScript, Go, Rust, Java, Kotlin, C#, and Swift through tree-sitter adapters.
- **Deterministic architecture views** — modules, symbols, patterns, dependency graph, file tree, impact analysis, onboarding, and cross-repo comparisons.

## Quick start

```bash
# Install the CLI
uv tool install archex

# Query any local repository
archex query ./my-project "How does authentication work?"

# Emit XML for an agent prompt or MCP-style workflow
archex query ./my-project "Where is connection pooling implemented?" --format xml

# Override the adaptive budget only when the caller needs a hard cap
archex query ./my-project "Explain the architecture" --budget 12000
```

No init step, language config, hosted model, or API key is required for ad-hoc local queries. GitHub URLs are also supported when network access is available:

```bash
archex query https://github.com/encode/httpx "Where is connection pooling implemented?" --format xml
```

## Choose your integration

### CLI: any agent, any shell

```bash
archex query ./repo "Where is cache invalidation handled?" --format xml
archex tree ./repo --depth 3
archex symbol ./repo "src/auth/middleware.py::authenticate#function"
```

### MCP server: Claude Code, Claude Desktop, and MCP clients

```json
{
  "mcpServers": {
    "archex": { "command": "archex", "args": ["mcp"] }
  }
}
```

Eight tools register automatically: `analyze_repo`, `query_repo`, `compare_repos`, `get_file_tree`, `get_file_outline`, `search_symbols`, `get_symbol`, and `get_symbols_batch`.

### Python API: applications and retrieval frameworks

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

## What archex gives you

| Outcome | Capabilities |
| --- | --- |
| **Find the right files** | BM25F weighted-field search, optional local vector retrieval, confidence-weighted RRF, local cross-encoder reranking, path/symbol boosts, dependency expansion. |
| **Spend fewer tokens** | Intent-routed budgets, file-diverse packing, nested-range suppression, raw-file baselines, token-efficiency gates, honest MCP envelope accounting. |
| **Give agents structured context** | XML, JSON, and Markdown context bundles with ranked chunks, provenance, imports, type definitions, and stable symbol IDs. |
| **Understand architecture deterministically** | Module detection, pattern recognition, interface extraction, architecture graph export, onboarding, impact analysis, and cross-repo comparison. |
| **Stay local and CI-friendly** | Repo-local `.archex/` indexes, generated artifacts outside source control, no hosted model dependency, deterministic gates. |

Full pipeline anatomy lives in [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).

## What archex is not

- **Not a chatbot** — it emits context bundles; another agent or LLM explains.
- **Not a hosted RAG service** — indexes and retrieval run locally unless you explicitly query a remote Git URL.
- **Not a vector database** — vector search is optional; BM25 and structural signals are first-class.
- **Not an LSP replacement** — use LSAP/LSP where compiler-backed type resolution matters; archex packages repository-scale context for agents.
- **Not a prompt template library** — output is structured retrieval evidence, not prompt prose.

## Installation

The core package handles the supported languages, structural analysis, and BM25 retrieval with zero API calls. Extras are opt-in:

```bash
uv tool install archex                    # CLI, system-wide
uv add archex                             # project dependency

# Agent integrations
uv tool install "archex[mcp]"             # MCP server
uv add "archex[langchain]"                # LangChain retriever
uv add "archex[llamaindex]"               # LlamaIndex retriever
uv add "archex[lsap]"                     # LSP type enrichment

# Vector retrieval
uv add "archex[vector]"                   # ONNX local embeddings
uv add "archex[vector-fast]"              # FastEmbed
uv add "archex[vector-torch]"             # sentence-transformers / torch
uv add "archex[splade]"                   # SPLADE sparse retrieval

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

### Surgical lookups that replace whole-file reads

```python
from archex.api import file_outline, file_tree, get_symbol, search_symbols
from archex.models import RepoSource

source = RepoSource(local_path="./my-project")

tree = file_tree(source, max_depth=3, language="python")
outline = file_outline(source, "src/auth/middleware.py")
matches = search_symbols(source, "authenticate", kind="function")
symbol = get_symbol(source, "src/auth/middleware.py::authenticate#function")
```

### Compare repositories

```python
from archex import compare
from archex.models import RepoSource

result = compare(
    RepoSource(local_path="./project-a"),
    RepoSource(local_path="./project-b"),
    dimensions=["error_handling", "api_surface"],
)
```

## CLI at a glance

```bash
archex analyze ./repo --format markdown              # architecture profile
archex query ./repo "How does auth work?"            # intent-budgeted context bundle
archex query ./repo "Find the query pipeline" --format xml
archex compare ./repo-a ./repo-b                     # cross-repo architectural diff
archex tree ./repo --depth 3                         # annotated file tree
archex outline ./repo src/auth/middleware.py         # symbol outline for one file
archex symbols ./repo "authenticate"                 # symbol search
archex symbol ./repo "src/auth.py::login#function"   # full source by stable ID
archex explain ./repo src/auth.py                    # deterministic structural explanation
archex graph export ./repo --format json             # architecture graph artifact
archex impact ./repo src/auth.py                     # deterministic blast-radius analysis
archex onboard ./repo                                # deterministic onboarding guide

# Repo-local lifecycle
archex init && archex index && archex status
archex dogfood . --all --baseline benchmarks/dogfood_baseline.json --format dogfood-delta
archex reset --force

# Cache and benchmarks
archex cache list | clean --max-age 168 | info
archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e
archex benchmark gate --input .archex/e2e --baseline .archex/e2e-baseline --warn-latency-ms 3000
```

Run `archex --help` or any subcommand with `--help` for the full option list.

## Repo-local mode

For agent or maintainer workflows tied to a single checked-out repo:

```bash
cd ./my-project
archex init     # creates .archex/, adds it to .gitignore
archex index    # build or refresh
archex status   # is the index fresh? does HEAD match? is the tree dirty?
archex query "Where is cache invalidation handled?"
```

The entire `.archex/` directory is generated state — SQLite index, vector artifacts, graph artifacts, cache metadata, and dogfood reports — and stays out of source control. `archex status --strict` fails on stale or dirty state, which is useful in CI gates.

## When to use archex

archex gives AI agents structural priors about codebases they have not seen. Pre-computed map → cheap, fast, complete. File-by-file exploration → expensive, slow, incomplete.

| Capability | archex | archex + LSAP | Claude Code | LSP |
| --- | --- | --- | --- | --- |
| Cold-start codebase understanding | **Yes** — pre-computed map | **Yes** — structural + semantic | Slow — sequential | No — needs session |
| Semantic type resolution | Syntactic tree-sitter signals | **Yes** — LSP hover/refs/defs | Via LLM reasoning | **Yes** — compiler-backed |
| Token-budget context assembly | **Yes** — ranked, packed | **Yes** — type-enriched | Manual selection | Not designed for it |
| Cross-repo structural comparison | **Yes** — deterministic dimensions | **Yes** | No | No |
| Offline / CI embeddable | **Yes** | Partial — needs language server | No | Partial |
| Works with any agent framework | **Yes** — CLI, MCP, Python API | **Yes** — async Python API | Claude-specific | Editor-specific |

## Performance and gates

archex optimizes the amount of context the downstream agent must read, not recall alone. Benchmark reports track recall, precision, F1, MRR, NDCG, MAP, latency, returned tokens, raw-file baselines, and token efficiency. Token efficiency is higher-is-better: `1 - returned_tokens / accessed_file_tokens`.

Latest local 35-task benchmark, compared with the previous accepted baseline:

| Strategy | Returned tokens | Weighted raw-baseline savings | Recall | Token efficiency |
| --- | ---: | ---: | ---: | ---: |
| `archex_query` | 7,110 → 6,037 (-15.1%) | 66.2% → 71.3% | 0.629 → 0.819 | 0.351 → 0.702 |
| `archex_query_fusion` | 7,173 → 7,293 (+1.7%) | 65.9% → 65.4% | 0.627 → 0.809 | 0.307 → 0.612 |
| `archex_query_fusion_rerank` | 7,178 → 7,307 (+1.8%) | 65.9% → 65.3% | 0.627 → 0.818 | 0.307 → 0.612 |

Release gates are intentionally tied to the product contract:

- hard fail if product-default token efficiency falls below the measured floor (`0.08`);
- hard fail if any gated strategy regresses recall against the accepted baseline when `--baseline` is supplied;
- hard fail if a baseline row is missing;
- warn, but do not fail, on absolute non-token rows such as rank-2 MRR or low recall rows already accepted in the baseline;
- dogfood must report zero regressions.

Reproduce the full retrieval gate and dogfood delta locally:

```bash
scripts/benchmark_pipeline.sh
```

The script removes prior `.archex/e2e-tokens` output, writes a fresh `.docs/pipeline.log`, runs benchmark generation, runs the baseline-aware gate, and then runs dogfood even when the gate fails. It exits non-zero at the end if any step failed.

## Language support

| Language | Extensions | Symbols |
| --- | --- | --- |
| **Python** | `.py` | Functions, classes, methods, types, constants, decorators |
| **TypeScript / JavaScript** | `.ts`, `.tsx`, `.js`, `.jsx` | Functions, classes, methods, types, interfaces, enums, constants |
| **Go** | `.go` | Functions, methods, structs, interfaces, constants |
| **Rust** | `.rs` | Functions, structs, enums, traits, impl blocks, macros |
| **Java** | `.java` | Classes, interfaces, enums, methods, fields, annotations |
| **Kotlin** | `.kt`, `.kts` | Classes, objects, functions, properties, extensions |
| **C#** | `.cs` | Classes, structs, interfaces, enums, methods, properties |
| **Swift** | `.swift` | Classes, structs, enums, protocols, actors, extensions |

Need another language? Register an adapter via Python entry points — no core changes required.

## Configuration

Configuration cascades from defaults through `~/.archex/config.toml`, repo-local `.archex/settings.toml`, `ARCHEX_*` environment variables, and explicit CLI/API arguments. Later sources override earlier ones.

```toml
# ~/.archex/config.toml
[default]
languages = ["python", "typescript"]
cache = true
cache_dir = "~/.archex/cache"
parallel = true
delta_threshold = 0.5
```

Repo-local settings created by `archex init`:

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

archex exposes plugin surfaces via Python entry points and protocols: language adapters, pattern detectors, chunkers, scoring weights, benchmark strategies, and embedders. Register an adapter in your own package:

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

## Learn more

- [Why archex](docs/WHY_ARCHEX.md) — the agent token problem this solves
- [System Overview](docs/OVERVIEW.md) — what archex is and is not
- [System Design](docs/SYSTEM_DESIGN.md) — pipeline anatomy and extensibility surfaces
- [Benchmark Readiness](docs/BENCHMARK_READINESS.md) — retrieval quality history
- [Roadmap](docs/ROADMAP.md) — what is next

## License

Apache 2.0 — see [LICENSE](LICENSE).
