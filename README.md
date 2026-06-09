# archex

[![CI](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathews-Tom/archex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/archex)](https://pypi.org/project/archex/)
[![Python](https://img.shields.io/pypi/pyversions/archex)](https://pypi.org/project/archex/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/Mathews-Tom/archex/graph/badge.svg)](https://codecov.io/gh/Mathews-Tom/archex)

**archex gives AI agents the right code context with fewer tokens.**

Your agent reads files. archex reads codebases.

Today, an agent that needs to understand unfamiliar code opens one file, then follows an import, then opens a type definition, then backtracks — burning 15,000 tokens to end up with maybe 60% of what it needed. archex replaces that whole exploration loop with one call: it indexes the repository locally, ranks the relevant code, pulls in the dependencies and type context the agent would have had to chase by hand, and hands back a compact, structured bundle. No hosted LLM, no prose, no API key.

> Want the longer version of the story first? Read [Why archex](docs/WHY_ARCHEX.md).

## Try it in 30 seconds

```bash
# Install the CLI (no init, no config, no API key)
uv tool install archex

# Ask any local repository a question
archex query ./my-project "How does authentication work?"

# Point it at a public GitHub repo when you have network access
archex query https://github.com/encode/httpx "Where is connection pooling implemented?"
```

That's the whole on-ramp. There is no project to set up and nothing to register — archex indexes the repo on the fly and answers.

## What you get back

archex returns a *context bundle*, not an answer. Here is a trimmed XML bundle for "How does authentication work?":

```xml
<context query="How does authentication work?">
  <structural-context>
    <file-tree><![CDATA[
src/auth/
  middleware.py
  tokens.py
  models.py
    ]]></file-tree>
  </structural-context>
  <chunks>
    <chunk file="src/auth/middleware.py" lines="42-78" symbol="authenticate" score="0.9312" tokens="284">
      <imports><![CDATA[from auth.tokens import verify_jwt]]></imports>
      <code><![CDATA[
def authenticate(request: Request) -> User:
    token = extract_bearer(request)
    claims = verify_jwt(token)
    return load_user(claims.sub)
      ]]></code>
    </chunk>
  </chunks>
  <type-definitions>
    <type-def file="src/auth/models.py" symbol="User" lines="10-24"><![CDATA[
@dataclass
class User: ...
    ]]></type-def>
  </type-definitions>
  <dependencies>
    <internal>auth.tokens.verify_jwt</internal>
    <external>pyjwt</external>
  </dependencies>
</context>
```

Ranked code chunks, the imports and type definitions they depend on, the dependency chain, and provenance — packed to a token budget. You decide which model, prompt, or editing workflow consumes it. Prefer JSON or Markdown instead? Add `--format json` or `--format markdown`.

## Why it helps

On the latest local 35-task benchmark, the product-default `archex query` reduced what the downstream agent had to read versus a raw-file crawl while keeping retrieval quality high:

- **71.3% fewer returned tokens than reading raw files** — archex returns 6,037 mean tokens per query instead of making the agent read the full raw-file baseline.
- **Recall 0.819** — the agent still gets most of what it actually needed.
- **Token efficiency 0.702** — `1 − returned_tokens / accessed_file_tokens`, higher is better.

Every number is measured and gated in CI, not asserted. Full baseline-vs-current release tables and gate rules live in [Performance and gates](#performance-and-gates).

## Who it's for

| If you are a…              | archex gives you…                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| **AI agent developer**     | Precise codebase context without burning the context window — one call replaces multi-file crawling.  |
| **Solo developer**         | An architectural map and pattern detection for an unfamiliar repo in minutes instead of hours.        |
| **Team lead**              | Cross-repo comparison and architectural-drift detection you can wire into CI.                          |
| **Open-source contributor**| A read on a project's structure before you open a PR — find the right place to make the change.        |

## Choose your integration

### CLI — any agent, any shell

```bash
archex query ./repo "Where is cache invalidation handled?" --format xml
archex tree ./repo --depth 3
archex symbol ./repo "src/auth/middleware.py::authenticate#function"
```

### MCP server — Claude Code, Claude Desktop, and other MCP clients

```json
{
  "mcpServers": {
    "archex": { "command": "archex", "args": ["mcp"] }
  }
}
```

Eight tools register automatically: `analyze_repo`, `query_repo`, `compare_repos`, `get_file_tree`, `get_file_outline`, `search_symbols`, `get_symbol`, and `get_symbols_batch`.

### Python API — applications and retrieval frameworks

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

## What archex does well

| Outcome | How |
| --- | --- |
| **Find the right files** | BM25F weighted-field search, optional local vector retrieval, confidence-weighted RRF, local cross-encoder reranking, path/symbol boosts, dependency expansion. |
| **Spend fewer tokens** | Intent-routed budgets, file-diverse packing, nested-range suppression, raw-file baselines, token-efficiency gates, honest MCP envelope accounting. |
| **Give agents structured context** | XML, JSON, and Markdown bundles with ranked chunks, provenance, imports, type definitions, and stable symbol IDs. |
| **Understand architecture deterministically** | Module detection, pattern recognition, interface extraction, architecture graph export, onboarding, impact analysis, and cross-repo comparison. |
| **Stay local and CI-friendly** | Repo-local `.archex/` indexes, generated artifacts outside source control, no hosted-model dependency, deterministic gates. |

Core properties:

- **Local first** — BM25F, optional local embeddings, optional local reranking. No hosted LLM inference.
- **Token-budgeted by design** — definition lookups stay small; broad architecture questions get larger budgets; explicit `--budget` always overrides.
- **Structured for agents** — bundles carry ranked chunks, symbol metadata, imports, type context, dependency expansion, and provenance.
- **Language-aware** — Python, TypeScript/JavaScript, Go, Rust, Java, Kotlin, C#, and Swift via tree-sitter adapters.
- **Deterministic architecture views** — modules, symbols, patterns, dependency graph, file tree, impact analysis, onboarding, and cross-repo comparisons.

```text
Repository → local index → intent classifier → hybrid retrieval → dependency/type expansion → token-budgeted bundle → agent / MCP client
```

Full pipeline anatomy lives in [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).

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

`query()` returns a `ContextBundle`, not a generated explanation. Feed that bundle to your agent, MCP client, or downstream LLM. Pass `token_budget=...` when you need an explicit override; otherwise archex uses the intent-routed budget.

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

## archex vs. the alternatives

archex gives AI agents structural priors about codebases they have not seen. A pre-computed map is cheap, fast, and complete; file-by-file exploration is expensive, slow, and incomplete.

| Capability | archex | archex + LSAP | Claude Code | LSP |
| --- | --- | --- | --- | --- |
| Cold-start codebase understanding | **Yes** — pre-computed map | **Yes** — structural + semantic | Slow — sequential | No — needs session |
| Semantic type resolution | Syntactic tree-sitter signals | **Yes** — LSP hover/refs/defs | Via LLM reasoning | **Yes** — compiler-backed |
| Token-budget context assembly | **Yes** — ranked, packed | **Yes** — type-enriched | Manual selection | Not designed for it |
| Cross-repo structural comparison | **Yes** — deterministic dimensions | **Yes** | No | No |
| Offline / CI embeddable | **Yes** | Partial — needs language server | No | Partial |
| Works with any agent framework | **Yes** — CLI, MCP, Python API | **Yes** — async Python API | Claude-specific | Editor-specific |

## Performance and gates

archex optimizes the amount of context the downstream agent must read, not recall alone. Benchmark reports track recall, precision, F1, MRR, NDCG, MAP, latency, returned tokens, raw-file baselines, and token efficiency (`1 − returned_tokens / accessed_file_tokens`, higher is better).
Default embedder, reranker, and strategy switches are evidence-gated in `docs/RETRIEVAL_DEFAULT_DECISIONS.md`; the 2026-06-09 retrieval-default run kept `archex_query` as the product default and did not refresh `benchmarks/dogfood_baseline.json`.

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
bash scripts/benchmark_pipeline.sh
```

The script resolves the repo root from its own location, removes prior `.archex/e2e-tokens` output, writes a fresh `.docs/pipeline.log`, streams the run to the terminal that started it, runs benchmark generation, runs the baseline-aware gate, and then runs dogfood even when the gate fails. It exits non-zero at the end if any step failed.

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

## What archex is not

- **Not a chatbot** — it emits context bundles; another agent or LLM does the explaining.
- **Not a hosted RAG service** — indexing and retrieval run locally unless you explicitly query a remote Git URL.
- **Not a vector database** — vector search is optional; BM25 and structural signals are first-class.
- **Not an LSP replacement** — use LSAP/LSP where compiler-backed type resolution matters; archex packages repository-scale context for agents.
- **Not a prompt template library** — output is structured retrieval evidence, not prompt prose.

## Development

```bash
git clone https://github.com/Mathews-Tom/archex.git
cd archex
uv sync --all-extras

uv run pytest                    # 2061 tests, 91% coverage (85% gate floor)
uv run ruff check && uv run ruff format --check .
uv run pyright                   # strict mode
```

Contribution guidelines and the dogfood gate workflow live in [CONTRIBUTING.md](CONTRIBUTING.md). New contributors welcome — language adapters and pattern detectors are the easiest first PRs.

## Learn more

- [Why archex](docs/WHY_ARCHEX.md) — the agent token problem this solves
- [System Overview](docs/OVERVIEW.md) — what archex is and is not
- [System Design](docs/SYSTEM_DESIGN.md) — pipeline anatomy and extensibility surfaces
- [Benchmark Readiness](docs/BENCHMARK_READINESS.md) — retrieval quality history
- [Roadmap](docs/ROADMAP.md) — what is next

## License

Apache 2.0 — see [LICENSE](LICENSE).
