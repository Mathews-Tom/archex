# archex — Why archex

> Your agent reads files. archex reads codebases.

---

## The Problem Every AI Developer Hits

You're building with AI agents — Cursor, Claude Code, Copilot, or your own framework. The agent needs to understand code it hasn't seen before. Maybe it's a library you're integrating, a microservice another team owns, or an open-source project you need to extend.

What happens today:

1. The agent opens a file. 2,000 tokens consumed. Most of it irrelevant.
2. The agent needs context. It opens another file. 3,500 more tokens.
3. It follows an import. Another file. 1,800 tokens.
4. It needs a type definition. Opens the models file. 4,000 tokens.
5. It still doesn't understand the dependency chain. Opens two more files.
6. **Total: 15,000+ tokens consumed. The agent has maybe 60% of what it needs. It's missing the config layer, the error handling path, and a critical interface it didn't know to look for.**

This is the fundamental problem: agents explore codebases the same way a junior developer would on their first day — opening files one by one, scanning for relevant code, losing context, backtracking.

archex replaces that entire exploration loop with a single call.

---

## What archex Does

archex indexes a codebase once using AST parsing (tree-sitter), builds a dependency graph, and then serves two types of consumers:

**For agents:** A single `query_repo("How does authentication work?", budget=8000)` call returns a pre-assembled context block — the right code chunks, their type definitions, the dependency chain between them, and structural metadata — all packed within your token budget. The agent gets everything it needs to implement, not just the file it asked for.

**For developers:** A single `analyze_repo("./my-project")` call returns a full architectural map — module boundaries, detected design patterns, public API surface, inter-module dependencies, and inferred trade-offs. No need to read thousands of lines to understand how a codebase is organized.

**For teams:** A single `compare_repos("./project-a", "./project-b", dimensions=["error_handling", "api_surface"])` call produces a structured comparison showing how two codebases approach the same architectural concerns differently.

---

## Token Efficiency: Measured, Not Claimed

Every archex tool response includes a `_meta` block reporting exactly how many tokens were saved compared to raw file access.

### Symbol Lookup

When your agent needs a specific function:

```text
Task: Retrieve the ConnectionPool class implementation

Without archex:  Open _pool.py                → 3,200 tokens
With archex:     get_symbol("...::Pool#class") →   340 tokens

Savings: 89%
```

### File Understanding

When your agent needs to understand a file's structure:

```text
Task: Understand what src/auth.py contains

Without archex:  Read entire file        → 4,800 tokens
With archex:     get_file_outline(file)  →   180 tokens (signatures + hierarchy only)

Savings: 96%
```

### Subsystem Understanding

When your agent needs to understand how a feature works end-to-end:

```text
Task: Understand how the auth middleware validates tokens, loads sessions,
      and checks permissions — including all type definitions and configs

Without archex:  Agent reads 8 files across 3 directories → 38,000 tokens
                 Agent misses the config dependency        → +1 backtrack read
                 Agent misses a type definition             → +1 more read
                 Total: ~45,000 tokens, 10 file reads, incomplete understanding

With archex:     query_repo("How does auth work?", budget=8000) → 7,600 tokens
                 Includes: auth middleware, session store, user model,
                 permission checker, config, 3 type definitions
                 Total: 7,600 tokens, 1 call, complete context

Savings: 83%
```

The 83% number is real, not theoretical. It accounts for the structural metadata archex includes (file map, module context, type definitions, dependency summary) that makes the agent's response more accurate on the first try.

### Repository Navigation

When your agent needs to find where to start:

```text
Task: Understand the structure of a 500-file repository

Without archex:  Read directory listing + skim files → 200,000+ tokens
With archex:     get_file_tree(repo)                 →   2,000 tokens

Savings: 99%
```

---

## What Makes archex Different

### Dependency-Aware Retrieval

Most code search tools find code that _matches your query_. archex finds code that _answers your question_ — including structurally related code you didn't know to ask for.

When you query "How does connection pooling work?", archex doesn't just return functions with "pool" in the name. It traces the dependency graph and also returns:

- The transport layer that _uses_ the pool
- The config class that _bounds_ the pool
- The type definitions that pool methods _reference_
- The health check that _monitors_ the pool

This dependency expansion is why agents using archex get correct implementations on the first attempt — they have the complete structural context, not isolated code fragments.

### Token Budget Management

Context windows are finite and expensive. archex doesn't just return "relevant files" — it assembles a precisely-sized context block:

1. **Retrieve** candidate code chunks via BM25 + optional vector search
2. **Expand** along the dependency graph to include structurally related code
3. **Rank** by relevance, structural importance (PageRank centrality), and type coverage
4. **Pack** greedily into your specified token budget, resolving type definitions and deduplicating overlapping content
5. **Attach** structural metadata: file map, module context, call chain, dependency summary

The result is a self-contained context block where every token earns its place.

### Architectural Intelligence

archex doesn't just index code — it understands code structure:

- **Module Detection:** Louvain community detection on the dependency graph finds natural module boundaries, even when the file system doesn't reflect them.
- **Pattern Recognition:** Rule-based detection identifies middleware chains, plugin systems, event buses, repository patterns, and strategy patterns — with confidence scores and file-level evidence.
- **Interface Extraction:** Identifies the public API surface with usage counts, parameter types, and documentation.
- **Trade-off Inference:** Structural evidence reveals design decisions (e.g., "chose sync over async," "centralized config over distributed") with optional LLM enrichment for deeper analysis.

### LLM-Optional

The entire structural pipeline — AST parsing, symbol extraction, dependency graphs, module detection, pattern recognition, chunking, BM25 indexing, token budget assembly — runs without any LLM calls.

This means:

- **Deterministic:** Same repo + same config = same output, always
- **Fast:** No API latency in the critical path
- **Free:** No token cost for structural analysis
- **Testable:** Every pipeline stage produces inspectable intermediate output

LLM enrichment is available for deeper analysis (module descriptions, trade-off rationale) but is always opt-in. You control when and whether to spend LLM tokens.

### Framework-Agnostic

archex is a Python library first, an MCP server second. It works with any agent framework:

```python
# Direct Python usage
from archex import query
bundle = query(source, "How does auth work?", token_budget=8000)

# MCP server for Claude Code / Claude Desktop
archex mcp

# LangChain retriever
from archex.integrations.langchain import ArchexRetriever

# LlamaIndex query engine
from archex.integrations.llamaindex import ArchexRetriever

# Any agent that can shell out
$ archex query ./repo "connection pooling" --budget 8000 --format xml
```

No framework lock-in. No heavyweight dependencies. Install what you need.

---

## Concrete Use Cases

### Use Case 1: Agent-Driven Feature Implementation

Your coding agent needs to add rate limiting to a service. It needs to understand the existing middleware chain, the config system, and the error handling conventions.

```python
# One call gives the agent everything it needs
bundle = query(
    RepoSource(local_path="./api-service"),
    "How do middleware, configuration, and error handling work together?",
    token_budget=12000,
)
# Agent receives: middleware chain code, config loader, error types,
# middleware registration pattern, relevant type definitions
# → Implements rate limiting consistent with existing patterns on first attempt
```

**Without archex:** The agent explores 10-15 files across multiple tool calls, misses the config convention, implements rate limiting inconsistently with existing middleware, and requires a human review cycle to fix the patterns.

### Use Case 2: Architecture Review Before Integration

You're evaluating whether to use Library A or Library B for your project. You need to understand their error handling, async patterns, and public API design — fast.

```python
result = compare(
    RepoSource(url="https://github.com/org/library-a"),
    RepoSource(url="https://github.com/org/library-b"),
    dimensions=["error_handling", "api_surface", "concurrency"],
)
# Structured comparison with evidence from both codebases
```

**Without archex:** You clone both repos, spend 2-3 hours reading source code, form opinions based on incomplete reading, miss important patterns buried deep in the codebase.

### Use Case 3: Onboarding to an Unfamiliar Codebase

New team member (or new agent) needs to understand a 500-file monorepo quickly.

```python
# Step 1: Get the lay of the land (< 2,000 tokens)
profile = analyze(RepoSource(local_path="./company-monorepo"))
# → 8 modules detected, 12 patterns, dependency graph, public API surface

# Step 2: Drill into a specific area (< 8,000 tokens)
bundle = query(
    RepoSource(local_path="./company-monorepo"),
    "How does the payment processing pipeline work?",
    token_budget=8000,
)
# → Complete payment flow: entry point → validation → processor → ledger → notifications
```

**Without archex:** The developer spends a full day reading code. The agent spends 200K+ tokens scanning files with diminishing returns.

### Use Case 4: Multi-Agent Workflows

In a SwarmLens-style multi-agent system, different agents handle different aspects of a task. The architecture agent needs to understand the target codebase; the implementation agent needs precise code context.

```python
# Architecture agent gets the high-level view
profile = analyze(source)
architecture_context = profile.to_markdown()
# → "The auth system uses middleware chain + session store + RBAC checker"

# Implementation agent gets the exact code it needs
bundle = query(source, "Implement a new permission check in the RBAC system", token_budget=8000)
implementation_context = bundle.to_prompt(format="xml")
# → Exact code, types, imports, and structural context for the implementation
```

Each agent consumes only the tokens relevant to its task. No duplication, no waste.

### Use Case 5: CI/CD Architectural Drift Detection

Run archex in your CI pipeline to detect when architectural patterns drift:

```bash
# In CI: compare current branch against main
archex compare ./main-checkout ./pr-checkout \
    --dimensions api_surface,error_handling,configuration \
    --format json
```

Detect when a PR introduces inconsistent error handling, breaks the public API convention, or adds a new module that doesn't follow the existing patterns.

---

## Language Support

archex parses code structurally using tree-sitter — no regex, no heuristics, no language-server dependencies.

| Language                    | Extensions                   | Symbols Extracted                                                            | Import Resolution                                   |
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

---

## Performance Characteristics

| Repo Size            | Files | First Index | Cached Query | Cache Size |
| -------------------- | ----- | ----------- | ------------ | ---------- |
| Small (click)        | ~50   | < 2s        | < 100ms      | ~1MB       |
| Medium (httpx)       | ~200  | 3-5s        | < 200ms      | ~3MB       |
| Large (FastAPI)      | ~500  | 8-15s       | < 500ms      | ~8MB       |
| Very Large (Next.js) | ~5000 | 40-90s      | < 2s         | ~50MB      |

- **Index once, query forever.** The first call indexes the repo. Subsequent queries hit the SQLite cache and skip parsing entirely.
- **Git-aware cache invalidation.** Cache keys include the HEAD commit hash. When you pull new changes, the next query triggers incremental re-indexing.
- **Parallel parsing.** Multi-core file parsing for faster indexing on large repos.
- **Lazy vector loading.** Embedding vectors are built on first hybrid query, not on import.

---

## Installation

```bash
# Core (BM25 retrieval, no embeddings)
uv add archex

# With local vector embeddings (Nomic Code via ONNX, ~50MB)
uv add "archex[vector]"

# With MCP server for Claude Code / Claude Desktop
uv add "archex[mcp]"

# Everything
uv add "archex[all]"
```

### System Requirements

- Python 3.11+
- Git on PATH (for remote repo cloning)
- ~30MB disk for core install
- ~50MB additional for vector embeddings

---

## Quick Start

### 30 Seconds: CLI

```bash
# Analyze a repo
archex analyze ./my-project --format markdown

# Query for implementation context
archex query ./my-project "How does caching work?" --budget 8000

# Compare two repos
archex compare ./project-a ./project-b --dimensions error_handling,api_surface
```

### 60 Seconds: Python API

```python
from archex import analyze, query
from archex.models import RepoSource

source = RepoSource(local_path="./my-project")

# Get the architectural map
profile = analyze(source)
print(f"{len(profile.module_map)} modules, {len(profile.pattern_catalog)} patterns detected")

# Get implementation context for your agent
bundle = query(source, "How does the payment flow work?", token_budget=8000)
print(bundle.to_prompt(format="xml"))
# → Self-contained XML context block ready to paste into any LLM prompt
```

### 90 Seconds: MCP Server

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

Your agent now has access to `analyze_repo`, `query_repo`, `compare_repos`, `get_file_tree`, `get_file_outline`, `search_symbols`, `get_symbol`, and `get_symbols_batch`.

---

## Architecture at a Glance

```text
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Acquire  │ → │  Parse   │ → │  Index   │ → │ Analyze  │ → │  Serve   │
│          │   │          │   │          │   │          │   │          │
│ git clone│   │tree-sitter│   │ BM25     │   │ Louvain  │   │ArchProfile│
│ local    │   │ AST walk │   │ Vector   │   │ Patterns │   │ Context  │
│ discover │   │ 8 langs  │   │ Dep Graph│   │ Interfaces│   │ Compare  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                    │
                              ┌─────┴─────┐
                              │  SQLite   │
                              │  Cache    │
                              │  (FTS5)   │
                              └───────────┘
```

- **Zero external services.** Everything runs locally. SQLite for storage, NetworkX for graph algorithms, tree-sitter for parsing.
- **LLM-optional.** The entire pipeline works without API keys. LLM enrichment is opt-in.
- **Framework-agnostic.** No coupling to LangChain, LlamaIndex, or any specific agent framework.

---

## Who Benefits

| Role                         | How archex Helps                                                                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **AI agent developers**      | Give your agents precise codebase context without burning through context windows. Single-call retrieval replaces multi-file exploration loops. |
| **Solo developers**          | Understand unfamiliar codebases in minutes instead of hours. Get architectural maps and pattern detection for free.                             |
| **Team leads**               | Compare how different projects solve the same problems. Detect architectural drift in CI.                                                       |
| **Open-source contributors** | Understand a project's architecture before submitting a PR. Find the right place to make changes.                                               |
| **Consultants**              | Rapidly assess client codebases. Produce architecture reports without weeks of code reading.                                                    |

---

## Design Principles

1. **Retrieval-first.** The primary value is getting the right code into the right context window at the right cost.
2. **No magic.** Every operation is explicit, composable, and inspectable. No hidden LLM calls, no implicit state.
3. **Index once, query forever.** Amortize the parsing cost. Subsequent queries are fast and free.
4. **Token-aware everywhere.** Every output is measured in tokens. Every tool reports its efficiency. Budget management is first-class.
5. **Structural over textual.** Dependency-aware retrieval beats keyword matching. Module detection beats directory listing. Pattern recognition beats code reading.

---

## Open Source

archex is Apache 2.0 licensed. Contributions welcome.

```bash
git clone https://github.com/determ-ai/archex.git
cd archex
uv sync --all-extras
uv run pytest  # 1274 tests, 92% coverage
```

Extensible via entry points: add language adapters and pattern detectors without modifying core.

---

_Index once. Query smart. Ship faster._
