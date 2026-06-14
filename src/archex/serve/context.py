"""ContextBundle assembly: retrieve, rank, and assemble chunks into a ContextBundle."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.models import (
    CodeChunk,
    ContextBundle,
    Module,
    RankedChunk,
    RetrievalMetadata,
    ScoringWeights,
    StructuralContext,
    SymbolKind,
    TypeDefinition,
)
from archex.observe import PipelineTrace, StepTiming

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph


@dataclass(frozen=True)
class _Hop2Expansion:
    graph: DependencyGraph
    candidate_map: dict[str, CodeChunk]
    chunks_by_file: dict[str, list[CodeChunk]]
    hop1_files_added: list[str]
    seed_files: set[str]
    expansion_priority: dict[str, float]
    query_terms: set[str]
    remaining_budget: int
    max_per_file: int


logger = logging.getLogger(__name__)

_TYPE_LIKE = {SymbolKind.CLASS, SymbolKind.TYPE, SymbolKind.INTERFACE}

# Propagated relevance for imported files that were already admitted as expansion candidates.
NEIGHBOR_IMPORT_TARGET_DECAY = 0.65

# Propagated relevance for importer files that were already admitted as expansion candidates.
NEIGHBOR_IMPORTER_DECAY = 0.35

MIN_SCORE_RATIO = 0.30
MIN_BUDGET_FILL_RATIO = 0.50


MAX_EXPANSION_FILES = 8

MAX_FILES = 8

# Files reached by ≥2 independent seeds receive a convergence bonus —
# multiple structural paths to a file corroborate its relevance.
CONVERGENCE_BONUS = 1.5

# Entry-point files define module interfaces but have low keyword density.
# BM25 systematically under-ranks them because they re-export rather than define.
_ENTRY_POINT_BOOST = 1.3
_ENTRY_POINT_NAMES = frozenset(
    {
        "__init__.py",
        "mod.rs",
        "lib.rs",
        "index.js",
        "index.ts",
        "index.jsx",
        "index.tsx",
        "index.mjs",
        "__init__.pyi",
    }
)

# Seeds below this fraction of max normalized seed score do not trigger expansion.
# Lowered from 0.10 — large repos have flatter BM25 distributions where
# 10% excluded valid seeds, leaving graph expansion permanently inert.
SEED_EXPANSION_MIN = 0.05

# Files below this fraction of the top file's aggregate score are excluded.
FILE_SCORE_CUTOFF = 0.10
FUSION_FILE_SCORE_CUTOFF = 0.05


def estimate_tokens(chunk: CodeChunk) -> int:
    if chunk.token_count > 0:
        return chunk.token_count
    return int(len(chunk.content.split()) * 1.3)


CLI_MAX_CHUNKS_PER_FILE = 2


def _file_score_cutoff_ratio(*, fusion_applied: bool) -> float:
    return FUSION_FILE_SCORE_CUTOFF if fusion_applied else FILE_SCORE_CUTOFF


def _is_test_file(file_path: str) -> bool:
    return file_path.startswith("test") or "/test" in file_path


def _is_support_file(file_path: str, query_terms: set[str]) -> bool:
    lower_path = file_path.lower()
    if _is_test_file(lower_path):
        return not query_terms & {"fixture", "fixtures", "test", "tests"}
    if any(marker in lower_path for marker in _SUPPORT_PATH_MARKERS):
        return not query_terms & _SUPPORT_QUERY_TERMS
    return False


def _type_definitions(chunks: list[RankedChunk]) -> list[TypeDefinition]:
    return [
        TypeDefinition(
            symbol=ranked.chunk.symbol_name or ranked.chunk.id,
            file_path=ranked.chunk.file_path,
            start_line=ranked.chunk.start_line,
            end_line=ranked.chunk.end_line,
            content=ranked.chunk.content,
        )
        for ranked in chunks
        if ranked.chunk.symbol_kind in _TYPE_LIKE
    ]


def passthrough_context(
    all_chunks: list[CodeChunk],
    question: str,
    token_budget: int,
) -> ContextBundle:
    """Return all chunks directly when total tokens fit within budget.

    Skips BM25/scoring overhead for small repos where retrieval adds no value.
    """
    assembly_start = time.perf_counter()
    total_tokens = sum(estimate_tokens(c) for c in all_chunks)
    included = [
        RankedChunk(chunk=chunk, relevance_score=1.0, final_score=1.0) for chunk in all_chunks
    ]
    included_files = sorted({c.file_path for c in all_chunks})
    file_tree = "\n".join(included_files)
    structural_context = StructuralContext(file_tree=file_tree)

    type_defs = _type_definitions(included)

    assembly_ms = (time.perf_counter() - assembly_start) * 1000

    meta = RetrievalMetadata(
        candidates_found=len(all_chunks),
        candidates_after_expansion=len(all_chunks),
        chunks_included=len(included),
        chunks_dropped=0,
        strategy="passthrough",
        assembly_time_ms=assembly_ms,
    )

    return ContextBundle(
        query=question,
        chunks=included,
        structural_context=structural_context,
        type_definitions=type_defs,
        token_count=total_tokens,
        token_budget=token_budget,
        truncated=False,
        retrieval_metadata=meta,
    )


_QUERY_STOP = frozenset(
    {
        "archex",
        "how",
        "does",
        "implement",
        "what",
        "handle",
        "manage",
        "function",
        "method",
        "class",
        "module",
        "file",
        "code",
        "work",
        "used",
        "using",
        "create",
        "make",
        "define",
        "call",
        "return",
        "type",
        "data",
        "value",
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "show",
        "find",
    }
)

# Architecture-intent synonyms: map broad architectural keywords to code-level
# equivalents. These keywords still control architecture-specific expansion.
_ARCH_SYNONYMS: dict[str, list[str]] = {
    "pipeline": ["workflow", "chain", "process", "pipe", "stage", "assembly", "context"],
    "middleware": ["handler", "interceptor", "filter", "hook", "router", "route", "layer", "stack"],
    "registry": ["register", "catalog", "factory", "provider"],
    "adapter": ["plugin", "connector", "driver", "bridge"],
    "injection": [
        "inject",
        "resolve",
        "depend",
        "depends",
        "dependant",
        "wire",
        "provider",
        "resolver",
    ],
    "routing": ["route", "router", "dispatch", "endpoint", "path", "handler", "register", "mount"],
    "index": ["indexing", "indexed", "cache", "config", "project", "store", "build"],
    "indexing": ["index", "indexed", "delta", "catalog", "cache", "store"],
    "dependency": [
        "depend",
        "depends",
        "dependant",
        "resolve",
        "inject",
        "require",
        "provider",
        "resolver",
    ],
    "session": ["connection", "pool", "client", "transport"],
    "hook": ["callback", "listener", "subscriber", "event", "state", "effect"],
    "orm": [
        "model",
        "schema",
        "mapper",
        "table",
        "entity",
        "queryset",
        "sql",
        "compiler",
        "expression",
        "where",
    ],
    "task": ["job", "worker", "celery", "dispatch", "execute"],
    "runtime": ["scheduler", "executor", "loop", "spawn"],
}

# Framework-semantic synonyms improve lexical and path alignment without making
# every framework implementation question trigger architecture-only expansion.
_FRAMEWORK_SYNONYMS: dict[str, list[str]] = {
    "validator": [
        "validate",
        "validation",
        "field_validator",
        "model_validator",
        "functional_validators",
        "validate_call",
    ],
    "validators": [
        "validate",
        "validation",
        "field_validator",
        "model_validator",
        "functional_validators",
        "validate_call",
    ],
    "decorator": [
        "decorators",
        "decoration",
        "parameter",
        "option",
        "argument",
        "command",
        "callback",
        "wrapper",
    ],
    "decorators": [
        "decorator",
        "decoration",
        "parameter",
        "option",
        "argument",
        "command",
        "callback",
        "wrapper",
    ],
    "parameter": ["param", "option", "argument", "decorator"],
    "parameters": ["param", "option", "argument", "decorator"],
    "route": ["router", "routing", "endpoint", "handler", "register", "mount"],
    "routes": ["router", "routing", "endpoint", "handler", "register", "mount"],
}

_CLI_LIFECYCLE_COMMAND_TERMS = frozenset({"cache", "config", "index", "init", "reset", "status"})
_SUPPORT_QUERY_TERMS = frozenset(
    {
        "benchmark",
        "benchmarks",
        "compare",
        "comparison",
        "dogfood",
        "evidence",
        "fixture",
        "fixtures",
        "readiness",
        "report",
        "reports",
        "test",
        "tests",
        "triage",
    }
)
_SUPPORT_PATH_MARKERS = frozenset(("/benchmark/", "/serve/compare/"))


# Architecture keywords that trigger 2-hop expansion.
_ARCH_KEYWORDS = frozenset(_ARCH_SYNONYMS.keys())


def _split_compound_token(token: str) -> list[str]:
    """Split a camelCase or snake_case token into component words.

    Pass the original mixed-case token; the caller lowercases all returned
    parts before use.  Returns the original token plus split components.
    Examples:
      "queryPipeline"  → ["queryPipeline", "query", "Pipeline"]
      "next_function"  → ["next_function", "next", "function"]
      "BM25Index"      → ["BM25Index", "BM25", "index"]
    """
    import re

    # Split snake_case by underscore
    if "_" in token:
        parts = [p for p in token.split("_") if p]
        return [token] + parts if len(parts) > 1 else [token]

    # Split camelCase / PascalCase by uppercase boundaries
    camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)", token)
    if len(camel_parts) > 1:
        return [token] + [p.lower() for p in camel_parts]

    return [token]


def _singular_query_variants(term: str) -> set[str]:
    if len(term) <= 3:
        return set()
    if term.endswith("ies") and len(term) > 4:
        return {term[:-3] + "y"}
    if term.endswith("s"):
        return {term[:-1]}
    return set()


def _query_terms(question: str) -> set[str]:
    """Extract lowercased content words from a query for expansion prioritization.

    Enhancements over the basic word-extraction:
    - Splits camelCase and snake_case tokens into components so path matching
      works against both joined and split forms.
    - Keeps compound phrase forms (e.g. "dependency_injection") alongside
      individual words so expansion scoring can match on both.
    - Expands architectural-intent keywords to code-level synonyms to close
      vocabulary gaps between natural-language queries and source identifiers.
    """
    import re

    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", question)

    expanded: set[str] = set()
    normalized_tokens: list[str] = []
    for tok in raw_tokens:
        low = tok.lower()
        if low in _QUERY_STOP:
            continue
        # Pass the original (mixed-case) token so camelCase boundaries are visible,
        # then lowercase all returned parts before adding to the term set.
        parts = _split_compound_token(tok)
        for p in parts:
            p_low = p.lower()
            if p_low not in _QUERY_STOP and len(p_low) >= 3:
                expanded.add(p_low)
                normalized_tokens.append(p_low)

    # Add bigram compound forms for adjacent non-stop pairs (e.g. dependency + injection
    # → "dependency_injection") so they match identifiers that use this combined form.
    clean = [t for t in normalized_tokens if t not in _QUERY_STOP and len(t) >= 3]
    for term in list(expanded):
        expanded.update(_singular_query_variants(term))
    for i in range(len(clean) - 1):
        compound = f"{clean[i]}_{clean[i + 1]}"
        expanded.add(compound)

    # Phrase-specific expansions keep product vocabulary aligned without making
    # every generic "query" question look like BM25 internals.
    question_lower = question.lower()
    if "query pipeline" in question_lower:
        expanded.update(
            {"api", "search", "retrieve", "retrieval", "lookup", "bm25", "rank", "score"}
        )
    if "initialize" in expanded or "initialise" in expanded:
        expanded.update({"init", "cli", "main", "project", "config"})
    if {"project", "state"} <= expanded:
        expanded.update({"cli", "project", "config"})
    state_lifecycle_query = bool({"fresh", "stale", "dirty", "corrupt"} & expanded)
    if state_lifecycle_query:
        expanded.update({"status", "fresh", "stale", "dirty", "corrupt", "delta", "project"})
    if {"build", "refresh"} & expanded and "index" in expanded:
        expanded.update({"index", "cli", "api", "cache", "project", "config"})
    if {"settings", "configuration"} & expanded or "runtime_configuration" in expanded:
        expanded.update({"config", "settings", "runtime", "cache", "project"})
    if "mcp" in expanded:
        expanded.update({"api", "context", "mcp_cmd", "model", "models"})
    if "query" in expanded and "cache" in expanded:
        expanded.update({"api", "config", "query_cmd"})
    if "reset" in expanded and "project" in expanded:
        expanded.update({"cli", "main"})
    if {"benchmark", "dogfood", "gate"} <= expanded:
        expanded.update({"baseline", "benchmark_cmd", "report", "reporter"})
    if "middleware" in expanded:
        expanded.update({"common", "wsgi", "asgi"})
    if "pooling" in expanded or "keep_alive" in expanded:
        expanded.update({"client", "config"})
    if {"dispatch", "execute"} & expanded and {"task", "tasks"} & expanded:
        expanded.update({"amqp", "broker", "message", "queue", "strategy", "worker"})
    if {"session", "sessions"} & expanded or "connection_pooling" in expanded:
        expanded.update({"adapter", "adapters", "model", "models", "request", "response"})
    if "orm" in expanded and "sql" in expanded:
        expanded.update({"query", "queries", "compiler", "where", "expression", "expressions"})
        expanded.update({"model", "models"})

    # Semantic synonym expansion
    for term in list(expanded):
        if term in _ARCH_SYNONYMS:
            expanded.update(_ARCH_SYNONYMS[term])
        if term in _FRAMEWORK_SYNONYMS:
            expanded.update(_FRAMEWORK_SYNONYMS[term])

    if state_lifecycle_query:
        expanded.difference_update({"build", "cache", "config", "store"})

    return expanded


def _adaptive_max_files(
    file_scores: list[tuple[str, float]],
    default: int = MAX_FILES,
) -> int:
    """Reduce MAX_FILES when BM25 has clear score separation.

    When the top file score is >3x the median score, reduce to 4.
    When >2x, reduce to 5.
    Otherwise keep the default of 6.
    """
    if len(file_scores) < 3:
        return min(default, len(file_scores))
    scores = [s for _, s in file_scores]
    top = scores[0]
    median = scores[len(scores) // 2]
    if median <= 0:
        return default
    ratio = top / median
    if ratio > 3.0:
        return 5
    if ratio > 2.0:
        return 6
    return default


def _is_architecture_query(question: str) -> bool:
    """Return True if the query contains any architecture keyword."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _ARCH_KEYWORDS)


def _is_entry_point(file_path: str) -> bool:
    """Return True if the file is a module entry point (e.g. __init__.py, mod.rs)."""
    basename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    return basename in _ENTRY_POINT_NAMES


def _path_alignment_boost(file_path: str, query_terms: set[str]) -> float:
    """Return a multiplier >1.0 when a file path matches query terms."""
    lower_path = file_path.lower()
    parts = lower_path.rsplit("/", 1)
    dir_path = parts[0] if len(parts) == 2 else ""
    basename = parts[-1]
    stem = basename.rsplit(".", 1)[0]
    normalized_stem = stem.lower().lstrip("_")
    path_terms = {segment for segment in dir_path.replace("-", "_").split("/") if len(segment) >= 3}
    path_terms.update(
        part for token in stem.replace("-", "_").split("_") for part in _split_compound_token(token)
    )
    path_terms = {term.lower() for term in path_terms if len(term) >= 3}
    if stem.lower() in query_terms or normalized_stem in query_terms:
        return 3.0
    matched_terms = path_terms & query_terms
    if not matched_terms:
        return 1.0
    if "cli" in path_terms and matched_terms & _CLI_LIFECYCLE_COMMAND_TERMS:
        return 2.2
    if "cli" in path_terms and "cli" in query_terms:
        return 1.6
    if matched_terms & _CLI_LIFECYCLE_COMMAND_TERMS:
        return 1.6
    return 1.35


def _type_alignment_score(
    chunk: CodeChunk,
    query_terms: set[str],
    *,
    definition_lookup: bool,
) -> float:
    if chunk.symbol_kind not in _TYPE_LIKE:
        return 0.0
    if definition_lookup:
        return 0.5
    symbol_parts: set[str] = set()
    if chunk.symbol_name:
        symbol_parts.update(part.lower() for part in _split_compound_token(chunk.symbol_name))
    path_lower = chunk.file_path.lower()
    if symbol_parts & query_terms or any(term in path_lower for term in query_terms):
        return 0.5
    return 0.0


def _aggregate_file_scores(ranked: list[RankedChunk]) -> dict[str, float]:
    per_file: dict[str, list[float]] = {}
    for rc in ranked:
        per_file.setdefault(rc.chunk.file_path, []).append(rc.final_score)

    aggregated: dict[str, float] = {}
    for file_path, scores in per_file.items():
        total = 0.0
        weight = 1.0
        for score in sorted(scores, reverse=True):
            total += score * weight
            weight *= 0.5
        aggregated[file_path] = total
    return aggregated


def _nested_included_range(
    chunk: CodeChunk,
    included_ranges: dict[str, list[tuple[int, int]]],
) -> bool:
    ranges = included_ranges.get(chunk.file_path)
    if not ranges:
        return False
    current = (chunk.start_line, chunk.end_line)
    for start, end in ranges:
        if current == (start, end):
            continue
        if start <= chunk.start_line and chunk.end_line <= end:
            return True
        if chunk.start_line <= start and end <= chunk.end_line:
            return True
    return False


def _try_include_ranked_chunk(
    rc: RankedChunk,
    included: list[RankedChunk],
    included_ids: set[str],
    included_ranges: dict[str, list[tuple[int, int]]],
    included_file_counts: dict[str, int],
    total_tokens: int,
    token_budget: int,
    max_chunks_per_file: int | None,
) -> int:
    if (
        rc.chunk.id in included_ids
        or _nested_included_range(rc.chunk, included_ranges)
        or (
            max_chunks_per_file is not None
            and included_file_counts.get(rc.chunk.file_path, 0) >= max_chunks_per_file
        )
    ):
        return total_tokens
    tokens = estimate_tokens(rc.chunk)
    if total_tokens + tokens > token_budget:
        return total_tokens
    included.append(rc)
    included_ids.add(rc.chunk.id)
    included_ranges.setdefault(rc.chunk.file_path, []).append(
        (rc.chunk.start_line, rc.chunk.end_line)
    )
    included_file_counts[rc.chunk.file_path] = included_file_counts.get(rc.chunk.file_path, 0) + 1
    return total_tokens + tokens


def _pack_ranked_chunks(
    ranked: list[RankedChunk],
    sorted_files: list[tuple[str, float]],
    top_files: set[str],
    token_budget: int,
    max_chunks_per_file: int | None = None,
) -> tuple[list[RankedChunk], int]:
    included: list[RankedChunk] = []
    included_ids: set[str] = set()
    included_ranges: dict[str, list[tuple[int, int]]] = {}
    included_file_counts: dict[str, int] = {}
    total_tokens = 0

    best_by_file: dict[str, RankedChunk] = {}
    for rc in ranked:
        best_by_file.setdefault(rc.chunk.file_path, rc)
    ordered_files = [
        file_path
        for file_path, _score in sorted_files
        if file_path in top_files and file_path in best_by_file
    ]
    ordered_files.sort(
        key=lambda file_path: (
            _is_test_file(file_path),
            -best_by_file[file_path].final_score,
        )
    )
    for file_path in ordered_files:
        total_tokens = _try_include_ranked_chunk(
            best_by_file[file_path],
            included,
            included_ids,
            included_ranges,
            included_file_counts,
            total_tokens,
            token_budget,
            max_chunks_per_file,
        )

    score_floor = ranked[0].final_score * MIN_SCORE_RATIO if ranked else 0.0
    min_fill_tokens = int(token_budget * MIN_BUDGET_FILL_RATIO)
    for rc in ranked:
        if rc.final_score < score_floor and total_tokens >= min_fill_tokens:
            break
        total_tokens = _try_include_ranked_chunk(
            rc,
            included,
            included_ids,
            included_ranges,
            included_file_counts,
            total_tokens,
            token_budget,
            max_chunks_per_file,
        )

    return included, total_tokens


def _normalized_scores(results: list[tuple[CodeChunk, float]]) -> dict[str, float]:
    max_score = max((score for _, score in results), default=1.0) or 1.0
    return {chunk.id: score / max_score for chunk, score in results}


def _seed_file_scores(
    search_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]] | None,
    splade_results: list[tuple[CodeChunk, float]] | None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for chunk, score in search_results:
        effective = score * (0.6 if _is_test_file(chunk.file_path) else 1.0)
        scores[chunk.file_path] = max(scores.get(chunk.file_path, 0.0), effective)
    if vector_results:
        for chunk, score in vector_results:
            if chunk.file_path in scores:
                continue
            effective = score * (0.6 if _is_test_file(chunk.file_path) else 1.0)
            scores[chunk.file_path] = effective
    if splade_results:
        for chunk, score in splade_results:
            effective = score * (0.6 if _is_test_file(chunk.file_path) else 1.0)
            scores[chunk.file_path] = max(scores.get(chunk.file_path, 0.0), effective)
    return scores


def _chunks_by_file(all_chunks: list[CodeChunk]) -> dict[str, list[CodeChunk]]:
    chunks_by_path: dict[str, list[CodeChunk]] = {}
    for chunk in all_chunks:
        chunks_by_path.setdefault(chunk.file_path, []).append(chunk)
    return chunks_by_path


def _add_file_chunks(
    candidate_map: dict[str, CodeChunk],
    chunks_by_file: dict[str, list[CodeChunk]],
    file_path: str,
    *,
    max_per_file: int,
) -> int:
    added = 0
    for chunk in chunks_by_file.get(file_path, []):
        if chunk.id in candidate_map:
            continue
        candidate_map[chunk.id] = chunk
        added += 1
        if added >= max_per_file:
            break
    return added


def _initial_candidate_map(
    search_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]] | None,
    splade_results: list[tuple[CodeChunk, float]] | None,
) -> dict[str, CodeChunk]:
    candidate_map = {chunk.id: chunk for chunk, _ in search_results}
    if vector_results:
        for chunk, _ in vector_results:
            candidate_map.setdefault(chunk.id, chunk)
    if splade_results:
        for chunk, _ in splade_results:
            candidate_map.setdefault(chunk.id, chunk)
    return candidate_map


def _normalized_rerank_scores(reranked: list[tuple[CodeChunk, float]]) -> dict[str, float]:
    if not reranked:
        return {}

    scores = [score for _, score in reranked]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {chunk.id: 1.0 for chunk, _ in reranked}

    span = max_score - min_score
    return {chunk.id: (score - min_score) / span for chunk, score in reranked}


def _unique_file_paths(results: list[tuple[CodeChunk, float]]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for chunk, _ in results:
        if chunk.file_path in seen:
            continue
        seen.add(chunk.file_path)
        paths.append(chunk.file_path)
    return paths


def _record_expansion_reason(
    reasons_by_file: dict[str, set[str]],
    reason_counts: dict[str, int],
    file_path: str,
    reason: str,
) -> None:
    file_reasons = reasons_by_file.setdefault(file_path, set())
    if reason in file_reasons:
        return
    file_reasons.add(reason)
    reason_counts[reason] = reason_counts.get(reason, 0) + 1


def _is_low_signal_expansion_candidate(
    file_path: str,
    reasons: set[str],
    source_count: int,
    query_terms: set[str],
) -> bool:
    if "hub" not in reasons or source_count > 1:
        return False
    if reasons & {"same_module", "entry_point"}:
        return False
    return not any(term in file_path.lower() for term in query_terms)


def _hop2_expansion_priority(expansion: _Hop2Expansion) -> dict[str, float]:
    hop1_files = set(expansion.hop1_files_added)
    hop2_priority: dict[str, float] = {}
    for hop1_fp in expansion.hop1_files_added:
        hop1_score = expansion.expansion_priority.get(hop1_fp, 0.0)
        for dep in expansion.graph.imports_of(hop1_fp):
            if dep in expansion.seed_files or dep in hop1_files:
                continue
            path_match = any(term in dep.lower() for term in expansion.query_terms)
            priority = hop1_score * (1.5 if path_match else 1.0)
            hop2_priority[dep] = max(hop2_priority.get(dep, 0.0), priority)
    return hop2_priority


def _add_hop2_expansion(expansion: _Hop2Expansion) -> tuple[list[str], int]:
    hop2_priority = _hop2_expansion_priority(expansion)
    hop2_files_added: list[str] = []
    test_candidates_skipped = 0
    for file_path in sorted(hop2_priority.keys(), key=lambda f: -hop2_priority[f]):
        if len(hop2_files_added) >= expansion.remaining_budget:
            break
        if _is_test_file(file_path):
            test_candidates_skipped += 1
            continue
        added = _add_file_chunks(
            expansion.candidate_map,
            expansion.chunks_by_file,
            file_path,
            max_per_file=expansion.max_per_file,
        )
        if added > 0:
            hop2_files_added.append(file_path)

    logger.debug(
        "graph_expansion 2-hop: arch_query=True, hop2_candidates=%d, hop2_added=%d",
        len(hop2_priority),
        len(hop2_files_added),
    )
    return hop2_files_added, test_candidates_skipped


def _dependency_subgraph(
    graph: DependencyGraph,
    included_files: list[str],
) -> dict[str, list[str]]:
    included_file_set = set(included_files)
    subgraph: dict[str, list[str]] = {}
    for edge in graph.file_edges():
        if edge.source in included_file_set and edge.target in included_file_set:
            subgraph.setdefault(edge.source, []).append(edge.target)
    return subgraph


def _neighbor_boosts(
    graph: DependencyGraph,
    seed_files: set[str],
    candidate_map: dict[str, CodeChunk],
    norm_seed_scores: dict[str, float],
    effective_expansion_min: float,
    bm25_by_id: dict[str, float],
    query_terms: set[str],
) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for file_path in seed_files:
        if norm_seed_scores.get(file_path, 0.0) < effective_expansion_min:
            continue
        seed_score = max(
            (
                bm25_by_id.get(chunk.id, 0.0)
                for chunk in candidate_map.values()
                if chunk.file_path == file_path
            ),
            default=0.0,
        )
        for dep in graph.imports_of(file_path):
            if dep in seed_files:
                continue
            path_match = any(term in dep.lower() for term in query_terms)
            decay = NEIGHBOR_IMPORT_TARGET_DECAY * (1.3 if path_match else 1.0)
            boosts[dep] = max(boosts.get(dep, 0.0), seed_score * decay)
        for importer in graph.imported_by(file_path):
            if importer in seed_files:
                continue
            boosts[importer] = max(
                boosts.get(importer, 0.0),
                seed_score * NEIGHBOR_IMPORTER_DECAY,
            )
    return boosts


def _file_to_module(modules: list[Module] | None) -> dict[str, Module]:
    if not modules:
        return {}
    return {file_path: module for module in modules for file_path in module.files}


def _zero_expansion_reason(
    *,
    seed_files: set[str],
    expansion_eligible_seeds: int,
    total_expansion_candidates: int,
    expansion_files_added: int,
) -> str:
    if not seed_files:
        return "no_seed_files"
    if expansion_eligible_seeds == 0:
        return "no_eligible_seeds"
    if total_expansion_candidates == 0:
        return "no_import_neighbors"
    if expansion_files_added == 0:
        return "candidates_filtered_or_missing_chunks"
    return ""


def assemble_context(
    search_results: list[tuple[CodeChunk, float]],
    graph: DependencyGraph,
    all_chunks: list[CodeChunk],
    question: str,
    token_budget: int = 8192,
    vector_results: list[tuple[CodeChunk, float]] | None = None,
    splade_results: list[tuple[CodeChunk, float]] | None = None,
    scoring_weights: ScoringWeights | None = None,
    modules: list[Module] | None = None,
    trace: PipelineTrace | None = None,
    expansion_min_override: float | None = None,
    avg_idf: float | None = None,
    reranker: object | None = None,
    rerank_candidate_limit: int = 4,
    apply_intent_budget: bool = True,
) -> ContextBundle:
    """Assemble a token-budgeted ContextBundle from search results and a dependency graph.

    When vector_results or splade_results are provided, uses score fusion to merge
    opt-in retrieval legs before scoring.
    When modules is provided, computes cohesion signal per chunk.
    When trace is provided, records step-level timings for graph_expansion, scoring,
    and assembly phases.
    When expansion_min_override is provided, uses it instead of SEED_EXPANSION_MIN
    for expansion gating (useful for architecture-broad queries with flat BM25 scores).
    """
    assembly_start = time.perf_counter()
    # Intent-based weight routing: when no explicit weights are provided,
    # classify the query intent and select optimized weight presets.
    from archex.serve.intent import (
        INTENT_TOKEN_BUDGETS,
        INTENT_WEIGHTS,
        QueryIntent,
        classify_intent,
    )

    intent = classify_intent(question)
    if apply_intent_budget:
        token_budget = min(token_budget, INTENT_TOKEN_BUDGETS[intent])
    weights = INTENT_WEIGHTS[intent] if scoring_weights is None else scoring_weights

    strategy = "hybrid+graph" if vector_results else "bm25+graph"
    if splade_results:
        strategy = "hybrid+splade+graph" if vector_results else "bm25+splade+graph"

    if not search_results and not vector_results and not splade_results:
        return ContextBundle(
            query=question,
            token_budget=token_budget,
            retrieval_metadata=RetrievalMetadata(strategy=strategy),
        )

    # Merge BM25 + vector via confidence-weighted RRF when both are available
    fusion_bm25_weight: float | None = None
    fusion_vector_weight: float | None = None
    fusion_skipped = False
    fusion_skip_reason = ""
    splade_fusion_skipped = False
    splade_fusion_skip_reason = ""
    bm25_cv_val: float | None = None
    effective_vector: list[tuple[CodeChunk, float]] = []
    effective_splade: list[tuple[CodeChunk, float]] = []
    base_results = search_results
    if vector_results:
        from archex.index.fusion import adaptive_rsf, bm25_score_cv, should_fuse

        # Gate fusion: skip when BM25 is confident and signals agree.
        # Always fuse when BM25 is empty — vector is the only signal.
        fuse, fuse_reason = should_fuse(search_results, vector_results, avg_idf=avg_idf)
        bm25_cv_val = bm25_score_cv(search_results)

        if fuse or not search_results:
            # Compute signal_agreement (Jaccard of BM25 top-20 and vector top-20 file paths)
            _k_agree = 20
            _bm25_top_k = {chunk.file_path for chunk, _ in search_results[:_k_agree]}
            _vec_top_k = {chunk.file_path for chunk, _ in vector_results[:_k_agree]}
            _union = _bm25_top_k | _vec_top_k
            signal_agreement_pre: float = (
                len(_bm25_top_k & _vec_top_k) / len(_union) if _union else 0.0
            )

            # RSF preserves score magnitude (unlike RRF which flattens to
            # rank-based 1/(k+rank)). Adaptive weights give vector meaningful
            # influence so unique vector hits can surface.
            merged, fusion_bm25_weight, fusion_vector_weight = adaptive_rsf(
                search_results, vector_results, signal_agreement_pre, bm25_cv_val
            )
            base_results = merged
            bm25_by_id = _normalized_scores(merged)
            effective_vector = vector_results
            logger.debug("Fusion applied (RSF): %s", fuse_reason)
        else:
            # BM25 is confident — skip fusion, use BM25 results only
            signal_agreement_pre = 0.0
            fusion_skipped = True
            fusion_skip_reason = fuse_reason
            strategy = "bm25+graph"  # downgrade strategy label
            bm25_by_id = _normalized_scores(search_results)
            logger.debug("Fusion skipped: %s", fuse_reason)
    else:
        signal_agreement_pre = 0.0
        bm25_by_id = _normalized_scores(search_results)

    if splade_results:
        from archex.index.fusion import adaptive_rsf, bm25_score_cv, should_fuse

        splade_cv = bm25_cv_val if bm25_cv_val is not None else bm25_score_cv(search_results)
        splade_fuse, splade_reason = should_fuse(
            search_results,
            splade_results,
            avg_idf=avg_idf,
        )
        if splade_fuse or not search_results:
            _k_agree = 20
            _bm25_top_k = {chunk.file_path for chunk, _ in search_results[:_k_agree]}
            _splade_top_k = {chunk.file_path for chunk, _ in splade_results[:_k_agree]}
            _union = _bm25_top_k | _splade_top_k
            splade_agreement = len(_bm25_top_k & _splade_top_k) / len(_union) if _union else 0.0
            merged, _, _ = adaptive_rsf(
                base_results,
                splade_results,
                splade_agreement,
                splade_cv,
            )
            base_results = merged
            bm25_by_id = _normalized_scores(merged)
            effective_splade = splade_results
            strategy = "hybrid+splade+graph" if effective_vector else "bm25+splade+graph"
            logger.debug("SPLADE fusion applied (RSF): %s", splade_reason)
        else:
            splade_fusion_skipped = True
            splade_fusion_skip_reason = splade_reason
            if not vector_results:
                strategy = "bm25+graph"
            logger.debug("SPLADE fusion skipped: %s", splade_reason)

    all_results = search_results + effective_vector + effective_splade
    seed_file_paths = _unique_file_paths(all_results)
    seed_files: set[str] = set(seed_file_paths)

    candidates_found = len(search_results)

    # Effective expansion threshold: caller override takes precedence over constant.
    effective_expansion_min = (
        expansion_min_override if expansion_min_override is not None else SEED_EXPANSION_MIN
    )

    # --- Graph expansion phase ---
    _expansion_start = time.perf_counter_ns()

    # Expand: follow directed imports from seed files, prioritized by seed score.
    # imports_of(file) = files this file depends on (high relevance — same call chain)
    # imported_by(file) = files that depend on this file (moderate relevance — consumers)
    seed_file_scores = _seed_file_scores(search_results, effective_vector, effective_splade)

    # Normalize seed file scores to [0, 1] for expansion gating
    max_seed_score = max(seed_file_scores.values()) if seed_file_scores else 1.0
    norm_seed_scores = {fp: s / max_seed_score for fp, s in seed_file_scores.items()}

    # Extract query terms for path-aware import prioritization
    file_to_module = _file_to_module(modules)

    q_terms = _query_terms(question)
    if intent == QueryIntent.CLI:
        cli_terms = set(_CLI_LIFECYCLE_COMMAND_TERMS) | {"main", "project"}
        if "api" in q_terms:
            cli_terms.add("api")
        alignment_terms = q_terms & cli_terms or q_terms
    else:
        alignment_terms = {term for term in q_terms if term not in _ARCH_KEYWORDS} or q_terms

    # Determine whether this is an architecture query (enables 2-hop expansion)
    is_arch_query = _is_architecture_query(question)

    # Expansion diagnostic logging
    qualifying_seeds = [
        fp for fp in seed_files if norm_seed_scores.get(fp, 0.0) >= effective_expansion_min
    ]
    expansion_eligible_seeds = len(qualifying_seeds)
    logger.debug(
        "graph_expansion: %d/%d seed files qualify (threshold=%.3f)",
        len(qualifying_seeds),
        len(seed_files),
        effective_expansion_min,
    )
    for fp in qualifying_seeds:
        imports_of_count = len(list(graph.imports_of(fp)))
        imported_by_count = len(list(graph.imported_by(fp)))
        logger.debug(
            "  seed %s (norm_score=%.3f): imports_of=%d imported_by=%d",
            fp,
            norm_seed_scores[fp],
            imports_of_count,
            imported_by_count,
        )
    expansion_priority: dict[str, float] = {}
    expansion_reasons_by_file: dict[str, set[str]] = {}
    expansion_reason_counts: dict[str, int] = {}
    expansion_source_count: dict[str, int] = {}
    import_neighbor_edges = 0
    same_module_candidates: set[str] = set()
    hub_candidates: set[str] = set()
    for file_path in seed_files:
        # Only expand from seeds above the confidence threshold
        if norm_seed_scores.get(file_path, 0.0) < effective_expansion_min:
            continue
        seed_score = seed_file_scores.get(file_path, 0.0)
        seed_module = file_to_module.get(file_path)
        # Direct imports get full seed score — they're in the same call chain
        for dep in graph.imports_of(file_path):
            if dep not in seed_files:
                import_neighbor_edges += 1
                if seed_module is not None and file_to_module.get(dep) == seed_module:
                    same_module_candidates.add(dep)
                if len(graph.imports_of(dep)) + len(graph.imported_by(dep)) >= 4:
                    hub_candidates.add(dep)
                    _record_expansion_reason(
                        expansion_reasons_by_file,
                        expansion_reason_counts,
                        dep,
                        "hub",
                    )
                # Boost imports whose file path matches a query term
                path_lower = dep.lower()
                path_match = any(t in path_lower for t in q_terms)
                priority = seed_score * (1.5 if path_match else 1.0)
                expansion_priority[dep] = expansion_priority.get(dep, 0.0) + priority
                expansion_source_count[dep] = expansion_source_count.get(dep, 0) + 1
                _record_expansion_reason(
                    expansion_reasons_by_file,
                    expansion_reason_counts,
                    dep,
                    "import_target",
                )
                dep_module = file_to_module.get(dep)
                if seed_module is not None and dep_module is not None:
                    module_reason = "same_module" if dep_module == seed_module else "cross_module"
                    _record_expansion_reason(
                        expansion_reasons_by_file,
                        expansion_reason_counts,
                        dep,
                        module_reason,
                    )
                if _is_entry_point(dep):
                    _record_expansion_reason(
                        expansion_reasons_by_file,
                        expansion_reason_counts,
                        dep,
                        "entry_point",
                    )
        # Importers get half seed score — they're consumers, not dependencies
        for imp in graph.imported_by(file_path):
            if imp not in seed_files:
                import_neighbor_edges += 1
                if seed_module is not None and file_to_module.get(imp) == seed_module:
                    same_module_candidates.add(imp)
                if len(graph.imports_of(imp)) + len(graph.imported_by(imp)) >= 4:
                    hub_candidates.add(imp)
                    _record_expansion_reason(
                        expansion_reasons_by_file,
                        expansion_reason_counts,
                        imp,
                        "hub",
                    )
                path_lower = imp.lower()
                path_match = any(t in path_lower for t in q_terms)
                priority = seed_score * (0.75 if path_match else 0.5)
                expansion_priority[imp] = expansion_priority.get(imp, 0.0) + priority
                expansion_source_count[imp] = expansion_source_count.get(imp, 0) + 1
                _record_expansion_reason(
                    expansion_reasons_by_file,
                    expansion_reason_counts,
                    imp,
                    "importer",
                )
                imp_module = file_to_module.get(imp)
                if seed_module is not None and imp_module is not None:
                    module_reason = "same_module" if imp_module == seed_module else "cross_module"
                    _record_expansion_reason(
                        expansion_reasons_by_file,
                        expansion_reason_counts,
                        imp,
                        module_reason,
                    )
                if _is_entry_point(imp):
                    _record_expansion_reason(
                        expansion_reasons_by_file,
                        expansion_reason_counts,
                        imp,
                        "entry_point",
                    )
    # Convergence bonus: files reached by multiple independent seeds are structurally
    # corroborated — more likely to be genuinely relevant than files from a single seed.
    for dep in expansion_priority:
        if expansion_source_count.get(dep, 0) >= 2:
            expansion_priority[dep] *= CONVERGENCE_BONUS

    total_expansion_candidates = len(expansion_priority)
    if total_expansion_candidates == 0:
        logger.debug(
            "graph_expansion: zero candidates found. seed_files=%s norm_scores=%s",
            list(seed_files),
            {fp: f"{norm_seed_scores.get(fp, 0.0):.3f}" for fp in seed_files},
        )
    else:
        logger.debug("graph_expansion: %d total expansion candidates", total_expansion_candidates)

    sorted_expansion = sorted(
        expansion_priority.keys(),
        key=lambda f: -expansion_priority[f],
    )

    # Build chunk lookup by file
    chunks_by_file = _chunks_by_file(all_chunks)

    # Collect candidate chunks (seed + file-capped expansion), dedup by id
    # Cap per-file to prevent one large file from monopolizing the expansion budget.
    # Skip test files in expansion — they add noise without improving relevance.
    max_per_file = 1 if intent == QueryIntent.CLI else 3
    # Vector-only seeds participate in scoring even when BM25 returned nothing
    # for that file.
    candidate_map = _initial_candidate_map(search_results, effective_vector, effective_splade)
    expansion_files_added = 0
    expansion_test_candidates_skipped = 0
    hop1_files_added: list[str] = []
    expanded_file_paths: list[str] = []
    for file_path in sorted_expansion:
        if _is_test_file(file_path):
            expansion_test_candidates_skipped += 1
            _record_expansion_reason(
                expansion_reasons_by_file,
                expansion_reason_counts,
                file_path,
                "test_file",
            )
            continue
        if _is_low_signal_expansion_candidate(
            file_path,
            expansion_reasons_by_file.get(file_path, set()),
            expansion_source_count.get(file_path, 0),
            q_terms,
        ):
            _record_expansion_reason(
                expansion_reasons_by_file,
                expansion_reason_counts,
                file_path,
                "skipped",
            )
            continue
        added = _add_file_chunks(
            candidate_map,
            chunks_by_file,
            file_path,
            max_per_file=max_per_file,
        )
        if added == 0:
            _record_expansion_reason(
                expansion_reasons_by_file,
                expansion_reason_counts,
                file_path,
                "skipped",
            )
        if added > 0:
            expansion_files_added += 1
            hop1_files_added.append(file_path)
            expanded_file_paths.append(file_path)
        if expansion_files_added >= MAX_EXPANSION_FILES:
            break

    expansion_zero_candidate_reason = _zero_expansion_reason(
        seed_files=seed_files,
        expansion_eligible_seeds=expansion_eligible_seeds,
        total_expansion_candidates=total_expansion_candidates,
        expansion_files_added=expansion_files_added,
    )

    # 2-hop expansion for architecture queries: follow imports_of for top hop-1 candidates
    if is_arch_query and hop1_files_added:
        remaining_budget = MAX_EXPANSION_FILES - expansion_files_added
        if remaining_budget > 0:
            hop2_files_added, hop2_test_candidates_skipped = _add_hop2_expansion(
                _Hop2Expansion(
                    graph=graph,
                    candidate_map=candidate_map,
                    chunks_by_file=chunks_by_file,
                    hop1_files_added=hop1_files_added,
                    seed_files=seed_files,
                    expansion_priority=expansion_priority,
                    query_terms=q_terms,
                    remaining_budget=remaining_budget,
                    max_per_file=max_per_file,
                )
            )
            expansion_test_candidates_skipped += hop2_test_candidates_skipped
            expansion_files_added += len(hop2_files_added)
            expanded_file_paths.extend(hop2_files_added)
            for file_path in hop2_files_added:
                _record_expansion_reason(
                    expansion_reasons_by_file,
                    expansion_reason_counts,
                    file_path,
                    "import_target",
                )
            if hop2_test_candidates_skipped:
                expansion_reason_counts["test_file"] = (
                    expansion_reason_counts.get("test_file", 0) + hop2_test_candidates_skipped
                )
    expanded_file_reasons = {
        file_path: sorted(expansion_reasons_by_file[file_path])
        for file_path in expanded_file_paths
        if file_path in expansion_reasons_by_file
    }

    candidates_after_expansion = len(candidate_map)

    if trace is not None:
        trace.add_step(
            StepTiming(
                name="graph_expansion",
                start_ns=_expansion_start,
                end_ns=time.perf_counter_ns(),
                metadata={
                    "seed_files": len(seed_files),
                    "expansion_files_added": expansion_files_added,
                    "candidates_after_expansion": candidates_after_expansion,
                    "expansion_eligible_seeds": expansion_eligible_seeds,
                    "expansion_candidates_found": total_expansion_candidates,
                    "expansion_import_neighbor_edges": import_neighbor_edges,
                    "expansion_same_module_candidates": len(same_module_candidates),
                    "expansion_hub_candidates": len(hub_candidates),
                    "expansion_test_candidates_skipped": expansion_test_candidates_skipped,
                    "expansion_zero_candidate_reason": expansion_zero_candidate_reason,
                    "expansion_reason_counts": ",".join(
                        f"{reason}:{count}"
                        for reason, count in sorted(expansion_reason_counts.items())
                    ),
                    "expanded_file_reasons": len(expanded_file_reasons),
                },
            )
        )

    # --- Cross-encoder reranking phase (opt-in) ---
    # Reranking is a scoring signal, not a candidate filter. Preserve the
    # complete candidate pool for file-level aggregation and only update scores
    # for the bounded window sent through the expensive cross-encoder.
    if reranker is not None:
        from archex.index.rerank import DEFAULT_TOP_K, CrossEncoderReranker

        if isinstance(reranker, CrossEncoderReranker):
            rerank_start = time.perf_counter_ns()
            candidate_list = [
                (chunk, bm25_by_id.get(chunk.id, 0.0)) for chunk in candidate_map.values()
            ]
            candidates_for_rerank = candidate_list[:rerank_candidate_limit]
            reranked = reranker.rerank(question, candidates_for_rerank, top_k=DEFAULT_TOP_K)
            for chunk_id, rerank_score in _normalized_rerank_scores(reranked).items():
                bm25_by_id[chunk_id] = max(bm25_by_id.get(chunk_id, 0.0), rerank_score)
            rerank_end = time.perf_counter_ns()
            if trace is not None:
                trace.add_step(
                    StepTiming(
                        name="rerank",
                        start_ns=rerank_start,
                        end_ns=rerank_end,
                        metadata={
                            "candidates_available": len(candidate_list),
                            "candidates_scored": len(candidates_for_rerank),
                            "candidates_returned": len(reranked),
                            "candidate_files_preserved": len(
                                {chunk.file_path for chunk in candidate_map.values()}
                            ),
                        },
                    )
                )
            logger.debug(
                "Cross-encoder reranked %d/%d candidates",
                len(candidates_for_rerank),
                len(candidate_list),
            )

    # --- Scoring phase ---
    _scoring_start = time.perf_counter_ns()

    # Get structural centrality scores
    centrality = graph.structural_centrality()

    # Signal agreement was computed pre-fusion; carry it forward for metadata
    signal_agreement: float | None = signal_agreement_pre if vector_results else None

    # Candidate file set for cohesion computation
    candidate_files = {c.file_path for c in candidate_map.values()}

    # Propagate BM25 relevance to import-expanded neighbors (directed).
    # Only propagate from seeds above the expansion confidence threshold.
    neighbor_boost = _neighbor_boosts(
        graph,
        seed_files,
        candidate_map,
        norm_seed_scores,
        effective_expansion_min,
        bm25_by_id,
        q_terms,
    )

    # Build RankedChunks
    ranked: list[RankedChunk] = []
    for chunk in candidate_map.values():
        relevance = bm25_by_id.get(chunk.id, 0.0) or neighbor_boost.get(chunk.file_path, 0.0)
        structural = centrality.get(chunk.file_path, 0.0)
        type_coverage = _type_alignment_score(
            chunk,
            alignment_terms,
            definition_lookup=intent == QueryIntent.DEFINITION_LOOKUP,
        )

        # Cohesion signal: proportion of co-module files present * module cohesion
        cohesion = 0.0
        mod = file_to_module.get(chunk.file_path)
        if mod and mod.files:
            co_present = sum(1 for f in mod.files if f in candidate_files)
            cohesion = (co_present / len(mod.files)) * mod.cohesion_score

        # Tests and benchmark/diagnostic helpers mirror runtime vocabulary.
        # Keep them searchable when explicitly requested, but rank product code first.
        support_penalty = 0.15 if _is_support_file(chunk.file_path, q_terms) else 1.0

        # Entry-point files (mod.rs, __init__.py, index.js) define module interfaces
        entry_boost = _ENTRY_POINT_BOOST if _is_entry_point(chunk.file_path) else 1.0

        # Directory-path alignment: files under directories matching query terms
        path_boost = _path_alignment_boost(chunk.file_path, alignment_terms)

        final = (
            (
                weights.relevance * relevance
                + weights.structural * structural
                + weights.type_coverage * type_coverage
                + weights.cohesion * cohesion
            )
            * support_penalty
            * entry_boost
            * path_boost
        )
        ranked.append(
            RankedChunk(
                chunk=chunk,
                relevance_score=relevance,
                structural_score=structural,
                type_coverage_score=type_coverage,
                cohesion_score=cohesion,
                final_score=final,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)

    # File-level ranking: aggregate per-file scores, apply score-relative cutoff,
    # then hard-cap at adaptive MAX_FILES to limit tail noise.
    # Use diminishing returns per file so one noisy file with many moderately
    # relevant chunks does not swamp a file with one or two highly relevant
    # chunks. The strongest chunk keeps full weight; each later chunk halves.
    file_agg = _aggregate_file_scores(ranked)
    sorted_files = sorted(file_agg.items(), key=lambda x: -x[1])
    top_file_score = sorted_files[0][1] if sorted_files else 0.0
    score_cutoff = top_file_score * _file_score_cutoff_ratio(
        fusion_applied=fusion_bm25_weight is not None or bool(effective_splade)
    )
    top_files: set[str] = set()
    adaptive_max = _adaptive_max_files(sorted_files)
    if intent == QueryIntent.CLI or q_terms & {
        "graph",
        "dependency",
        "dependencies",
        "import",
        "imports",
        "edges",
    }:
        cap = 5 if intent == QueryIntent.CLI and "api" in q_terms else 4
        if intent == QueryIntent.CLI and "api" in q_terms:
            adaptive_max = cap
        else:
            adaptive_max = min(adaptive_max, cap)
    aligned_files = {
        fp for fp, _score in sorted_files if _path_alignment_boost(fp, alignment_terms) > 1.0
    }
    for fp, _score in sorted_files:
        if fp not in aligned_files:
            continue
        top_files.add(fp)
        if len(top_files) >= adaptive_max:
            break
    for fp, score in sorted_files:
        if len(top_files) >= adaptive_max:
            break
        if fp in top_files:
            continue
        if score < score_cutoff:
            break
        top_files.add(fp)
    ranked = [rc for rc in ranked if rc.chunk.file_path in top_files]

    # Pack at least one high-scoring chunk per selected file before spending
    # remaining budget on extra chunks. This preserves file-level recall when one
    # high-scoring file has many chunks, while nested-range suppression prevents
    # class/module chunks and their child chunks from duplicating the same lines.
    included, total_tokens = _pack_ranked_chunks(
        ranked,
        sorted_files,
        top_files,
        token_budget,
        max_chunks_per_file=CLI_MAX_CHUNKS_PER_FILE if intent == QueryIntent.CLI else None,
    )

    chunks_dropped = len(ranked) - len(included)
    truncated = chunks_dropped > 0

    if trace is not None:
        trace.add_step(
            StepTiming(
                name="scoring",
                start_ns=_scoring_start,
                end_ns=time.perf_counter_ns(),
                metadata={
                    "candidates_scored": candidates_after_expansion,
                    "files_selected": len(top_files),
                    "chunks_included": len(included),
                    "chunks_dropped": chunks_dropped,
                },
            )
        )

    # --- Assembly phase ---
    _assembly_start = time.perf_counter_ns()

    # Build StructuralContext
    included_files = sorted({rc.chunk.file_path for rc in included})
    file_tree = "\n".join(included_files)

    structural_context = StructuralContext(
        file_tree=file_tree,
        file_dependency_subgraph=_dependency_subgraph(graph, included_files),
    )

    type_defs = _type_definitions(included)

    if trace is not None:
        trace.add_step(
            StepTiming(
                name="assembly",
                start_ns=_assembly_start,
                end_ns=time.perf_counter_ns(),
                metadata={
                    "included_files": len(included_files),
                    "type_definitions": len(type_defs),
                    "total_tokens": total_tokens,
                },
            )
        )

    assembly_ms = (time.perf_counter() - assembly_start) * 1000

    meta = RetrievalMetadata(
        candidates_found=candidates_found,
        candidates_after_expansion=candidates_after_expansion,
        chunks_included=len(included),
        chunks_dropped=chunks_dropped,
        strategy=strategy,
        assembly_time_ms=assembly_ms,
        signal_agreement=signal_agreement,
        fusion_bm25_weight=fusion_bm25_weight,
        fusion_vector_weight=fusion_vector_weight,
        seed_files_found=len(seed_file_paths),
        seed_file_paths=seed_file_paths,
        expanded_file_paths=expanded_file_paths,
        expansion_eligible_seeds=expansion_eligible_seeds,
        expansion_candidates_found=len(expansion_priority),
        expansion_files_added=expansion_files_added,
        fusion_skipped=fusion_skipped,
        fusion_skip_reason=fusion_skip_reason,
        expansion_zero_candidate_reason=expansion_zero_candidate_reason,
        expansion_import_neighbor_edges=import_neighbor_edges,
        expansion_same_module_candidates=len(same_module_candidates),
        expansion_hub_candidates=len(hub_candidates),
        expansion_test_candidates_skipped=expansion_test_candidates_skipped,
        expansion_reason_counts=dict(expansion_reason_counts),
        expanded_file_reasons=expanded_file_reasons,
        bm25_cv=bm25_cv_val,
        splade_results=len(splade_results or []),
        splade_used=bool(effective_splade),
        splade_fusion_skipped=splade_fusion_skipped,
        splade_fusion_skip_reason=splade_fusion_skip_reason,
    )

    return ContextBundle(
        query=question,
        chunks=included,
        structural_context=structural_context,
        type_definitions=type_defs,
        token_count=total_tokens,
        token_budget=token_budget,
        truncated=truncated,
        retrieval_metadata=meta,
    )
