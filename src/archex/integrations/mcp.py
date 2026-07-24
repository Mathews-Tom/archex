"""MCP integration: expose archex capabilities as Model Context Protocol tools."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from archex.api import (
    analyze,
    compare,
    context,
    file_outline,
    file_tree,
    get_file_token_count,
    get_files_token_count,
    get_repo_total_tokens,
    get_symbol,
    get_symbols_batch,
    index_repository,
    query,
    scout,
    search_symbols,
)
from archex.config import load_config, load_index_config
from archex.context_facade import (
    ContextBudgets,
    ContextFilters,
    ContextRequest,
    render_context_markdown,
)
from archex.explain import (
    ExplainError,
    explain_file,
    explain_graph_file,
    explain_graph_module,
    explain_graph_symbol,
    explain_module,
    explain_symbol,
    render_explain_context,
)
from archex.graph_artifact import (
    GraphArtifactError,
    build_arch_graph_from_store,
    load_arch_graph,
)
from archex.graph_query import (
    GraphDirection,
    GraphEdgeSummary,
    GraphHubsResult,
    GraphNeighborsResult,
    GraphNodeLookupResult,
    GraphNodeSummary,
    GraphPathResult,
    GraphQuery,
    GraphQueryError,
    GraphStatsResult,
)
from archex.impact import (
    ImpactError,
    ImpactFileChange,
    analyze_diff_impact,
    analyze_impact,
    git_changed_files,
    git_diff_hunks,
    render_impact_report,
)
from archex.metrics.capture import record_query_usage, record_scout_usage, record_structural_usage
from archex.metrics.health import note_metrics_recording_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import ContextBundle, PipelineTiming, RepoSource, RetrievalProfile
from archex.onboarding import OnboardingError, render_onboarding_markdown
from archex.reporting import compute_meta, count_tokens
from archex.scout import DEFAULT_SCOUT_TOKEN_BUDGET, ScoutFormat, ScoutResult, render_scout
from archex.serve.compare import validate_dimensions
from archex.serve.intent import DEFAULT_TOKEN_BUDGET, QueryIntent
from archex.serve.renderers.xml import render_xml, render_xml_envelope
from archex.serve.runtime import QueryRuntime
from archex.utils import resolve_source

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"json", "markdown"}
DEFAULT_GRAPH_TOKEN_BUDGET = 2000


def handle_analyze_repo(repo_url: str, output_format: str = "json") -> str:
    """Analyze a repository and return an architecture profile.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to analyze.
        output_format: Output format — 'json' or 'markdown'. Defaults to 'json'.

    Returns:
        JSON envelope with ArchProfile content and _meta efficiency block.
    """
    if output_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"format must be one of {sorted(_SUPPORTED_FORMATS)}, got {output_format!r}"
        )

    source = resolve_source(repo_url)
    pt = PipelineTiming()
    profile = analyze(source, timing=pt)

    content = profile.to_markdown() if output_format == "markdown" else profile.to_json()

    raw_tokens = get_repo_total_tokens(source)
    meta = compute_meta(
        tool_name="analyze_repo",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="full_analysis",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    _record_structural_metrics(
        source,
        "analyze_repo",
        content,
        raw_tokens,
        whole_repo_tokens=raw_tokens,
    )
    return json.dumps({"content": content, "_meta": meta.model_dump()}, indent=2)


def handle_query_repo(
    repo_url: str,
    question: str,
    budget: int | None = None,
    runtime: QueryRuntime | None = None,
    profile: str | None = None,
) -> str:
    """Retrieve context from a repository for a natural-language question.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to query.
        question: Natural-language question to answer from the codebase.
        budget: Optional explicit token budget override. Defaults to adaptive
            intent routing with a product ceiling of 8192.
        runtime: Optional warm QueryRuntime shared across calls in one server
            process. Omit for the exact pre-runtime per-call behavior.
        profile: Optional named retrieval profile — 'fast' (bm25 only, zero
            vector/model work), 'balanced' (adds module prefiltering), or
            'deep' (adds vector search and reranking). Omit to use the
            repo's configured IndexConfig unchanged.

    Returns:
        JSON envelope with ContextBundle content and _meta efficiency block.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if budget is not None and budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    resolved_profile: RetrievalProfile | None = None
    if profile is not None:
        try:
            resolved_profile = RetrievalProfile(profile)
        except ValueError as exc:
            valid = sorted(p.value for p in RetrievalProfile)
            raise ValueError(f"profile must be one of {valid}, got {profile!r}") from exc

    source = resolve_source(repo_url)
    pt = PipelineTiming()
    token_budget = DEFAULT_TOKEN_BUDGET if budget is None else budget
    bundle = query(
        source,
        question,
        token_budget=token_budget,
        timing=pt,
        explicit_token_budget=budget is not None,
        runtime=runtime,
        profile=resolved_profile,
    )

    content = render_xml(bundle, include_receipt=False)
    metadata = bundle.retrieval_metadata
    raw_file_paths = sorted(
        {
            *metadata.seed_file_paths,
            *metadata.expanded_file_paths,
            *(c.chunk.file_path for c in bundle.chunks),
        }
    )
    raw_tokens = get_files_token_count(source, raw_file_paths)
    envelope_overhead = count_tokens(render_xml_envelope(bundle))
    meta = compute_meta(
        tool_name="query_repo",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        envelope_overhead_tokens=envelope_overhead,
        strategy="bm25+graph",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
        delta=pt.delta_meta,
    )
    _record_query_metrics(source, bundle, raw_tokens)
    return json.dumps(
        {
            "content": content,
            "receipt": _receipt_payload(bundle.receipt),
            "_meta": meta.model_dump(),
        },
        indent=2,
    )


def handle_context(
    repo_url: str,
    query_text: str,
    intent: str | None = None,
    profile: str | None = None,
    filters: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
    handles: list[str] | None = None,
    output_format: str = "json",
) -> str:
    """Retrieve the primary agent-facing context result for a repository question.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to query.
        query_text: Natural-language question to answer from the codebase.
        intent: Optional query-intent override — pins the scoring-weight
            preset and default token budget instead of auto-classifying
            from `query_text`.
        profile: Optional named retrieval profile — 'fast', 'balanced', or
            'deep'. Omit to use the repo's configured retrieval settings.
        filters: Optional `{include_paths, exclude_paths, languages}`
            deterministic post-retrieval candidate filter.
        budgets: Optional `{token_budget}` explicit budget override.
        handles: Optional exact fetch handles — bypasses broad search and
            returns exactly these candidates.
        output_format: 'json' or 'markdown'. Defaults to 'json'.

    Returns:
        JSON envelope with content, candidate_map, fetch_handles,
        relation_paths, route, receipt, next_action, and the standard
        _meta efficiency block.
    """
    if not query_text.strip():
        raise ValueError("query must not be empty")
    if output_format not in {"json", "markdown"}:
        raise ValueError(f"format must be one of ['json', 'markdown'], got {output_format!r}")
    try:
        request = ContextRequest(
            query=query_text,
            intent=QueryIntent(intent) if intent is not None else None,
            profile=RetrievalProfile(profile) if profile is not None else None,
            filters=ContextFilters(**(filters or {})),
            budgets=ContextBudgets(**(budgets or {})),
            handles=list(handles or []),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid context request: {exc}") from exc

    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = context(source, request, timing=pt)

    content: Any
    if output_format == "markdown":
        content = render_context_markdown(result)
    else:
        content = [chunk.model_dump(mode="json") for chunk in result.selected_code]

    raw_file_paths = sorted({c.chunk.file_path for c in result.bundle.chunks})
    raw_tokens = get_files_token_count(source, raw_file_paths)
    response_text = content if isinstance(content, str) else json.dumps(content)
    meta = compute_meta(
        tool_name="context",
        response_text=response_text,
        raw_file_tokens=raw_tokens or 0,
        strategy="context_facade",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
        delta=pt.delta_meta,
    )
    _record_query_metrics(source, result.bundle, raw_tokens, tool_name="context")
    return json.dumps(
        {
            "content": content,
            "candidate_map": [item.model_dump(mode="json") for item in result.candidate_map],
            "fetch_handles": result.fetch_handles,
            "relation_paths": result.relation_paths.model_dump(mode="json"),
            "route": result.route.model_dump(mode="json"),
            "receipt": _receipt_payload(result.bundle.receipt),
            "next_action": result.next_action.value if result.next_action else None,
            "_meta": meta.model_dump(),
        },
        indent=2,
    )


def handle_scout_repo(
    repo_url: str,
    question: str,
    budget: int | None = None,
    output_format: ScoutFormat = "json",
) -> str:
    """Return a token-capped structural map with stable second-call handles."""
    _validate_output_format(output_format)
    source = resolve_source(repo_url)
    token_budget = DEFAULT_SCOUT_TOKEN_BUDGET if budget is None else budget
    pt = PipelineTiming()
    result = scout(
        source,
        question,
        token_budget=token_budget,
        output_format=output_format,
        timing=pt,
    )
    # M1 narrowed render_scout's default to a minimal JSON dump; the MCP
    # surface is explicitly out of scope for that change, so pin full=True
    # to keep this tool's response shape unchanged.
    rendered = render_scout(result, output_format=output_format, include_receipt=False, full=True)
    content = json.loads(rendered) if output_format == "json" else rendered
    raw = get_repo_total_tokens(source)
    meta = compute_meta(
        tool_name="scout_repo",
        response_text=rendered,
        raw_file_tokens=raw or 0,
        strategy="scout",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
        delta=pt.delta_meta,
    )
    receipt = _receipt_payload(result.receipt)
    _record_scout_metrics(source, result, rendered, raw)
    return json.dumps(
        {"content": content, "receipt": receipt, "_meta": meta.model_dump()},
        indent=2,
    )


def _record_query_metrics(
    source: RepoSource,
    bundle: ContextBundle,
    raw_tokens: int | None,
    *,
    tool_name: str = "query_repo",
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        whole_repo_tokens = _whole_repo_tokens(source)
        record_query_usage(
            source,
            bundle,
            surface="mcp",
            tool_name=tool_name,
            tokens_raw_equivalent=raw_tokens,
            whole_repo_tokens=whole_repo_tokens,
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def _record_scout_metrics(
    source: RepoSource,
    result: ScoutResult,
    rendered: str,
    whole_repo_tokens: int | None,
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        raw_tokens = get_files_token_count(source, _scout_file_paths(result))
        record_scout_usage(
            source,
            result,
            surface="mcp",
            tool_name="scout_repo",
            tokens_returned=count_tokens(rendered),
            tokens_raw_equivalent=raw_tokens,
            whole_repo_tokens=whole_repo_tokens,
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def _whole_repo_tokens(source: RepoSource) -> int | None:
    try:
        return get_repo_total_tokens(source)
    except Exception:
        logger.debug("MCP metrics whole-repo tokens unavailable", exc_info=True)
        return None


def _scout_file_paths(result: ScoutResult) -> list[str]:
    paths = {item.path for item in result.ranked_files}
    paths.update(symbol.file_path for symbol in result.symbols)
    for module in result.modules:
        paths.update(module.relevant_files)
    for edge in result.graph:
        if edge.source_path is not None:
            paths.add(edge.source_path)
        if edge.target_path is not None:
            paths.add(edge.target_path)
    return sorted(paths)


def _record_structural_metrics(
    source: RepoSource,
    tool_name: str,
    response_text: str,
    raw_tokens: int | None,
    *,
    whole_repo_tokens: int | None = None,
    file_count: int = 0,
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        record_structural_usage(
            source,
            surface="mcp",
            tool_name=tool_name,
            tokens_returned=count_tokens(response_text),
            tokens_raw_equivalent=raw_tokens,
            whole_repo_tokens=whole_repo_tokens,
            file_count=file_count,
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def handle_compare_repos(
    repo_a: str,
    repo_b: str,
    dimensions: str = "api_surface,error_handling",
) -> str:
    """Compare two repositories across architectural dimensions.

    Args:
        repo_a: Local path or HTTP(S) URL of the first repository.
        repo_b: Local path or HTTP(S) URL of the second repository.
        dimensions: Comma-separated list of dimensions to compare.
            Supported values: error_handling, api_surface, state_management,
            concurrency, testing, configuration.
            Defaults to 'api_surface,error_handling'.

    Returns:
        JSON envelope with ComparisonResult content and _meta efficiency block.
    """
    dim_list = [d.strip() for d in dimensions.split(",") if d.strip()]
    if not dim_list:
        raise ValueError("dimensions must be a non-empty comma-separated list")
    validate_dimensions(dim_list)

    source_a = resolve_source(repo_a)
    source_b = resolve_source(repo_b)
    t0 = time.perf_counter()
    result = compare(source_a, source_b, dimensions=dim_list)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    content = result.model_dump_json(indent=2)
    raw_a = get_repo_total_tokens(source_a)
    raw_b = get_repo_total_tokens(source_b)
    raw_tokens = raw_a + raw_b if raw_a is not None and raw_b is not None else None
    meta = compute_meta(
        tool_name="compare_repos",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="full_comparison",
        query_time_ms=elapsed_ms,
    )
    _record_structural_metrics(
        source_a,
        "compare_repos",
        content,
        raw_tokens,
        whole_repo_tokens=raw_tokens,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_get_file_tree(repo_url: str, max_depth: int = 5, language: str | None = None) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = file_tree(source, max_depth=max_depth, language=language, timing=pt)
    content = result.model_dump_json(indent=2)
    raw_tokens = get_repo_total_tokens(source)
    meta = compute_meta(
        tool_name="get_file_tree",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="file_tree",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    _record_structural_metrics(
        source,
        "get_file_tree",
        content,
        raw_tokens,
        whole_repo_tokens=raw_tokens,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_get_file_outline(repo_url: str, file_path: str) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = file_outline(source, file_path=file_path, timing=pt)
    content = result.model_dump_json(indent=2)
    meta = compute_meta(
        tool_name="get_file_outline",
        response_text=content,
        raw_file_tokens=result.token_count_raw,
        strategy="file_outline",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    _record_structural_metrics(
        source,
        "get_file_outline",
        content,
        result.token_count_raw,
        file_count=1,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_search_symbols(
    repo_url: str,
    query_text: str,
    kind: str | None = None,
    language: str | None = None,
    limit: int = 20,
) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    matches = search_symbols(
        source, query=query_text, kind=kind, language=language, limit=limit, timing=pt
    )
    match_data = [m.model_dump() for m in matches]
    content = json.dumps(match_data, indent=2)
    unique_files = list({m.file_path for m in matches})
    raw_tokens = get_files_token_count(source, unique_files) if unique_files else 0
    meta = compute_meta(
        tool_name="search_symbols",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="symbol_search",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    _record_structural_metrics(
        source,
        "search_symbols",
        content,
        raw_tokens,
        file_count=len(unique_files),
    )
    return json.dumps({"content": match_data, "_meta": meta.model_dump()}, indent=2)


def handle_get_symbol(repo_url: str, symbol_id: str) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = get_symbol(source, symbol_id=symbol_id, timing=pt)
    if result is None:
        return json.dumps({"error": "Symbol not found", "symbol_id": symbol_id})
    content = result.model_dump_json(indent=2)
    raw_tokens = get_file_token_count(source, result.file_path)
    meta = compute_meta(
        tool_name="get_symbol",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="symbol_lookup",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    _record_structural_metrics(
        source,
        "get_symbol",
        content,
        raw_tokens,
        file_count=1,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_get_symbols_batch(repo_url: str, symbol_ids: list[str]) -> str:
    if len(symbol_ids) > 50:
        raise ValueError(f"symbol_ids must contain at most 50 entries, got {len(symbol_ids)}")
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    results = get_symbols_batch(source, symbol_ids=symbol_ids, timing=pt)
    result_data = [s.model_dump() if s else None for s in results]
    content = json.dumps(result_data, indent=2)
    unique_files = list({s.file_path for s in results if s is not None})
    raw_tokens = get_files_token_count(source, unique_files) if unique_files else 0
    meta = compute_meta(
        tool_name="get_symbols_batch",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="symbol_batch",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    _record_structural_metrics(
        source,
        "get_symbols_batch",
        content,
        raw_tokens,
        file_count=len(unique_files),
    )
    return json.dumps({"content": result_data, "_meta": meta.model_dump()}, indent=2)


def handle_get_impact(
    repo_url: str,
    base: str = "main",
    changed_files: list[str] | None = None,
    output_format: str = "json",
    diff_ref: str | None = None,
) -> str:
    """Analyze deterministic blast radius for changed files.

    Args:
        repo_url: Local filesystem path of the repository. Git diff mode
            (the default, when changed_files is omitted) requires a local
            checkout with the given base ref reachable.
        base: Base ref for git diff mode. Ignored when changed_files or
            diff_ref is given.
        changed_files: Explicit changed file paths, bypassing git diff mode.
            Cannot be combined with diff_ref.
        output_format: Output format — 'json' or 'markdown'. Defaults to 'json'.
        diff_ref: Enable diff-scoped symbol impact: resolve the diff (this ref
            vs. the working tree) to touched symbols and classify each with a
            LOW/MEDIUM/HIGH risk tier from deterministic graph signals. Adds
            affected_symbols to the report; output is unchanged when omitted.

    Returns:
        JSON envelope with ImpactReport content and _meta efficiency block.
    """
    _validate_output_format(output_format)
    if changed_files and diff_ref is not None:
        raise ImpactError("get_impact: diff_ref cannot be combined with changed_files")
    source = resolve_source(repo_url)
    repo_root: Path | None = None
    if source.local_path is not None:
        repo_root = Path(source.local_path).expanduser().resolve()
    if changed_files:
        changes = [ImpactFileChange(path=path) for path in changed_files]
    else:
        if repo_root is None:
            raise ImpactError(
                "get_impact requires changed_files for a remote repo_url; "
                "git diff mode needs a local checkout"
            )
        changes = git_changed_files(repo_root, diff_ref if diff_ref is not None else base)

    started = time.perf_counter()
    config = load_config(source)
    index_config = load_index_config(source)
    pt = PipelineTiming()
    store = index_repository(source, config=config, timing=pt, index_config=index_config)
    try:
        if diff_ref is not None:
            assert repo_root is not None  # guaranteed: local checkout resolved above
            hunks = git_diff_hunks(repo_root, diff_ref)
            report = analyze_diff_impact(store, repo_root, changes, hunks, diff_ref)
        else:
            report = analyze_impact(store, changes)
    finally:
        store.close()
    query_time_ms = (time.perf_counter() - started) * 1000

    content = render_impact_report(report, output_format)
    raw_tokens = (
        get_files_token_count(source, report.affected_files) if report.affected_files else 0
    )
    meta = compute_meta(
        tool_name="get_impact",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="impact_analysis",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=query_time_ms,
    )
    _record_structural_metrics(
        source,
        "get_impact",
        content,
        raw_tokens,
        file_count=len(report.affected_files),
    )
    return json.dumps({"content": content, "_meta": meta.model_dump()}, indent=2)


def handle_explain_target(
    repo_url: str | None = None,
    target: str | None = None,
    module_name: str | None = None,
    graph_path: str | None = None,
    output_format: str = "json",
) -> str:
    """Explain a file, symbol, or module from indexed structural data.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository. Required unless
            graph_path is given.
        target: File path or `path::name#kind` symbol identifier to explain.
            Mutually exclusive with module_name.
        module_name: Module path to explain. Mutually exclusive with target.
        graph_path: Read an exported graph artifact instead of indexing
            repo_url.
        output_format: Output format — 'json' or 'markdown'. Defaults to 'json'.

    Returns:
        JSON envelope with ExplainContext content and _meta efficiency block.
    """
    _validate_output_format(output_format)
    if module_name is not None and target is not None:
        raise ExplainError("use either target or module_name, not both")
    if module_name is None and target is None:
        raise ExplainError("explain_target requires target or module_name")
    if graph_path is None and repo_url is None:
        raise ExplainError("explain_target requires repo_url when graph_path is not provided")

    started = time.perf_counter()
    source: RepoSource | None = None
    pt: PipelineTiming | None = None
    if graph_path is not None:
        graph = load_arch_graph(Path(graph_path))
        if module_name is not None:
            context = explain_graph_module(graph, module_name)
        elif target is not None and ("::" in target or "#" in target):
            context = explain_graph_symbol(graph, target)
        elif target is not None:
            context = explain_graph_file(graph, target)
        else:
            raise ExplainError("explain_target requires target or module_name")
        raw_tokens = max(count_tokens(graph.to_json()), 1)
    else:
        assert repo_url is not None
        source = resolve_source(repo_url)
        config = load_config(source)
        index_config = load_index_config(source)
        pt = PipelineTiming()
        store = index_repository(source, config=config, timing=pt, index_config=index_config)
        try:
            if module_name is not None:
                context = explain_module(store, module_name)
            elif target is not None and ("::" in target or "#" in target):
                context = explain_symbol(store, target)
            elif target is not None:
                context = explain_file(store, target)
            else:
                raise ExplainError("explain_target requires target or module_name")
        finally:
            store.close()
        raw_tokens = get_files_token_count(source, context.files) if context.files else 0
    query_time_ms = (time.perf_counter() - started) * 1000

    content = render_explain_context(context, output_format)
    meta = compute_meta(
        tool_name="explain_target",
        response_text=content,
        raw_file_tokens=raw_tokens or 0,
        strategy="explain_context",
        cached=pt.cached if pt is not None else False,
        index_time_ms=pt.index_ms if pt is not None else 0.0,
        query_time_ms=query_time_ms,
    )
    if source is not None:
        _record_structural_metrics(
            source,
            "explain_target",
            content,
            raw_tokens,
            file_count=len(context.files),
        )
    return json.dumps({"content": content, "_meta": meta.model_dump()}, indent=2)


def handle_generate_onboarding(
    repo_url: str | None = None,
    graph_path: str | None = None,
    max_files: int = 40,
) -> str:
    """Generate a deterministic onboarding guide from graph/index data.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository. Required
            unless graph_path is given.
        graph_path: Read an exported graph artifact instead of indexing
            repo_url.
        max_files: Maximum paths per capped section. Defaults to 40.

    Returns:
        JSON envelope with the markdown onboarding guide and _meta
        efficiency block. Onboarding output is markdown-only.
    """
    if graph_path is None and repo_url is None:
        raise OnboardingError(
            "generate_onboarding requires repo_url when graph_path is not provided"
        )

    started = time.perf_counter()
    source: RepoSource | None = None
    pt: PipelineTiming | None = None
    if graph_path is not None:
        graph = load_arch_graph(Path(graph_path))
        raw_tokens = max(count_tokens(graph.to_json()), 1)
    else:
        assert repo_url is not None
        source = resolve_source(repo_url)
        repo_root = Path(source.local_path).expanduser().resolve() if source.local_path else None
        config = load_config(source)
        index_config = load_index_config(source)
        pt = PipelineTiming()
        store = index_repository(source, config=config, timing=pt, index_config=index_config)
        try:
            graph = build_arch_graph_from_store(store, repo_root=repo_root)
        finally:
            store.close()
        raw_tokens = get_repo_total_tokens(source) or 0
    query_time_ms = (time.perf_counter() - started) * 1000

    content = render_onboarding_markdown(graph, max_files=max_files)
    meta = compute_meta(
        tool_name="generate_onboarding",
        response_text=content,
        raw_file_tokens=raw_tokens,
        strategy="onboarding_guide",
        cached=pt.cached if pt is not None else False,
        index_time_ms=pt.index_ms if pt is not None else 0.0,
        query_time_ms=query_time_ms,
    )
    if source is not None:
        _record_structural_metrics(
            source,
            "generate_onboarding",
            content,
            raw_tokens,
            whole_repo_tokens=raw_tokens,
        )
    return json.dumps({"content": content, "_meta": meta.model_dump()}, indent=2)


def handle_graph_lookup(
    graph_path: str,
    node: str,
    output_format: str = "json",
    limit: int = 25,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.lookup(node, limit=limit)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_lookup",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_lookup_markdown,
    )


def handle_graph_neighbors(
    graph_path: str,
    node: str,
    output_format: str = "json",
    direction: GraphDirection = "both",
    depth: int = 1,
    limit: int = 25,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.neighbors(node, direction=direction, depth=depth, limit=limit)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path, "node": node})
    return _graph_tool_response(
        "graph_neighbors",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_neighbors_markdown,
    )


def handle_graph_path(
    graph_path: str,
    source: str,
    target: str,
    output_format: str = "json",
    direction: GraphDirection = "both",
    max_edges: int = 100,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.shortest_path(
            source,
            target,
            direction=direction,
            max_edges=max_edges,
        )
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_path",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_path_markdown,
    )


def handle_graph_stats(
    graph_path: str,
    output_format: str = "json",
    hub_limit: int = 10,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.stats(hub_limit=hub_limit)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_stats",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_stats_markdown,
    )


def handle_graph_hubs(
    graph_path: str,
    output_format: str = "json",
    limit: int = 25,
    threshold: int | None = None,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.hubs(limit=limit, threshold=threshold)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_hubs",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_hubs_markdown,
    )


_GRAPH_QUERY_CACHE_MAXSIZE = 16
_graph_query_cache: OrderedDict[tuple[str, int], tuple[tuple[int, int], GraphQuery]] = OrderedDict()
_graph_query_cache_lock = threading.Lock()


def _graph_artifact_cache_token(graph_path: str) -> tuple[int, int]:
    """Cache-busting token for `_cached_graph_query`: (mtime_ns, size_bytes).

    `graph_path` normally names a fixed, repeatedly re-exported project path
    (`archex graph export` writes there). A long-running `archex mcp` process
    would otherwise keep serving the first snapshot it loaded for that path
    forever, even after the user re-exports fresh data. Keying the cache on
    the artifact's mtime and size makes a re-export a cache miss. Pairing
    size with mtime — both free from the same stat() call — narrows the
    (unavoidable, filesystem-timestamp-resolution-dependent) collision
    window: two re-exports landing within one mtime tick on a coarse
    filesystem still typically differ in byte size. Falls back to (0, 0)
    for a missing/unreadable path so the eventual `GraphArtifactError` from
    `GraphQuery.from_artifact` still surfaces normally instead of raising
    here.
    """
    try:
        st = Path(graph_path).expanduser().resolve().stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def _cached_graph_query(graph_path: str, hub_degree: int) -> GraphQuery:
    """Return a cached GraphQuery for (graph_path, hub_degree), rebuilding on re-export.

    Deliberately not a plain `functools.lru_cache`: keying on the artifact
    token as part of the cache key (rather than checking it explicitly)
    would let every re-export of the same path accumulate a new,
    never-superseded lru_cache slot instead of replacing the stale one —
    trading the staleness bug for a bounded-but-real memory-footprint
    regression for exactly the frequent-re-export workflow this exists to
    serve well. This cache holds at most one live GraphQuery per distinct
    (graph_path, hub_degree) pair, evicting the least-recently-used pair
    once more than `_GRAPH_QUERY_CACHE_MAXSIZE` distinct pairs are resident.
    """
    key = (graph_path, hub_degree)
    token = _graph_artifact_cache_token(graph_path)
    with _graph_query_cache_lock:
        cached = _graph_query_cache.get(key)
        if cached is not None and cached[0] == token:
            _graph_query_cache.move_to_end(key)
            return cached[1]

    # Built outside the lock: from_artifact() parses a JSON artifact and
    # builds adjacency indices, which must not serialize every concurrent
    # graph-tool call across every repo behind one lock while it runs.
    # A concurrent re-check for the same key is a redundant rebuild, not a
    # correctness issue — whichever result is stored last simply wins.
    query = GraphQuery.from_artifact(Path(graph_path).expanduser().resolve(), hub_degree=hub_degree)
    with _graph_query_cache_lock:
        _graph_query_cache[key] = (token, query)
        _graph_query_cache.move_to_end(key)
        while len(_graph_query_cache) > _GRAPH_QUERY_CACHE_MAXSIZE:
            _graph_query_cache.popitem(last=False)
    return query


def clear_graph_query_cache() -> None:
    """Clear cached graph artifact handles used by MCP graph tools."""
    with _graph_query_cache_lock:
        _graph_query_cache.clear()


def _graph_tool_response(
    tool_name: str,
    graph_query: GraphQuery,
    result: GraphNodeLookupResult
    | GraphNeighborsResult
    | GraphPathResult
    | GraphStatsResult
    | GraphHubsResult,
    output_format: str,
    token_budget: int,
    started: float,
    markdown_renderer: Any,
) -> str:
    _validate_output_format(output_format)
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1")
    if output_format == "markdown":
        rendered_content, budget_truncated = _apply_token_budget(
            markdown_renderer(result),
            token_budget,
        )
        response_text = rendered_content
        content: str | dict[str, Any] = rendered_content
    else:
        content = result.model_dump(mode="json")
        response_text = json.dumps(content, indent=2, sort_keys=True)
        budget_truncated = False
    raw_tokens = max(count_tokens(graph_query.graph.to_json()), 1)
    query_time_ms = (time.perf_counter() - started) * 1000
    meta = compute_meta(
        tool_name=tool_name,
        response_text=response_text,
        raw_file_tokens=raw_tokens,
        strategy="graph_query",
        query_time_ms=query_time_ms,
    ).model_dump()
    meta["format"] = output_format
    meta["token_budget"] = token_budget
    meta["token_budget_truncated"] = budget_truncated
    return json.dumps({"content": content, "_meta": meta}, indent=2, sort_keys=True)


def _receipt_payload(receipt: Any) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return receipt.model_dump(mode="json")


def _validate_output_format(output_format: str) -> None:
    if output_format not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {output_format!r}; expected json or markdown")


def _apply_token_budget(text: str, token_budget: int) -> tuple[str, bool]:
    if count_tokens(text) <= token_budget:
        return text, False
    marker = "\n\n[truncated: token budget reached]\n"
    lines: list[str] = []
    for line in text.splitlines():
        candidate = "\n".join([*lines, line]) + marker
        if count_tokens(candidate) > token_budget:
            break
        lines.append(line)
    if not lines:
        return marker.lstrip(), True
    return "\n".join(lines).rstrip() + marker, True


def _render_lookup_markdown(result: GraphNodeLookupResult) -> str:
    lines = [
        f"# Graph Lookup: {result.query}",
        "",
        f"- Match kind: `{result.match_kind or 'none'}`",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    lines.extend(_render_node_list(result.matches))
    return _finish_markdown(lines)


def _render_neighbors_markdown(result: GraphNeighborsResult) -> str:
    lines = [
        f"# Graph Neighbors: {result.seed.id}",
        "",
        f"- Path: `{result.seed.path or result.seed.id}`",
        f"- Direction: `{result.direction}`",
        f"- Depth: {result.depth}",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
        "## Edges",
        "",
    ]
    lines.extend(_render_edge_list(result.edges))
    if result.hubs:
        lines.extend(["", "## Terminal Hubs", ""])
        lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_path_markdown(result: GraphPathResult) -> str:
    source = result.source.path if result.source is not None else result.source_query
    target = result.target.path if result.target is not None else result.target_query
    lines = [
        f"# Graph Path: {source} -> {target}",
        "",
        f"- Found: {_yes_no(result.found)}",
        f"- Direction: `{result.direction}`",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    if result.nodes:
        lines.extend(["## Nodes", ""])
        lines.extend(_render_node_list(result.nodes))
        lines.extend(["", "## Edges", ""])
        lines.extend(_render_edge_list(result.edges))
    if result.avoided_hubs:
        lines.extend(["", "## Avoided Hubs", ""])
        lines.extend(_render_node_list(result.avoided_hubs))
    return _finish_markdown(lines)


def _render_stats_markdown(result: GraphStatsResult) -> str:
    lines = [
        f"# Graph Stats: {result.project}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Nodes | {result.nodes} |",
        f"| Edges | {result.edges} |",
        f"| Files | {result.files} |",
        f"| Max degree | {result.max_degree} |",
        "",
    ]
    if result.edge_types:
        lines.extend(["## Edge Types", "", "| Type | Count |", "| --- | ---: |"])
        for edge_type, count in sorted(result.edge_types.items()):
            lines.append(f"| {edge_type} | {count} |")
        lines.append("")
    if result.hubs:
        lines.extend(["## Hubs", ""])
        lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_hubs_markdown(result: GraphHubsResult) -> str:
    lines = [
        "# Graph Hubs",
        "",
        f"- Threshold: {result.threshold}",
        f"- Limit: {result.limit}",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_edge_list(edges: list[GraphEdgeSummary]) -> list[str]:
    if not edges:
        return ["No edges."]
    lines = [
        "| Source path | Kind | Target path | Confidence | Evidence |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for edge in edges:
        evidence = "; ".join(edge.evidence) if edge.evidence else ""
        lines.append(
            "| "
            f"`{edge.source.path or edge.source.id}` | "
            f"{edge.type} | "
            f"`{edge.target.path or edge.target.id}` | "
            f"{edge.confidence} ({edge.confidence_score:.2f}) | "
            f"{evidence} |"
        )
    return lines


def _render_node_list(nodes: list[GraphNodeSummary]) -> list[str]:
    if not nodes:
        return ["No nodes."]
    lines = ["| Path | ID | Type | Degree |", "| --- | --- | --- | ---: |"]
    for node in nodes:
        lines.append(f"| `{node.path or ''}` | `{node.id}` | {node.type} | {node.degree} |")
    return lines


def _finish_markdown(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


async def _run_mcp_tool(
    loop: asyncio.AbstractEventLoop,
    name: str,
    arguments: dict[str, Any],
    runtime: QueryRuntime | None = None,
) -> str:
    if name == "analyze_repo":
        repo_url: str = arguments["repo_url"]
        fmt: str = arguments.get("format", "json")
        return await loop.run_in_executor(None, handle_analyze_repo, repo_url, fmt)
    if name == "scout_repo":
        repo_url = arguments["repo_url"]
        question = arguments["question"]
        budget_arg = arguments.get("budget")
        budget = int(budget_arg) if budget_arg is not None else None
        fmt_arg = arguments.get("format", "json")
        scout_fmt: ScoutFormat = "markdown" if fmt_arg == "markdown" else "json"
        return await loop.run_in_executor(
            None, handle_scout_repo, repo_url, question, budget, scout_fmt
        )
    if name == "query_repo":
        repo_url = arguments["repo_url"]
        question: str = arguments["question"]
        budget_arg = arguments.get("budget")
        budget = int(budget_arg) if budget_arg is not None else None
        profile_arg: str | None = arguments.get("profile")
        return await loop.run_in_executor(
            None, handle_query_repo, repo_url, question, budget, runtime, profile_arg
        )
    if name == "context":
        context_repo_url: str = arguments["repo_url"]
        context_query: str = arguments["query"]
        context_intent: str | None = arguments.get("intent")
        context_profile: str | None = arguments.get("profile")
        context_filters: dict[str, Any] | None = arguments.get("filters")
        context_budgets: dict[str, Any] | None = arguments.get("budgets")
        context_handles: list[str] | None = arguments.get("handles")
        context_format: str = arguments.get("format", "json")
        return await loop.run_in_executor(
            None,
            handle_context,
            context_repo_url,
            context_query,
            context_intent,
            context_profile,
            context_filters,
            context_budgets,
            context_handles,
            context_format,
        )
    if name == "compare_repos":
        repo_a: str = arguments["repo_a"]
        repo_b: str = arguments["repo_b"]
        dims: str = arguments.get("dimensions", "api_surface,error_handling")
        return await loop.run_in_executor(None, handle_compare_repos, repo_a, repo_b, dims)
    if name == "get_file_tree":
        repo_url = arguments["repo_url"]
        max_depth: int = int(arguments.get("max_depth", 5))
        language: str | None = arguments.get("language")
        return await loop.run_in_executor(None, handle_get_file_tree, repo_url, max_depth, language)
    if name == "get_file_outline":
        repo_url = arguments["repo_url"]
        file_path: str = arguments["file_path"]
        return await loop.run_in_executor(None, handle_get_file_outline, repo_url, file_path)
    if name == "search_symbols":
        repo_url = arguments["repo_url"]
        sym_query: str = arguments["query"]
        kind: str | None = arguments.get("kind")
        language = arguments.get("language")
        limit: int = int(arguments.get("limit", 20))
        return await loop.run_in_executor(
            None, handle_search_symbols, repo_url, sym_query, kind, language, limit
        )
    if name == "get_symbol":
        repo_url = arguments["repo_url"]
        symbol_id: str = arguments["symbol_id"]
        return await loop.run_in_executor(None, handle_get_symbol, repo_url, symbol_id)
    if name == "get_symbols_batch":
        repo_url = arguments["repo_url"]
        symbol_ids: list[str] = arguments["symbol_ids"]
        return await loop.run_in_executor(None, handle_get_symbols_batch, repo_url, symbol_ids)
    if name == "get_impact":
        repo_url = arguments["repo_url"]
        base: str = arguments.get("base", "main")
        impact_changed_files: list[str] | None = arguments.get("changed_files")
        fmt = arguments.get("format", "json")
        impact_diff_ref: str | None = arguments.get("diff")
        return await loop.run_in_executor(
            None, handle_get_impact, repo_url, base, impact_changed_files, fmt, impact_diff_ref
        )
    if name == "explain_target":
        explain_repo_url: str | None = arguments.get("repo_url")
        explain_target_arg: str | None = arguments.get("target")
        module_name: str | None = arguments.get("module_name")
        explain_graph_path: str | None = arguments.get("graph_path")
        fmt = arguments.get("format", "json")
        return await loop.run_in_executor(
            None,
            handle_explain_target,
            explain_repo_url,
            explain_target_arg,
            module_name,
            explain_graph_path,
            fmt,
        )
    if name == "generate_onboarding":
        onboard_repo_url: str | None = arguments.get("repo_url")
        onboard_graph_path: str | None = arguments.get("graph_path")
        max_files = int(arguments.get("max_files", 40))
        return await loop.run_in_executor(
            None, handle_generate_onboarding, onboard_repo_url, onboard_graph_path, max_files
        )
    if name == "graph_lookup":
        graph_path = arguments["graph_path"]
        node: str = arguments["node"]
        fmt = arguments.get("format", "json")
        limit = int(arguments.get("limit", 25))
        hub_degree = int(arguments.get("hub_degree", 50))
        token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
        return await loop.run_in_executor(
            None,
            handle_graph_lookup,
            graph_path,
            node,
            fmt,
            limit,
            hub_degree,
            token_budget,
        )
    if name == "graph_neighbors":
        graph_path = arguments["graph_path"]
        node = arguments["node"]
        fmt = arguments.get("format", "json")
        direction: GraphDirection = arguments.get("direction", "both")
        depth = int(arguments.get("depth", 1))
        limit = int(arguments.get("limit", 25))
        hub_degree = int(arguments.get("hub_degree", 50))
        token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
        return await loop.run_in_executor(
            None,
            handle_graph_neighbors,
            graph_path,
            node,
            fmt,
            direction,
            depth,
            limit,
            hub_degree,
            token_budget,
        )
    if name == "graph_path":
        graph_path = arguments["graph_path"]
        source: str = arguments["source"]
        target: str = arguments["target"]
        fmt = arguments.get("format", "json")
        direction = arguments.get("direction", "both")
        max_edges = int(arguments.get("max_edges", 100))
        hub_degree = int(arguments.get("hub_degree", 50))
        token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
        return await loop.run_in_executor(
            None,
            handle_graph_path,
            graph_path,
            source,
            target,
            fmt,
            direction,
            max_edges,
            hub_degree,
            token_budget,
        )
    if name == "graph_stats":
        graph_path = arguments["graph_path"]
        fmt = arguments.get("format", "json")
        hub_limit = int(arguments.get("hub_limit", 10))
        hub_degree = int(arguments.get("hub_degree", 50))
        token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
        return await loop.run_in_executor(
            None,
            handle_graph_stats,
            graph_path,
            fmt,
            hub_limit,
            hub_degree,
            token_budget,
        )
    if name == "graph_hubs":
        graph_path = arguments["graph_path"]
        fmt = arguments.get("format", "json")
        limit = int(arguments.get("limit", 25))
        threshold_arg = arguments.get("threshold")
        threshold = int(threshold_arg) if threshold_arg is not None else None
        hub_degree = int(arguments.get("hub_degree", 50))
        token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
        return await loop.run_in_executor(
            None,
            handle_graph_hubs,
            graph_path,
            fmt,
            limit,
            threshold,
            hub_degree,
            token_budget,
        )
    raise ValueError(f"Unknown tool: {name!r}")


def _tool_schemas() -> list[dict[str, Any]]:
    """Full unscoped MCP tool schema definitions (name/description/inputSchema).

    Single source of truth for `build_server`'s `list_tools()` handler, tool-scope
    filtering (`resolve_tool_scope`), and the `archex mcp-schema-size` measurement
    command, so scoping and reported schema sizes can never drift from the tools
    the server actually registers.
    """
    return [
        {
            "name": "analyze_repo",
            "description": (
                "Analyze a code repository and return an architecture profile including "
                "modules, design patterns, interfaces, dependency graph, and architectural "
                "decisions. Works with local paths and remote Git URLs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local filesystem path or HTTP(S) Git URL of the repository."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                        "description": "Output format for the architecture profile.",
                    },
                },
                "required": ["repo_url"],
            },
        },
        {
            "name": "scout_repo",
            "description": (
                "Return a compact structural scout map for a repository question. "
                "The map contains ranked files, module boundaries, top symbols, graph "
                "sketches, and stable file/symbol/chunk handles, but no code bodies."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local filesystem path or HTTP(S) Git URL of the repository."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": "Natural-language scout question.",
                    },
                    "budget": {
                        "type": "integer",
                        "default": DEFAULT_SCOUT_TOKEN_BUDGET,
                        "description": "Hard token cap for the scout map.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                },
                "required": ["repo_url", "question"],
            },
        },
        {
            "name": "query_repo",
            "description": (
                "Retrieve relevant code context from a repository to answer a "
                "natural-language question. Returns a ranked set of code chunks "
                "within the specified token budget, suitable for use as LLM context."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local filesystem path or HTTP(S) Git URL of the repository."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": ("Natural-language question to answer from the codebase."),
                    },
                    "budget": {
                        "type": "integer",
                        "description": (
                            "Optional explicit token budget override. Omit to use "
                            "adaptive intent routing with the 8192 product ceiling."
                        ),
                    },
                    "profile": {
                        "type": "string",
                        "enum": ["fast", "balanced", "deep"],
                        "description": (
                            "Optional named retrieval profile: 'fast' (bm25 only, zero "
                            "vector/model work), 'balanced' (adds module prefiltering), "
                            "or 'deep' (adds vector search and reranking). Omit to use "
                            "the repo's configured retrieval settings unchanged."
                        ),
                    },
                },
                "required": ["repo_url", "question"],
            },
        },
        {
            "name": "context",
            "description": (
                "Primary agent-facing context retrieval: query, intent, profile, "
                "filters, budgets, and handles as one contract. Returns a compact "
                "candidate map, exact fetch handles, selected code, relation paths, "
                "the route decision, a receipt, and a recommended next action. A "
                "thin facade over query_repo — query_repo and the other specialized "
                "tools remain fully supported."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local filesystem path or HTTP(S) Git URL of the repository."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": ("Natural-language question to answer from the codebase."),
                    },
                    "intent": {
                        "type": "string",
                        "enum": [intent.value for intent in QueryIntent],
                        "description": (
                            "Optional: pin the query intent instead of "
                            "auto-classifying it from the query text. Determines "
                            "the scoring-weight preset and default token budget."
                        ),
                    },
                    "profile": {
                        "type": "string",
                        "enum": [profile.value for profile in RetrievalProfile],
                        "description": (
                            "Optional named retrieval profile. Omit to use the "
                            "repo's configured retrieval settings unchanged."
                        ),
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional deterministic post-retrieval candidate "
                            "filters — never changes ranking or adds candidates."
                        ),
                        "properties": {
                            "include_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "fnmatch glob(s) a candidate's file path must match."
                                ),
                            },
                            "exclude_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "fnmatch glob(s) that exclude a candidate by file path."
                                ),
                            },
                            "languages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": ("Restrict returned candidates to these languages."),
                            },
                        },
                    },
                    "budgets": {
                        "type": "object",
                        "description": "Optional token-budget input.",
                        "properties": {
                            "token_budget": {
                                "type": "integer",
                                "description": (
                                    "Explicit token budget override. Omit to resolve "
                                    "from 'intent' or the query's own auto-scaling."
                                ),
                            },
                        },
                    },
                    "handles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional exact fetch handle(s) — bypasses broad search "
                            "and returns exactly these candidates."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "description": "Output format. Defaults to 'json'.",
                    },
                },
                "required": ["repo_url", "query"],
            },
        },
        {
            "name": "compare_repos",
            "description": (
                "Compare two code repositories across architectural dimensions such as "
                "API surface, error handling, concurrency model, testing, "
                "state management, and configuration."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_a": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the first repository.",
                    },
                    "repo_b": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the second repository.",
                    },
                    "dimensions": {
                        "type": "string",
                        "default": "api_surface,error_handling",
                        "description": (
                            "Comma-separated dimensions to compare. "
                            "Supported: api_surface, error_handling, concurrency, "
                            "testing, state_management, configuration."
                        ),
                    },
                },
                "required": ["repo_a", "repo_b"],
            },
        },
        {
            "name": "get_file_tree",
            "description": (
                "Return a hierarchical file tree for a repository, optionally filtered "
                "by language and depth."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the repository.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum directory depth to traverse.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter results to files of this language.",
                    },
                },
                "required": ["repo_url"],
            },
        },
        {
            "name": "get_file_outline",
            "description": (
                "Return a structural outline of a single file — symbols, classes, "
                "functions, and their locations."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the repository.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Relative path of the file within the repository.",
                    },
                },
                "required": ["repo_url", "file_path"],
            },
        },
        {
            "name": "search_symbols",
            "description": (
                "Search for symbols (functions, classes, variables) in a repository "
                "by name, kind, and/or language."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the repository.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query to match against symbol names.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Filter by symbol kind (e.g. function, class).",
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by programming language.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of results to return.",
                    },
                },
                "required": ["repo_url", "query"],
            },
        },
        {
            "name": "get_symbol",
            "description": "Retrieve a single symbol by its stable symbol ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the repository.",
                    },
                    "symbol_id": {
                        "type": "string",
                        "description": "Stable symbol identifier.",
                    },
                },
                "required": ["repo_url", "symbol_id"],
            },
        },
        {
            "name": "get_symbols_batch",
            "description": (
                "Retrieve multiple symbols by their stable symbol IDs in a single call. "
                "Maximum 50 IDs per request."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Local path or HTTP(S) URL of the repository.",
                    },
                    "symbol_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stable symbol identifiers (max 50).",
                    },
                },
                "required": ["repo_url", "symbol_ids"],
            },
        },
        {
            "name": "get_impact",
            "description": (
                "Deterministic blast-radius impact analysis for changed files: affected "
                "files, modules, public interfaces, test surface, risk assessment. Diffs "
                "against a base ref or an explicit changed-file list; pass 'diff' for "
                "per-symbol LOW/MEDIUM/HIGH risk tiers."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local repository path (git diff mode requires a local checkout)."
                        ),
                    },
                    "base": {
                        "type": "string",
                        "default": "main",
                        "description": (
                            "Base ref for git diff mode; ignored if changed_files or diff is set."
                        ),
                    },
                    "changed_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Explicit changed files, bypassing git diff mode. Mutually "
                            "exclusive with diff."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                        "description": "Output format for the impact report.",
                    },
                    "diff": {
                        "type": "string",
                        "description": (
                            "Resolve the diff (this ref vs. working tree) to touched symbols "
                            "with a per-symbol risk tier, added as affected_symbols. Mutually "
                            "exclusive with changed_files."
                        ),
                    },
                },
                "required": ["repo_url"],
            },
        },
        {
            "name": "explain_target",
            "description": (
                "Structural explain for a file, symbol, or module: public interfaces, "
                "internal symbols, imports/imported-by, module context, complexity signals."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local path or HTTP(S) URL of the repository. Required "
                            "unless graph_path is given."
                        ),
                    },
                    "target": {
                        "type": "string",
                        "description": (
                            "File path or `path::name#kind` symbol identifier to "
                            "explain. Mutually exclusive with module_name."
                        ),
                    },
                    "module_name": {
                        "type": "string",
                        "description": ("Module path to explain. Mutually exclusive with target."),
                    },
                    "graph_path": {
                        "type": "string",
                        "description": (
                            "Read an exported graph artifact instead of indexing repo_url."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                        "description": "Output format for the explain context.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_onboarding",
            "description": (
                "Generate a deterministic onboarding guide from graph/index data: "
                "repository overview, architecture modules, entry points, public "
                "interfaces, recommended reading order, complexity hotspots, test "
                "surface, and configuration surface. Markdown-only output."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": (
                            "Local path or HTTP(S) URL of the repository. Required "
                            "unless graph_path is given."
                        ),
                    },
                    "graph_path": {
                        "type": "string",
                        "description": (
                            "Read an exported graph artifact instead of indexing repo_url."
                        ),
                    },
                    "max_files": {
                        "type": "integer",
                        "default": 40,
                        "description": "Maximum paths per capped section.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "graph_lookup",
            "description": (
                "Look up nodes in an exported graph artifact (no reindexing). Exact "
                "ID/path matches win over fuzzy ones."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                    "node": {
                        "type": "string",
                        "description": "Node ID, path, label, or query.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                    "limit": {"type": "integer", "default": 25},
                    "hub_degree": {"type": "integer", "default": 50},
                    "token_budget": {
                        "type": "integer",
                        "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                        "description": "Maximum markdown content tokens to return.",
                    },
                },
                "required": ["graph_path", "node"],
            },
        },
        {
            "name": "graph_neighbors",
            "description": (
                "Graph neighbors for a node from an exported artifact (no reindexing). "
                "Edges carry kind, confidence, evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                    "node": {
                        "type": "string",
                        "description": "Node ID, path, label, or query.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                        "default": "both",
                    },
                    "depth": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 25},
                    "hub_degree": {"type": "integer", "default": 50},
                    "token_budget": {
                        "type": "integer",
                        "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                        "description": "Maximum markdown content tokens to return.",
                    },
                },
                "required": ["graph_path", "node"],
            },
        },
        {
            "name": "graph_path",
            "description": (
                "Shortest structural path between two nodes in an exported graph "
                "artifact (no reindexing)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                    "source": {"type": "string", "description": "Source node ID or path."},
                    "target": {"type": "string", "description": "Target node ID or path."},
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                        "default": "both",
                    },
                    "max_edges": {"type": "integer", "default": 100},
                    "hub_degree": {"type": "integer", "default": 50},
                    "token_budget": {
                        "type": "integer",
                        "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                        "description": "Maximum markdown content tokens to return.",
                    },
                },
                "required": ["graph_path", "source", "target"],
            },
        },
        {
            "name": "graph_stats",
            "description": (
                "Deterministic graph stats and hub summary from an exported artifact "
                "(no reindexing)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                    "hub_limit": {"type": "integer", "default": 10},
                    "hub_degree": {"type": "integer", "default": 50},
                    "token_budget": {
                        "type": "integer",
                        "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                        "description": "Maximum markdown content tokens to return.",
                    },
                },
                "required": ["graph_path"],
            },
        },
        {
            "name": "graph_hubs",
            "description": ("High-degree hubs from an exported graph artifact (no reindexing)."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                    "limit": {"type": "integer", "default": 25},
                    "threshold": {"type": "integer"},
                    "hub_degree": {"type": "integer", "default": 50},
                    "token_budget": {
                        "type": "integer",
                        "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                        "description": "Maximum markdown content tokens to return.",
                    },
                },
                "required": ["graph_path"],
            },
        },
    ]


#: Every MCP tool name archex registers, in registration order. Derived from
#: `_tool_schemas()` so it can never drift from the tools `build_server` exposes.
ALL_TOOL_NAMES: tuple[str, ...] = tuple(schema["name"] for schema in _tool_schemas())

_GRAPH_TOOL_NAMES: frozenset[str] = frozenset(
    {"graph_lookup", "graph_neighbors", "graph_path", "graph_stats", "graph_hubs"}
)

#: Named convenience tool-scope profiles for `archex mcp --tools` and
#: `install-client`/`setup --tool-scope`. "all" (every tool, the unscoped
#: default) is intentionally absent here -- `resolve_tool_scope` treats it,
#: `None`, and the empty string as the same "no filtering" case so callers
#: can test `is None` instead of special-casing a profile name.
TOOL_SCOPE_PROFILES: dict[str, frozenset[str]] = {
    "core": frozenset(ALL_TOOL_NAMES) - _GRAPH_TOOL_NAMES,
    "graph": _GRAPH_TOOL_NAMES,
}


def resolve_tool_scope(spec: str | None) -> frozenset[str] | None:
    """Resolve a `--tools`/`--tool-scope` spec into a set of tool names.

    `None`, the empty string, or `"all"` mean unscoped -- every registered
    tool, the current default behavior. Otherwise `spec` is either a named
    profile from `TOOL_SCOPE_PROFILES` or a comma-separated explicit
    allowlist of tool names.

    Raises:
        ValueError: `spec` names a tool that does not exist, so a typo
            fails fast instead of silently registering zero tools.
    """
    if spec is None or spec.strip() in ("", "all"):
        return None
    if spec in TOOL_SCOPE_PROFILES:
        return TOOL_SCOPE_PROFILES[spec]
    names = frozenset(name.strip() for name in spec.split(",") if name.strip())
    unknown = names - frozenset(ALL_TOOL_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown MCP tool name(s): {', '.join(sorted(unknown))}. "
            f"Known tools: {', '.join(ALL_TOOL_NAMES)}"
        )
    return names


def measure_tool_schema_size(tool_names: frozenset[str] | None = None) -> dict[str, Any]:
    """Serialized MCP tool-schema size for `tool_names` (`None` means every tool).

    Serializes each tool's `{name, description, inputSchema}` as compact,
    sort-keyed JSON -- the same shape `list_tools()` advertises -- so the
    reported byte counts track what a client actually registers.
    """
    schemas = _tool_schemas()
    if tool_names is not None:
        schemas = [schema for schema in schemas if schema["name"] in tool_names]
    per_tool_chars = {schema["name"]: len(json.dumps(schema, sort_keys=True)) for schema in schemas}
    return {
        "tool_count": len(schemas),
        "total_chars": sum(per_tool_chars.values()),
        "per_tool_chars": per_tool_chars,
    }


def build_server(
    runtime: QueryRuntime | None = None, tool_names: frozenset[str] | None = None
) -> Any:
    """Build and return a configured MCP Server instance.

    `tool_names`, when given, scopes the tools this server advertises via
    `list_tools()` to exactly that subset (see `resolve_tool_scope`). Every
    tool name still dispatches through `call_tool` regardless of scoping --
    scoping only shrinks the advertised schema surface, it never changes
    which tool names a client can successfully call.

    Raises:
        ImportError: If the `mcp` package is not installed.
    """
    try:
        import mcp.types as mcp_types
        from mcp.server import Server
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. Install it with: uv add mcp"
        ) from exc

    if runtime is None:
        runtime = QueryRuntime()

    server: Server[None, Any] = Server("archex")  # type: ignore[type-arg]

    @server.list_tools()  # pyright: ignore[reportUnusedFunction]
    async def list_tools() -> list[mcp_types.Tool]:  # pyright: ignore[reportUnusedFunction]
        schemas = _tool_schemas()
        if tool_names is not None:
            schemas = [schema for schema in schemas if schema["name"] in tool_names]
        return [mcp_types.Tool(**schema) for schema in schemas]

    @server.call_tool()  # pyright: ignore[reportUnusedFunction]
    async def call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str,
        arguments: dict[str, Any],
    ) -> list[mcp_types.TextContent]:
        loop = asyncio.get_running_loop()
        result_text = await _run_mcp_tool(loop, name, arguments, runtime)

        return [mcp_types.TextContent(type="text", text=result_text)]

    return server


def _ignored_watch_path(path: str) -> bool:
    parts = Path(path).parts
    return any(part in {".git", ".archex", "__pycache__", ".pytest_cache"} for part in parts)


def _start_index_watch(repo_path: Path, debounce_ms: int) -> Any:
    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise ImportError("watch mode requires the 'watchdog' package") from exc

    class DebouncedIndexHandler(FileSystemEventHandler):
        def __init__(self) -> None:
            self._timer: threading.Timer | None = None
            self._lock = threading.Lock()
            self._pending_refresh = False
            self._refreshing = False

        def on_any_event(self, event: FileSystemEvent) -> None:
            src_path = str(event.src_path)
            if event.is_directory or _ignored_watch_path(src_path):
                return
            self._schedule()

        def _schedule(self) -> None:
            with self._lock:
                if self._refreshing:
                    self._pending_refresh = True
                    return
                self._arm_timer_locked()

        def _arm_timer_locked(self) -> None:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(debounce_ms / 1000.0, self._refresh)
            self._timer.daemon = True
            self._timer.start()

        def _refresh(self) -> None:
            with self._lock:
                if self._refreshing:
                    return
                self._refreshing = True
            try:
                source = RepoSource(local_path=str(repo_path))
                timing = PipelineTiming()
                store = index_repository(source, timing=timing)
                try:
                    logger.info("MCP watch refreshed %s via %s", repo_path, timing.strategy)
                finally:
                    store.close()
            finally:
                with self._lock:
                    self._refreshing = False
                    if self._pending_refresh:
                        self._pending_refresh = False
                        self._arm_timer_locked()

    observer = Observer()
    observer.schedule(DebouncedIndexHandler(), str(repo_path), recursive=True)
    observer.start()
    logger.info("MCP watch enabled for %s", repo_path)
    return observer


async def run_stdio_server(
    *,
    watch: bool = False,
    watch_path: str = ".",
    watch_debounce_ms: int = 300,
    tool_names: frozenset[str] | None = None,
) -> None:
    """Run the archex MCP server over stdio."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. Install it with: uv add mcp"
        ) from exc

    observer: Any | None = None
    if watch:
        observer = _start_index_watch(Path(watch_path).expanduser().resolve(), watch_debounce_ms)

    # One QueryRuntime lives for this server process's whole lifetime, shared
    # across every query_repo call so repeat warm queries against the same
    # generation skip re-hydrating from SQLite.
    runtime = QueryRuntime()
    server = build_server(runtime=runtime, tool_names=tool_names)
    try:
        async with stdio_server() as (read_stream, write_stream):
            init_opts = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_opts, raise_exceptions=True)
    finally:
        runtime.close()
        if observer is not None:
            observer.stop()
            observer.join(timeout=5)
