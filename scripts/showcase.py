"""archex end-to-end showcase — exercises every public API against a real repo."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import archex
from archex.api import (
    analyze,
    compare,
    file_outline,
    file_tree,
    get_file_token_count,
    get_files_token_count,
    get_repo_total_tokens,
    get_symbol,
    get_symbols_batch,
    query,
    search_symbols,
)
from archex.models import (
    ArchProfile,
    ComparisonResult,
    ContextBundle,
    FileOutline,
    FileTree,
    FileTreeEntry,
    RepoSource,
    SymbolMatch,
    SymbolSource,
)
from archex.reporting import count_tokens

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

W = 64  # output width


def banner(source: str) -> None:
    bar = "═" * W
    print(bar)
    print(f"  archex showcase — v{archex.__version__}")
    print(f"  Source: {source}")
    print(bar)
    print()


def section(num: int, title: str, elapsed: float) -> None:
    tag = f" {num}. {title} "
    trail = f" {elapsed:.2f}s "
    dashes = "─" * max(1, W - len(tag) - len(trail) - 2)
    print(f"─── {tag}{dashes}{trail}───")


def indent(text: str, prefix: str = "  ") -> None:
    for line in text.splitlines():
        print(f"{prefix}{line}")


def summary_table(timings: list[tuple[str, float]]) -> None:
    print()
    print(f"─── Summary {'─' * (W - 14)}───")
    print(f"  {'Operation':<32} {'Time':>8}")
    print(f"  {'─' * 32} {'─' * 8}")
    total = 0.0
    for name, elapsed in timings:
        print(f"  {name:<32} {elapsed:>7.2f}s")
        total += elapsed
    print(f"  {'─' * 32} {'─' * 8}")
    print(f"  {'Total':<32} {total:>7.2f}s")
    print("═" * W)


# ---------------------------------------------------------------------------
# Results collector
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyRow:
    """One row in the token efficiency comparison table."""

    operation: str
    raw_tokens: int
    archex_tokens: int
    savings_pct: float


@dataclass
class ShowcaseResults:
    """Collects all intermediate results for the insights section."""

    # Token efficiency rows
    efficiency: list[EfficiencyRow] = field(default_factory=lambda: [])

    # Section 1: file_tree
    tree: FileTree | None = None

    # Section 2: analyze
    profile: ArchProfile | None = None

    # Section 3: token budget
    repo_tokens: int = 0
    single_file_path: str = ""
    single_file_tokens: int = 0
    batch_file_count: int = 0
    batch_tokens: int = 0

    # Section 4: file_outline
    outline: FileOutline | None = None

    # Section 5-7: symbols
    search_matches: list[SymbolMatch] = field(default_factory=lambda: [])
    retrieved_symbol: SymbolSource | None = None
    batch_results: list[SymbolSource | None] = field(default_factory=lambda: [])

    # Section 8-9: query
    cold_bundle: ContextBundle | None = None
    warm_bundle: ContextBundle | None = None
    cold_elapsed: float = 0.0
    warm_elapsed: float = 0.0

    # Section 10: output formats
    xml_chars: int = 0
    json_chars: int = 0
    markdown_chars: int = 0

    # Section 11: compare
    comparison: ComparisonResult | None = None

    # Token budget used
    token_budget: int = 8192


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _first_py_file(entries: list[FileTreeEntry]) -> str | None:
    for entry in entries:
        if not entry.is_directory and entry.language == "python":
            return entry.path
        if entry.children:
            result = _first_py_file(entry.children)
            if result:
                return result
    return None


def _first_non_dir(entries: list[FileTreeEntry]) -> str | None:
    for entry in entries:
        if not entry.is_directory:
            return entry.path
        if entry.children:
            result = _first_non_dir(entry.children)
            if result:
                return result
    return None


def _collect_file_paths(entries: list[FileTreeEntry], paths: list[str]) -> None:
    for entry in entries:
        if not entry.is_directory:
            paths.append(entry.path)
        if entry.children:
            _collect_file_paths(entry.children, paths)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def run_file_tree(source: RepoSource, r: ShowcaseResults) -> float:
    t0 = time.perf_counter()
    tree = file_tree(source)
    r.repo_tokens = get_repo_total_tokens(source)
    elapsed = time.perf_counter() - t0
    r.tree = tree
    section(1, "Repository Overview", elapsed)
    indent(f"Files: {tree.total_files} | Languages: {len(tree.languages)}")
    for lang, count in sorted(tree.languages.items(), key=lambda x: -x[1]):
        indent(f"  {lang}: {count} files")
    print()

    archex_tok = count_tokens(tree.model_dump_json())
    if r.repo_tokens > 0:
        savings = (1 - archex_tok / r.repo_tokens) * 100
        r.efficiency.append(EfficiencyRow("file_tree()", r.repo_tokens, archex_tok, savings))

    return elapsed


def run_analyze(source: RepoSource, r: ShowcaseResults) -> float:
    t0 = time.perf_counter()
    profile = analyze(source)
    elapsed = time.perf_counter() - t0
    r.profile = profile
    section(2, "Architectural Analysis", elapsed)
    indent(f"Modules: {len(profile.module_map)}")
    for mod in profile.module_map[:5]:
        indent(f"  {mod.name} ({mod.file_count} files, cohesion {mod.cohesion_score:.2f})")
    if len(profile.module_map) > 5:
        indent(f"  ... and {len(profile.module_map) - 5} more")
    indent(f"Patterns: {len(profile.pattern_catalog)}")
    for pat in profile.pattern_catalog[:5]:
        indent(f"  {pat.display_name} [{pat.category}] {pat.confidence:.0%}")
    indent(f"Interfaces: {len(profile.interface_surface)}")
    indent(
        f"Stats: {profile.stats.total_files} files, {profile.stats.total_lines} lines, "
        f"{profile.stats.symbol_count} symbols"
    )
    print()

    archex_tok = count_tokens(profile.to_markdown())
    if r.repo_tokens > 0:
        savings = (1 - archex_tok / r.repo_tokens) * 100
        r.efficiency.append(EfficiencyRow("analyze()", r.repo_tokens, archex_tok, savings))

    return elapsed


def run_token_budget(source: RepoSource, r: ShowcaseResults) -> float:
    tree = r.tree
    if tree is None:
        section(3, "Token Budget", 0.0)
        indent("(no tree available)")
        print()
        return 0.0

    t0 = time.perf_counter()

    if r.repo_tokens == 0:
        r.repo_tokens = get_repo_total_tokens(source)

    file_paths: list[str] = []
    _collect_file_paths(tree.entries, file_paths)
    single_file = file_paths[0] if file_paths else None
    if single_file:
        r.single_file_path = single_file
        r.single_file_tokens = get_file_token_count(source, single_file)

    batch_files = file_paths[:5]
    if batch_files:
        r.batch_file_count = len(batch_files)
        r.batch_tokens = get_files_token_count(source, batch_files)

    elapsed = time.perf_counter() - t0
    section(3, "Token Budget", elapsed)
    indent(f"Total repo tokens: {r.repo_tokens:,}")
    if single_file:
        indent(f"Single file ({single_file}): {r.single_file_tokens:,} tokens")
    indent(f"Batch ({r.batch_file_count} files): {r.batch_tokens:,} tokens")
    print()
    return elapsed


def run_file_outline(source: RepoSource, r: ShowcaseResults) -> float:
    tree = r.tree
    if tree is None:
        section(4, "File Outline", 0.0)
        indent("(no tree available)")
        print()
        return 0.0

    target = _first_py_file(tree.entries) or _first_non_dir(tree.entries)
    if not target:
        section(4, "File Outline", 0.0)
        indent("(no files found)")
        print()
        return 0.0

    t0 = time.perf_counter()
    outline = file_outline(source, target)
    elapsed = time.perf_counter() - t0
    r.outline = outline
    section(4, "File Outline", elapsed)
    indent(
        f"File: {outline.file_path} ({outline.language}, {outline.lines} lines, "
        f"{outline.token_count_raw} tokens)"
    )
    for sym in outline.symbols[:10]:
        children_tag = f" [{len(sym.children)} children]" if sym.children else ""
        indent(f"  {sym.kind}: {sym.name}{children_tag}")
    if len(outline.symbols) > 10:
        indent(f"  ... and {len(outline.symbols) - 10} more symbols")
    print()

    raw_tok = outline.token_count_raw
    archex_tok = count_tokens(outline.model_dump_json())
    if raw_tok > 0:
        savings = (1 - archex_tok / raw_tok) * 100
        r.efficiency.append(EfficiencyRow("file_outline()", raw_tok, archex_tok, savings))

    return elapsed


def run_search_symbols(source: RepoSource, r: ShowcaseResults) -> float:
    t0 = time.perf_counter()
    results = search_symbols(source, "config", kind="function", limit=10)
    elapsed = time.perf_counter() - t0
    r.search_matches = results
    section(5, "Symbol Search", elapsed)
    indent(f"Query: 'config' (kind=function) → {len(results)} results")
    for m in results[:5]:
        sig = f" — {m.signature}" if m.signature else ""
        indent(f"  {m.name} ({m.file_path}:{m.start_line}){sig}")
    if len(results) > 5:
        indent(f"  ... and {len(results) - 5} more")
    print()

    if results:
        unique_files = list({m.file_path for m in results})
        raw_tok = get_files_token_count(source, unique_files)
        archex_tok = count_tokens(json.dumps([m.model_dump() for m in results]))
        if raw_tok > 0:
            savings = (1 - archex_tok / raw_tok) * 100
            r.efficiency.append(EfficiencyRow("search_symbols()", raw_tok, archex_tok, savings))

    return elapsed


def run_get_symbol(source: RepoSource, r: ShowcaseResults) -> float:
    if not r.search_matches:
        section(6, "Symbol Retrieval", 0.0)
        indent("(no symbols to retrieve)")
        print()
        return 0.0

    target_id = r.search_matches[0].symbol_id
    t0 = time.perf_counter()
    sym = get_symbol(source, target_id)
    elapsed = time.perf_counter() - t0
    r.retrieved_symbol = sym
    section(6, "Symbol Retrieval", elapsed)
    if sym:
        indent(f"Symbol: {sym.name} ({sym.kind})")
        indent(f"File: {sym.file_path}:{sym.start_line}-{sym.end_line}")
        indent(f"Tokens: {sym.token_count}")
        source_lines = sym.source.splitlines()
        preview = source_lines[:5]
        for line in preview:
            indent(f"  │ {line}")
        if len(source_lines) > 5:
            indent(f"  │ ... ({len(source_lines) - 5} more lines)")

        raw_tok = get_file_token_count(source, sym.file_path)
        archex_tok = sym.token_count
        if raw_tok > 0:
            savings = (1 - archex_tok / raw_tok) * 100
            r.efficiency.append(EfficiencyRow("get_symbol()", raw_tok, archex_tok, savings))
    else:
        indent(f"Symbol {target_id} not found")
    print()
    return elapsed


def run_batch_symbols(source: RepoSource, r: ShowcaseResults) -> float:
    ids = [m.symbol_id for m in r.search_matches[:3]]
    if not ids:
        section(7, "Batch Retrieval", 0.0)
        indent("(no symbols to batch retrieve)")
        print()
        return 0.0

    t0 = time.perf_counter()
    results = get_symbols_batch(source, ids)
    elapsed = time.perf_counter() - t0
    r.batch_results = results
    section(7, "Batch Retrieval", elapsed)
    indent(f"Requested: {len(ids)} symbols")
    found_syms = [x for x in results if x is not None]
    indent(f"Found: {len(found_syms)}/{len(ids)}")
    for x in found_syms:
        indent(f"  {x.name} ({x.kind}, {x.token_count} tokens)")
    print()

    if found_syms:
        unique_files = list({s.file_path for s in found_syms})
        raw_tok = get_files_token_count(source, unique_files)
        archex_tok = sum(s.token_count for s in found_syms)
        if raw_tok > 0:
            savings = (1 - archex_tok / raw_tok) * 100
            r.efficiency.append(EfficiencyRow("get_symbols_batch()", raw_tok, archex_tok, savings))

    return elapsed


def run_query_cold(source: RepoSource, budget: int, r: ShowcaseResults) -> float:
    question = "What are the main entry points and how is the codebase structured?"
    t0 = time.perf_counter()
    bundle = query(source, question, token_budget=budget)
    elapsed = time.perf_counter() - t0
    r.cold_bundle = bundle
    r.cold_elapsed = elapsed
    section(8, "Query (cold)", elapsed)
    indent(f"Question: {question!r}")
    indent(f"Chunks: {len(bundle.chunks)} | Tokens: {bundle.token_count:,}/{bundle.token_budget:,}")
    fill = (bundle.token_count / bundle.token_budget * 100) if bundle.token_budget else 0
    indent(f"Fill rate: {fill:.1f}% | Truncated: {bundle.truncated}")
    unique_files = {rc.chunk.file_path for rc in bundle.chunks}
    indent(f"Unique files: {len(unique_files)}")
    for rc in bundle.chunks[:3]:
        indent(f"  {rc.chunk.file_path}:{rc.chunk.start_line} (score {rc.final_score:.3f})")
    if len(bundle.chunks) > 3:
        indent(f"  ... and {len(bundle.chunks) - 3} more chunks")
    print()

    if unique_files:
        raw_tok = get_files_token_count(source, list(unique_files))
        archex_tok = bundle.token_count
        if raw_tok > 0:
            savings = (1 - archex_tok / raw_tok) * 100
            r.efficiency.append(EfficiencyRow("query()", raw_tok, archex_tok, savings))

    return elapsed


def run_query_warm(source: RepoSource, budget: int, r: ShowcaseResults) -> float:
    question = "How does error handling work across the codebase?"
    t0 = time.perf_counter()
    bundle = query(source, question, token_budget=budget)
    elapsed = time.perf_counter() - t0
    r.warm_bundle = bundle
    r.warm_elapsed = elapsed
    section(9, "Query (warm)", elapsed)
    indent(f"Question: {question!r}")
    indent(f"Chunks: {len(bundle.chunks)} | Tokens: {bundle.token_count:,}/{bundle.token_budget:,}")
    fill = (bundle.token_count / bundle.token_budget * 100) if bundle.token_budget else 0
    indent(f"Fill rate: {fill:.1f}% | Truncated: {bundle.truncated}")
    print()
    return elapsed


def run_output_formats(r: ShowcaseResults) -> float:
    bundle = r.warm_bundle or r.cold_bundle
    if not bundle:
        section(10, "Output Formats", 0.0)
        indent("(no query bundle available)")
        print()
        return 0.0

    t0 = time.perf_counter()
    xml_out = bundle.to_prompt(format="xml")
    json_out = bundle.to_prompt(format="json")
    md_out = bundle.to_prompt(format="markdown")
    elapsed = time.perf_counter() - t0
    r.xml_chars = len(xml_out)
    r.json_chars = len(json_out)
    r.markdown_chars = len(md_out)

    section(10, "Output Formats", elapsed)
    indent(f"XML:      {r.xml_chars:>8,} chars")
    indent(f"JSON:     {r.json_chars:>8,} chars")
    indent(f"Markdown: {r.markdown_chars:>8,} chars")
    print()
    return elapsed


def run_compare(source: RepoSource, r: ShowcaseResults) -> float:
    t0 = time.perf_counter()
    result = compare(source, source)
    elapsed = time.perf_counter() - t0
    r.comparison = result
    section(11, "Cross-Repo Compare (self)", elapsed)
    indent(f"Dimensions: {len(result.dimensions)}")
    for dim in result.dimensions[:5]:
        indent(f"  {dim.dimension}")
        if dim.trade_offs:
            indent(f"    Trade-off: {dim.trade_offs[0]}")
    if len(result.dimensions) > 5:
        indent(f"  ... and {len(result.dimensions) - 5} more dimensions")
    print()

    raw_tok = r.repo_tokens * 2  # self-compare: both repos are the same
    archex_tok = count_tokens(result.model_dump_json())
    if raw_tok > 0:
        savings = (1 - archex_tok / raw_tok) * 100
        r.efficiency.append(EfficiencyRow("compare()", raw_tok, archex_tok, savings))

    return elapsed


# ---------------------------------------------------------------------------
# Section 12: Insights
# ---------------------------------------------------------------------------


def _size_label(total_files: int, total_lines: int) -> str:
    if total_files > 500 or total_lines > 50_000:
        return "large"
    if total_files > 100 or total_lines > 10_000:
        return "medium"
    return "small"


def _cohesion_label(avg: float) -> str:
    if avg >= 0.8:
        return "high (tightly coupled modules)"
    if avg >= 0.5:
        return "moderate"
    return "low (loosely coupled, well-decomposed)"


def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(value / max_val * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _render_efficiency_table(rows: list[EfficiencyRow]) -> None:
    """Render the token efficiency comparison table."""
    print()
    indent("TOKEN EFFICIENCY — Raw Files vs archex")
    indent("─" * 56)
    indent(f"{'Operation':<22} {'Raw Tokens':>10}  {'archex Tokens':>13}  {'Savings':>7}")
    indent(f"{'─' * 22} {'─' * 10}  {'─' * 13}  {'─' * 7}")
    total_raw = 0
    total_archex = 0
    for row in rows:
        total_raw += row.raw_tokens
        total_archex += row.archex_tokens
        indent(
            f"{row.operation:<22} {row.raw_tokens:>10,}  {row.archex_tokens:>13,}  "
            f"{row.savings_pct:>6.1f}%"
        )
    indent(f"{'─' * 22} {'─' * 10}  {'─' * 13}  {'─' * 7}")
    total_savings = (1 - total_archex / total_raw) * 100 if total_raw > 0 else 0.0
    indent(f"{'Total':<22} {total_raw:>10,}  {total_archex:>13,}  {total_savings:>6.1f}%")
    print()

    negative = [row for row in rows if row.savings_pct < 0]
    if negative:
        for row in negative:
            indent(
                f"* {row.operation}: negative savings — raw file too small "
                f"({row.raw_tokens} tokens) for structured output to compress"
            )
    print()
    indent(
        f"archex delivers the same information in {total_archex:,} tokens\n"
        f"  that raw file dumping would cost {total_raw:,} tokens."
    )
    print()


def run_insights(r: ShowcaseResults) -> None:
    print()
    print(f"─── 12. Insights {'─' * (W - 19)}───")

    # --- Part A: Efficiency Table ---
    if r.efficiency:
        _render_efficiency_table(r.efficiency)

    profile = r.profile
    tree = r.tree

    # --- Part B: Existing insights ---

    # --- Codebase Profile ---
    indent("CODEBASE PROFILE")
    indent("─" * 40)
    if profile and tree:
        stats = profile.stats
        size = _size_label(stats.total_files, stats.total_lines)
        symbol_density = stats.symbol_count / stats.total_files if stats.total_files else 0
        lines_per_file = stats.total_lines / stats.total_files if stats.total_files else 0

        indent(f"Scale:          {size} ({stats.total_files} files, {stats.total_lines:,} lines)")
        indent(f"Languages:      {len(tree.languages)}")
        indent(f"Symbol density: {symbol_density:.1f} symbols/file")
        indent(f"Avg file size:  {lines_per_file:.0f} lines/file")
        print()

        # Language breakdown bar chart
        if tree.languages:
            indent("LANGUAGE DISTRIBUTION")
            indent("─" * 40)
            total_lang_files = sum(tree.languages.values())
            for lang, count in sorted(tree.languages.items(), key=lambda x: -x[1]):
                pct = count / total_lang_files * 100 if total_lang_files else 0
                bar = _bar(count, total_lang_files)
                indent(f"  {lang:<12} {bar} {count:>4} ({pct:.0f}%)")
            print()
    else:
        indent("  (analysis data unavailable)")
        print()

    # --- Architecture Quality ---
    if profile:
        indent("ARCHITECTURE QUALITY")
        indent("─" * 40)

        module_count = len(profile.module_map)
        pattern_count = len(profile.pattern_catalog)
        interface_count = len(profile.interface_surface)
        stats = profile.stats

        # Module analysis
        if profile.module_map:
            multi_file_modules = [m for m in profile.module_map if m.file_count > 1]
            single_file_modules = module_count - len(multi_file_modules)
            avg_cohesion = sum(m.cohesion_score for m in profile.module_map) / module_count
            avg_files_per_module = stats.total_files / module_count if module_count else 0

            indent(f"Modules:        {module_count} total")
            indent(f"  Multi-file:   {len(multi_file_modules)}")
            indent(f"  Single-file:  {single_file_modules}")
            indent(f"  Avg files/mod:{avg_files_per_module:.1f}")
            indent(f"  Avg cohesion: {avg_cohesion:.2f} — {_cohesion_label(avg_cohesion)}")

            if module_count > 0:
                granularity_ratio = module_count / stats.total_files if stats.total_files else 0
                if granularity_ratio > 0.8:
                    indent("  ⚠ Module-per-file: module detection found minimal clustering")
                elif granularity_ratio < 0.1:
                    indent("  ✓ Well-clustered: files are grouped into meaningful modules")
                else:
                    indent("  ~ Moderate clustering")
        else:
            indent("Modules:        0 (no module structure detected)")

        # Patterns
        if pattern_count > 0:
            high_conf = [p for p in profile.pattern_catalog if p.confidence >= 0.7]
            low_conf = [p for p in profile.pattern_catalog if p.confidence < 0.7]
            indent(f"Patterns:       {pattern_count} detected")
            if high_conf:
                names = ", ".join(f"{p.display_name} ({p.confidence:.0%})" for p in high_conf)
                indent(f"  High conf:    {names}")
            if low_conf:
                names = ", ".join(f"{p.display_name} ({p.confidence:.0%})" for p in low_conf)
                indent(f"  Low conf:     {names}")
        else:
            indent("Patterns:       none detected")

        # Interface density
        if stats.symbol_count > 0:
            iface_ratio = interface_count / stats.symbol_count * 100
            indent(
                f"Interfaces:     {interface_count} ({iface_ratio:.0f}% of symbols are public API)"
            )
        print()

    # --- Token Budget ---
    if r.repo_tokens > 0:
        indent("TOKEN BUDGET")
        indent("─" * 40)
        budget = r.token_budget
        coverage_pct = budget / r.repo_tokens * 100
        queries_for_full = r.repo_tokens / budget if budget else 0

        indent(f"Total tokens:   {r.repo_tokens:,}")
        indent(f"Query budget:   {budget:,}")
        indent(f"Coverage/query: {coverage_pct:.1f}% of repo per query")
        indent(f"Full coverage:  ~{queries_for_full:.0f} queries needed")

        # Fill rate analysis
        cold = r.cold_bundle
        warm = r.warm_bundle
        if cold:
            cold_fill = cold.token_count / cold.token_budget * 100 if cold.token_budget else 0
            indent(f"Cold fill rate: {cold_fill:.1f}% ({cold.token_count:,}/{cold.token_budget:,})")
        if warm:
            warm_fill = warm.token_count / warm.token_budget * 100 if warm.token_budget else 0
            indent(f"Warm fill rate: {warm_fill:.1f}% ({warm.token_count:,}/{warm.token_budget:,})")

        # Output format comparison
        if r.xml_chars and r.json_chars and r.markdown_chars:
            smallest = min(r.xml_chars, r.json_chars, r.markdown_chars)
            largest = max(r.xml_chars, r.json_chars, r.markdown_chars)
            overhead = (largest - smallest) / smallest * 100
            if smallest == r.xml_chars:
                best = "XML"
            elif smallest == r.markdown_chars:
                best = "Markdown"
            else:
                best = "JSON"
            indent(f"Most compact:   {best} ({smallest:,} chars)")
            indent(f"Format spread:  {overhead:.0f}% overhead (largest vs smallest)")
        print()

    # --- Performance Profile ---
    indent("PERFORMANCE PROFILE")
    indent("─" * 40)
    cold_t = r.cold_elapsed
    warm_t = r.warm_elapsed
    if cold_t > 0 and warm_t > 0:
        speedup = cold_t / warm_t
        indent(f"Cold query:     {cold_t:.2f}s")
        indent(f"Warm query:     {warm_t:.2f}s")
        indent(f"Cache speedup:  {speedup:.1f}×")
    elif cold_t > 0:
        indent(f"Cold query:     {cold_t:.2f}s")
        indent("Warm query:     (not available)")

    # Chunk diversity analysis
    cold = r.cold_bundle
    if cold and cold.chunks:
        unique_files = len({rc.chunk.file_path for rc in cold.chunks})
        score_spread = cold.chunks[0].final_score - cold.chunks[-1].final_score
        indent(f"Chunk count:    {len(cold.chunks)}")
        indent(f"Unique files:   {unique_files}")
        first = cold.chunks[0].final_score
        last = cold.chunks[-1].final_score
        indent(f"Score spread:   {first:.3f} → {last:.3f} (Δ{score_spread:.3f})")

    # Symbol search effectiveness
    if r.search_matches:
        indent(f"Symbol search:  {len(r.search_matches)} matches for 'config'")
    else:
        indent("Symbol search:  no matches (search term may not exist in this repo)")
    print()

    # --- Comparison Sanity ---
    if r.comparison:
        indent("COMPARISON SANITY CHECK")
        indent("─" * 40)
        dims = r.comparison.dimensions
        indent(f"Dimensions:     {len(dims)}")
        all_symmetric = all(d.repo_a_approach == d.repo_b_approach for d in dims)
        if all_symmetric:
            indent("Self-compare:   ✓ symmetric (all dimensions match)")
        else:
            mismatches = [d.dimension for d in dims if d.repo_a_approach != d.repo_b_approach]
            indent(f"Self-compare:   ⚠ asymmetric on {len(mismatches)} dimensions")
            for name in mismatches[:3]:
                indent(f"  - {name}")
        print()

    print("═" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="archex end-to-end showcase — exercises every public API.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=".",
        help="Local path or Git URL (default: .)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=8192,
        help="Token budget for queries (default: 8192)",
    )
    args = parser.parse_args()

    raw: str = args.source
    if raw.startswith("http://") or raw.startswith("https://"):
        source = RepoSource(url=raw)
    else:
        source = RepoSource(local_path=raw)

    banner(raw)
    r = ShowcaseResults(token_budget=args.budget)
    timings: list[tuple[str, float]] = []

    # 1. file_tree — needed for downstream sections
    timings.append(("file_tree()", run_file_tree(source, r)))

    # 2. analyze
    timings.append(("analyze()", run_analyze(source, r)))

    # 3. Token budget
    try:
        timings.append(("token counting", run_token_budget(source, r)))
    except Exception as exc:
        section(3, "Token Budget", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 4. File outline
    try:
        timings.append(("file_outline()", run_file_outline(source, r)))
    except Exception as exc:
        section(4, "File Outline", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 5. Symbol search
    try:
        timings.append(("search_symbols()", run_search_symbols(source, r)))
    except Exception as exc:
        section(5, "Symbol Search", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 6. Get symbol
    try:
        timings.append(("get_symbol()", run_get_symbol(source, r)))
    except Exception as exc:
        section(6, "Symbol Retrieval", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 7. Batch symbols
    try:
        timings.append(("get_symbols_batch()", run_batch_symbols(source, r)))
    except Exception as exc:
        section(7, "Batch Retrieval", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 8. Query cold
    try:
        timings.append(("query() cold", run_query_cold(source, args.budget, r)))
    except Exception as exc:
        section(8, "Query (cold)", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 9. Query warm
    try:
        timings.append(("query() warm", run_query_warm(source, args.budget, r)))
    except Exception as exc:
        section(9, "Query (warm)", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 10. Output formats
    try:
        timings.append(("to_prompt() ×3", run_output_formats(r)))
    except Exception as exc:
        section(10, "Output Formats", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # 11. Compare (self-compare)
    try:
        timings.append(("compare()", run_compare(source, r)))
    except Exception as exc:
        section(11, "Cross-Repo Compare (self)", 0.0)
        indent(f"FAILED: {exc}")
        print()

    # Summary table
    summary_table(timings)

    # 12. Insights
    run_insights(r)


if __name__ == "__main__":
    main()
