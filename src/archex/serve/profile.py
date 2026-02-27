"""ArchProfile assembly: combine analysis outputs into a final ArchProfile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.models import (
    ArchDecision,
    ArchProfile,
    CodebaseStats,
    DependencyGraphSummary,
    DetectedPattern,
    Interface,
    LanguageStats,
    Module,
    ParsedFile,
    RepoMetadata,
    SymbolKind,
    SymbolRef,
    Visibility,
)

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph


def _compute_stats(parsed_files: list[ParsedFile]) -> CodebaseStats:
    """Aggregate file, line, and symbol counts per language."""
    lang_files: dict[str, int] = {}
    lang_lines: dict[str, int] = {}
    lang_symbols: dict[str, int] = {}
    total_files = 0
    total_lines = 0
    total_symbols = 0

    for pf in parsed_files:
        lang = pf.language
        lang_files[lang] = lang_files.get(lang, 0) + 1
        lang_lines[lang] = lang_lines.get(lang, 0) + pf.lines
        sym_count = len(pf.symbols)
        lang_symbols[lang] = lang_symbols.get(lang, 0) + sym_count
        total_files += 1
        total_lines += pf.lines
        total_symbols += sym_count

    languages: dict[str, LanguageStats] = {}
    for lang in lang_files:
        percentage = (lang_files[lang] / total_files * 100.0) if total_files > 0 else 0.0
        languages[lang] = LanguageStats(
            files=lang_files[lang],
            lines=lang_lines.get(lang, 0),
            symbols=lang_symbols.get(lang, 0),
            percentage=round(percentage, 2),
        )

    return CodebaseStats(
        total_files=total_files,
        total_lines=total_lines,
        languages=languages,
        symbol_count=total_symbols,
    )


def _extract_interfaces(parsed_files: list[ParsedFile]) -> list[Interface]:
    """Extract public functions and classes as the interface surface."""
    interfaces: list[Interface] = []
    public_kinds = {SymbolKind.FUNCTION, SymbolKind.CLASS, SymbolKind.METHOD}

    for pf in parsed_files:
        for sym in pf.symbols:
            if sym.visibility != Visibility.PUBLIC:
                continue
            if sym.kind not in public_kinds:
                continue

            ref = SymbolRef(
                name=sym.name,
                qualified_name=sym.qualified_name,
                file_path=sym.file_path,
                kind=sym.kind,
            )
            signature = sym.signature or sym.name
            interfaces.append(
                Interface(
                    symbol=ref,
                    signature=signature,
                    docstring=sym.docstring,
                )
            )

    return interfaces


def build_profile(
    repo_metadata: RepoMetadata,
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,
    modules: list[Module] | None = None,
    patterns: list[DetectedPattern] | None = None,
    interfaces: list[Interface] | None = None,
    decisions: list[ArchDecision] | None = None,
) -> ArchProfile:
    """Build an ArchProfile from repo metadata, parsed files, dependency graph, and analysis."""
    stats = _compute_stats(parsed_files)

    # Use provided interfaces or fall back to basic extraction
    iface_surface = interfaces if interfaces is not None else _extract_interfaces(parsed_files)

    dep_summary = DependencyGraphSummary(
        nodes=graph.file_count + graph.symbol_count,
        edges=graph.file_edge_count,
        file_count=graph.file_count,
        symbol_count=graph.symbol_count,
    )

    stats.internal_edge_count = graph.file_edge_count
    stats.module_count = len(modules) if modules else 0

    return ArchProfile(
        repo=repo_metadata,
        stats=stats,
        module_map=modules or [],
        pattern_catalog=patterns or [],
        interface_surface=iface_surface,
        decision_log=decisions or [],
        dependency_graph=dep_summary,
    )
