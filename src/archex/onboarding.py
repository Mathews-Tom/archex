"""Deterministic onboarding guide rendering from graph artifacts.

Two profiles project the same `ArchGraph`. `full` is the historical guide and is
byte-for-byte unchanged. `compact` is an opt-in strict-budget orientation view:
it trades per-item enumeration for adaptive directory clusters, the existing
graph-degree hub ranking, and an explicit receipt of what the budget dropped.
Neither profile parses a repository or computes a second architecture map.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from archex.graph_artifact import ArchGraph, GraphNode, GraphNodeType
from archex.graph_query import GraphQuery
from archex.reporting import count_tokens

FULL_PROFILE = "full"
COMPACT_PROFILE = "compact"

#: Selectable onboarding profiles. `full` is the default everywhere.
ONBOARDING_PROFILES: tuple[str, ...] = (FULL_PROFILE, COMPACT_PROFILE)

#: Default ceiling for the compact profile, in `count_tokens` tokens.
DEFAULT_COMPACT_TOKEN_BUDGET = 900

#: Section identifiers, in the order the allocator fills them.
_SECTION_ENTRY_POINTS = "entry_points"
_SECTION_CLUSTERS = "directory_clusters"
_SECTION_READING_ORDER = "reading_order"
_SECTION_TESTS = "test_surface"
_SECTION_CONFIG = "configuration_surface"
_SECTION_SOURCE_PATHS = "source_paths"

_OMISSION_TOKEN_BUDGET = "token_budget"
_OMISSION_SECTION_CAP = "section_cap"
_OMISSION_FOLDED_DIRECTORIES = "folded_directories"
_OMISSION_UNRENDERABLE_PATH = "unrenderable_path"

_MAX_LANGUAGES = 5
_MAX_PREFIX_DEPTH = 64
_CLUSTER_ROW_CAP = 18
_READING_ORDER_CAP = 12
_ENTRY_POINT_CAP = 8
_TEST_CLUSTER_CAP = 8
_CONFIG_CAP = 8

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class OnboardingError(ValueError):
    """Raised when onboarding output cannot be rendered."""


class OrientationOmission(BaseModel):
    """One section's dropped items, so a consumer never guesses at truncation."""

    section: str
    omitted_items: int = Field(ge=1)
    total_items: int = Field(ge=1)
    reason: str


class OrientationReceipt(BaseModel):
    """Budget accounting and omissions for one rendered orientation view."""

    profile: str
    requested_budget: int = Field(gt=0)
    consumed_budget: int = Field(ge=0)
    sections: list[str] = Field(default_factory=list[str])
    omissions: list[OrientationOmission] = Field(default_factory=list[OrientationOmission])


class CompactOrientation(BaseModel):
    """A budget-bounded orientation view and its receipt."""

    content: str
    receipt: OrientationReceipt


def render_handle(text: str) -> str:
    """Render repository-controlled text as an inline code span it cannot escape.

    Paths are repository content, and a backtick is a legal POSIX filename
    character that `git ls-files` emits unquoted, so an unescaped `` `path` ``
    row lets a repository author close the span and inject prose into whatever
    consumes the view -- including an agent session primer. This sizes the
    delimiter to one longer than the longest backtick run, and pads when the
    text starts or ends with a backtick, per CommonMark's code-span rules.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def is_renderable_path(path: str) -> bool:
    """Whether a path can be rendered as one markdown row without forging structure.

    A control character -- a newline above all -- would let repository content
    fabricate headings and receipt lines that a consumer cannot distinguish
    from the renderer's own. Such paths are dropped rather than escaped,
    because no inline escaping keeps them on one line.
    """
    return not _CONTROL_CHARACTERS.search(path)


def render_onboarding_markdown(graph: ArchGraph, *, max_files: int = 40) -> str:
    if max_files <= 0:
        raise OnboardingError("max-files must be greater than zero")
    file_nodes = _nodes_of_type(graph, GraphNodeType.FILE)
    entry_nodes = _nodes_of_type(graph, GraphNodeType.ENTRY_POINT)
    interface_nodes = _nodes_of_type(graph, GraphNodeType.INTERFACE)
    config_nodes = _nodes_of_type(graph, GraphNodeType.CONFIG)
    test_nodes = _nodes_of_type(graph, GraphNodeType.TEST)
    modules = _modules(file_nodes)
    reading_order = _reading_order(entry_nodes, interface_nodes, file_nodes, test_nodes, max_files)
    hotspots = _complexity_hotspots(file_nodes, max_files)
    lines = [
        f"# Onboarding: {graph.project.name}",
        "",
        "## Repository Overview",
        "",
        f"- **Files:** {graph.project.total_files}",
        f"- **Lines:** {graph.project.total_lines}",
        f"- **Graph nodes:** {len(graph.nodes)}",
        f"- **Graph edges:** {len(graph.edges)}",
        "",
        "## Languages And File Counts",
        "",
    ]
    lines.extend(_language_lines(graph))
    lines.extend(["", "## Architecture Modules", ""])
    lines.extend(_module_lines(modules))
    lines.extend(["", "## Entry Points", ""])
    lines.extend(_node_path_lines(entry_nodes, max_files))
    lines.extend(["", "## Public Interfaces", ""])
    lines.extend(_node_path_lines(interface_nodes, max_files))
    lines.extend(["", "## Recommended Reading Order", ""])
    lines.extend(
        f"{index}. {render_handle(path)}" for index, path in enumerate(reading_order, start=1)
    )
    lines.extend(["", "## Complexity Hotspots", ""])
    lines.extend(_hotspot_lines(hotspots))
    lines.extend(["", "## Test Surface", ""])
    lines.extend(_node_path_lines(test_nodes, max_files))
    lines.extend(["", "## Configuration Surface", ""])
    lines.extend(_node_path_lines(config_nodes, max_files))
    lines.extend(["", "## Generated Artifact Metadata", ""])
    lines.extend(
        [
            f"- **Schema version:** `{graph.schema_version.value}`",
            f"- **archex version:** `{graph.metadata.archex_version}`",
            f"- **Commit:** `{graph.metadata.commit_hash or 'none'}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _nodes_of_type(graph: ArchGraph, node_type: GraphNodeType) -> list[GraphNode]:
    """Nodes of one type, excluding any whose path cannot be rendered as one row."""
    return [
        node
        for node in graph.nodes
        if node.type == node_type and is_renderable_path(node.path or node.label)
    ]


def _unrenderable_path_count(graph: ArchGraph, node_types: Sequence[GraphNodeType]) -> int:
    return sum(
        1
        for node in graph.nodes
        if node.type in node_types and not is_renderable_path(node.path or node.label)
    )


def _language_lines(graph: ArchGraph) -> list[str]:
    if not graph.project.languages:
        return ["- None"]
    return [
        f"- {render_handle(language)}: {count} files"
        for language, count in sorted(graph.project.languages.items())
    ]


def _modules(file_nodes: list[GraphNode]) -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    for node in file_nodes:
        if node.path is None:
            continue
        module_name = node.path.split("/", maxsplit=1)[0] if "/" in node.path else "root"
        modules.setdefault(module_name, []).append(node.path)
    return {name: sorted(paths) for name, paths in sorted(modules.items())}


def _module_lines(modules: dict[str, list[str]]) -> list[str]:
    if not modules:
        return ["- None"]
    return [f"- {render_handle(name)}: {len(paths)} files" for name, paths in modules.items()]


def _node_path_lines(nodes: list[GraphNode], max_items: int) -> list[str]:
    if not nodes:
        return ["- None"]
    return [f"- {render_handle(node.path or node.label)}" for node in nodes[:max_items]]


def _reading_order(
    entry_nodes: list[GraphNode],
    interface_nodes: list[GraphNode],
    file_nodes: list[GraphNode],
    test_nodes: list[GraphNode],
    max_files: int,
) -> list[str]:
    ordered: list[str] = []
    for node in (
        entry_nodes
        + interface_nodes
        + sorted(
            file_nodes,
            key=lambda item: (
                -item.complexity.public_interface_count,
                -item.complexity.import_fan_in,
                item.path or item.label,
            ),
        )
        + test_nodes
    ):
        path = node.path or node.label
        if path not in ordered:
            ordered.append(path)
        if len(ordered) >= max_files:
            break
    return ordered


def _complexity_hotspots(file_nodes: list[GraphNode], max_files: int) -> list[GraphNode]:
    return sorted(
        file_nodes,
        key=lambda node: (
            -node.complexity.token_count,
            -node.complexity.symbol_count,
            node.path or node.label,
        ),
    )[:max_files]


def _hotspot_lines(nodes: list[GraphNode]) -> list[str]:
    if not nodes:
        return ["- None"]
    return [
        f"- {render_handle(node.path or node.label)}: {node.complexity.token_count} tokens, "
        f"{node.complexity.symbol_count} symbols"
        for node in nodes
    ]


def render_compact_orientation(
    graph: ArchGraph,
    *,
    token_budget: int = DEFAULT_COMPACT_TOKEN_BUDGET,
) -> CompactOrientation:
    """Render the compact orientation profile under a hard token ceiling.

    Every listed item carries an exact fetch handle: a file row carries its
    repository-relative path and a cluster row carries its exact directory
    prefix. Folded and budget-dropped items are reported as counts in the
    receipt, never replaced by an approximate path.

    Raises:
        OnboardingError: if `token_budget` is not positive or cannot hold the
            overview plus the omissions receipt.
    """
    if token_budget <= 0:
        raise OnboardingError("token-budget must be greater than zero")
    file_nodes = _nodes_of_type(graph, GraphNodeType.FILE)
    test_nodes = _nodes_of_type(graph, GraphNodeType.TEST)
    config_nodes = _nodes_of_type(graph, GraphNodeType.CONFIG)
    entry_nodes = _nodes_of_type(graph, GraphNodeType.ENTRY_POINT)
    unrenderable = _unrenderable_path_count(
        graph,
        (GraphNodeType.FILE, GraphNodeType.TEST, GraphNodeType.CONFIG, GraphNodeType.ENTRY_POINT),
    )
    extra_omissions = (
        [
            OrientationOmission(
                section=_SECTION_SOURCE_PATHS,
                omitted_items=unrenderable,
                total_items=unrenderable,
                reason=_OMISSION_UNRENDERABLE_PATH,
            )
        ]
        if unrenderable
        else []
    )

    sections = [
        _entry_point_section(entry_nodes),
        _cluster_section(file_nodes, config_nodes),
        _reading_order_section(graph, entry_nodes),
        _test_section(test_nodes),
        _config_section(config_nodes),
    ]
    header = _compact_header(graph)
    floor = _render_body(header, sections, {section.name: 0 for section in sections})
    reserve = _render_omissions([*_worst_case_omissions(sections), *extra_omissions])
    minimum = count_tokens(floor + "\n" + reserve)
    if minimum > token_budget:
        raise OnboardingError(
            f"token-budget {token_budget} cannot hold the compact orientation overview "
            f"and its omission receipt; at least {minimum} tokens are required"
        )

    content, receipt = _fit_sections(header, sections, token_budget, extra_omissions)
    return CompactOrientation(content=content, receipt=receipt)


@dataclass
class _Section:
    """One budget-allocatable block: a title, ordered rows, and pre-known omissions."""

    name: str
    title: str
    rows: list[str]
    omissions: list[OrientationOmission] = field(default_factory=list[OrientationOmission])


def _compact_header(graph: ArchGraph) -> list[str]:
    languages = sorted(graph.project.languages.items(), key=lambda item: (-item[1], item[0]))
    shown = languages[:_MAX_LANGUAGES]
    language_text = ", ".join(f"{name} {count}" for name, count in shown) or "none"
    if len(languages) > len(shown):
        language_text += f" (+{len(languages) - len(shown)} more)"
    return [
        f"## Orientation: {graph.project.name} (compact)",
        "",
        f"- Files {graph.project.total_files} | lines {graph.project.total_lines} | "
        f"nodes {len(graph.nodes)} | edges {len(graph.edges)}",
        f"- Languages: {language_text}",
        f"- archex `{graph.metadata.archex_version}` | commit "
        f"`{graph.metadata.commit_hash or 'none'}`",
    ]


def _entry_point_section(entry_nodes: list[GraphNode]) -> _Section:
    paths = _unique_paths(entry_nodes)
    return _capped_section(
        _SECTION_ENTRY_POINTS,
        "### Entry points",
        [f"- {render_handle(path)}" for path in paths[:_ENTRY_POINT_CAP]],
        total_items=len(paths),
    )


def _cluster_section(file_nodes: list[GraphNode], config_nodes: list[GraphNode]) -> _Section:
    paths = _unique_paths([*file_nodes, *config_nodes])
    clusters = _cluster_directories(paths, _CLUSTER_ROW_CAP)
    folded_dirs = sum(cluster.folded_dirs for cluster in clusters)
    omissions: list[OrientationOmission] = []
    if folded_dirs:
        omissions.append(
            OrientationOmission(
                section=_SECTION_CLUSTERS,
                omitted_items=folded_dirs,
                total_items=folded_dirs + len(clusters),
                reason=_OMISSION_FOLDED_DIRECTORIES,
            )
        )
    return _Section(
        _SECTION_CLUSTERS,
        f"### Directory clusters ({len(paths)} source files)",
        [_cluster_row(cluster, "file") for cluster in clusters],
        omissions=omissions,
    )


def _reading_order_section(graph: ArchGraph, entry_nodes: list[GraphNode]) -> _Section:
    ordered = _unique_paths(entry_nodes)[:2]
    rows = [
        f"{index}. {render_handle(path)} (entry point)"
        for index, path in enumerate(ordered, start=1)
    ]
    hubs = _file_hubs(graph, _READING_ORDER_CAP + len(ordered))
    total = len(ordered) + len(hubs)
    for path, degree in hubs:
        if path in ordered:
            total -= 1
            continue
        ordered.append(path)
        rows.append(f"{len(ordered)}. {render_handle(path)} (degree {degree})")
    return _capped_section(
        _SECTION_READING_ORDER,
        "### Recommended reading order (entry points, then graph hubs)",
        rows[: _READING_ORDER_CAP + 2],
        total_items=total,
    )


def _test_section(test_nodes: list[GraphNode]) -> _Section:
    paths = _unique_paths(test_nodes)
    clusters = _cluster_directories(paths, _TEST_CLUSTER_CAP)
    folded_dirs = sum(cluster.folded_dirs for cluster in clusters)
    omissions: list[OrientationOmission] = []
    if folded_dirs:
        omissions.append(
            OrientationOmission(
                section=_SECTION_TESTS,
                omitted_items=folded_dirs,
                total_items=folded_dirs + len(clusters),
                reason=_OMISSION_FOLDED_DIRECTORIES,
            )
        )
    return _Section(
        _SECTION_TESTS,
        f"### Test surface ({len(paths)} files)",
        [_cluster_row(cluster, "test file") for cluster in clusters],
        omissions=omissions,
    )


def _config_section(config_nodes: list[GraphNode]) -> _Section:
    paths = _unique_paths(config_nodes)
    return _capped_section(
        _SECTION_CONFIG,
        f"### Configuration surface ({len(paths)} files)",
        [f"- {render_handle(path)}" for path in paths[:_CONFIG_CAP]],
        total_items=len(paths),
    )


def _capped_section(
    name: str,
    title: str,
    rows: Sequence[str],
    *,
    total_items: int,
) -> _Section:
    """Build a section whose item list was capped before budget allocation."""
    omissions: list[OrientationOmission] = []
    if total_items > len(rows):
        omissions.append(
            OrientationOmission(
                section=name,
                omitted_items=total_items - len(rows),
                total_items=total_items,
                reason=_OMISSION_SECTION_CAP,
            )
        )
    return _Section(name, title, list(rows), omissions=omissions)


def _unique_paths(nodes: Iterable[GraphNode]) -> list[str]:
    seen: dict[str, None] = {}
    for node in nodes:
        path = node.path or node.label
        if path:
            seen.setdefault(path, None)
    return list(seen)


def _file_hubs(graph: ArchGraph, needed: int) -> list[tuple[str, int]]:
    """Rank source files by the existing graph-degree hub ranking.

    Hub summaries come from the graph rather than from `_nodes_of_type`, so the
    unrenderable-path filter has to be applied here too.
    """
    if needed <= 0 or not graph.nodes:
        return []
    query = GraphQuery(graph, hub_degree=1)
    for limit in (max(needed * 16, 256), len(graph.nodes)):
        selected: list[tuple[str, int]] = []
        for hub in query.hubs(limit=limit, threshold=1).hubs:
            if hub.type != GraphNodeType.FILE.value or not hub.path:
                continue
            if not is_renderable_path(hub.path):
                continue
            selected.append((hub.path, hub.degree))
            if len(selected) >= needed:
                return selected
        if limit >= len(graph.nodes):
            return selected
    return []


@dataclass(frozen=True)
class _Cluster:
    """A directory prefix, the files it accounts for, and the directories folded into it.

    `residual` marks a row whose files live under `prefix` but outside every
    listed subdirectory, so the prefix stays an exact fetch handle for them.
    `folded_dirs` counts the subdirectories that were too small to list.
    """

    prefix: str
    file_count: int
    folded_dirs: int = 0
    residual: bool = False


def _cluster_row(cluster: _Cluster, noun: str) -> str:
    label = cluster.prefix or "./"
    if cluster.residual:
        return (
            f"- {render_handle(label)} {cluster.file_count} {noun}(s) "
            "outside the listed subdirectories"
        )
    return f"- {render_handle(label)} {cluster.file_count} {noun}(s)"


def _cluster_directories(paths: Sequence[str], max_rows: int) -> list[_Cluster]:
    """Cluster paths by directory prefix, refining the largest cluster first.

    The refinement is deterministic: the largest splittable cluster whose
    children still fit `max_rows` is expanded, ties broken by prefix. Children
    holding fewer than a corpus-derived threshold of files are folded into a
    residual row that keeps their parent prefix listed, so no file loses a
    locator. Single-child directory chains are compressed, so a deep package
    root is reported at the depth where it actually branches.

    Each path is split into segments once. Directory depth is repository
    controlled, so re-splitting every path on every descent step would make a
    deeply nested corpus cost depth times the corpus.
    """
    if not paths or max_rows < 1:
        return []
    entries = [_DirEntry(tuple(path.split("/")[:-1])) for path in sorted(paths)]
    fold_below = max(2, len(entries) // (max_rows * 4))
    frontier = [_Group(_compress_depth(0, entries), entries)]
    residuals: list[_Cluster] = []
    while len(frontier) + len(residuals) < max_rows:
        expanded = False
        for group in sorted(frontier, key=lambda item: (-len(item.entries), item.prefix)):
            if len(group.entries) <= 1:
                continue
            children, folded_dirs, folded_files = _split_group(group, fold_below)
            if not children or (len(children) == 1 and not folded_dirs and not folded_files):
                continue
            rows = len(frontier) - 1 + len(residuals) + len(children) + (1 if folded_files else 0)
            if rows > max_rows:
                continue
            frontier.remove(group)
            frontier.extend(children)
            if folded_files:
                residuals.append(
                    _Cluster(
                        group.prefix,
                        folded_files,
                        folded_dirs=folded_dirs,
                        residual=True,
                    )
                )
            expanded = True
            break
        if not expanded:
            break
    clusters = [_Cluster(group.prefix, len(group.entries)) for group in frontier]
    clusters.extend(residuals)
    return sorted(clusters, key=lambda cluster: (-cluster.file_count, cluster.prefix))


@dataclass(frozen=True)
class _DirEntry:
    """One path reduced to the directory segments that can carry a handle."""

    segments: tuple[str, ...]


@dataclass(frozen=True)
class _Group:
    """A frontier cluster: the depth its prefix is cut at, and its members."""

    depth: int
    entries: list[_DirEntry]

    @property
    def prefix(self) -> str:
        if self.depth == 0:
            return ""
        return "/".join(self.entries[0].segments[: self.depth]) + "/"


def _split_group(group: _Group, fold_below: int) -> tuple[list[_Group], int, int]:
    """Group members one directory level below `group.depth`, folding small children."""
    children: dict[str, list[_DirEntry]] = {}
    direct = 0
    for entry in group.entries:
        if len(entry.segments) <= group.depth:
            direct += 1
            continue
        children.setdefault(entry.segments[group.depth], []).append(entry)
    kept: list[_Group] = []
    folded_dirs = 0
    folded_files = direct
    for _segment, members in sorted(children.items()):
        if len(members) < fold_below:
            folded_dirs += 1
            folded_files += len(members)
            continue
        kept.append(_Group(_compress_depth(group.depth + 1, members), members))
    return kept, folded_dirs, folded_files


def _compress_depth(depth: int, entries: Sequence[_DirEntry]) -> int:
    """Descend through single-child directory chains to the first branch point."""
    current = depth
    while current < _MAX_PREFIX_DEPTH:
        segment: str | None = None
        for entry in entries:
            if len(entry.segments) <= current:
                return current
            if segment is None:
                segment = entry.segments[current]
            elif entry.segments[current] != segment:
                return current
        if segment is None:
            return current
        current += 1
    return current


def _fit_sections(
    header: Sequence[str],
    sections: Sequence[_Section],
    token_budget: int,
    extra_omissions: Sequence[OrientationOmission] = (),
) -> tuple[str, OrientationReceipt]:
    """Fill sections in priority order under a hard token ceiling.

    The omissions receipt is reserved before any row is admitted, using the
    worst case each section could report plus any omission already known
    (an unrenderable path, say), so the receipt itself can never be the
    content that gets truncated.
    """
    reserve = count_tokens(_render_omissions([*_worst_case_omissions(sections), *extra_omissions]))
    kept: dict[str, int] = {section.name: 0 for section in sections}
    for section in sections:
        for count in range(1, len(section.rows) + 1):
            trial = {**kept, section.name: count}
            if count_tokens(_render_body(header, sections, trial)) + reserve > token_budget:
                break
            kept[section.name] = count

    omissions = [*_actual_omissions(sections, kept), *extra_omissions]
    body = _render_body(header, sections, kept)
    content = body + "\n" + _render_omissions(omissions)
    while count_tokens(content) > token_budget:
        trimmed = _trim_last_row(sections, kept)
        if trimmed is None:
            break
        omissions = [*_actual_omissions(sections, kept), *extra_omissions]
        content = _render_body(header, sections, kept) + "\n" + _render_omissions(omissions)
    return content, OrientationReceipt(
        profile=COMPACT_PROFILE,
        requested_budget=token_budget,
        consumed_budget=count_tokens(content),
        sections=[section.name for section in sections if kept[section.name]],
        omissions=omissions,
    )


def _trim_last_row(sections: Sequence[_Section], kept: dict[str, int]) -> str | None:
    """Drop one row from the lowest-priority non-empty section."""
    for section in reversed(sections):
        if kept[section.name]:
            kept[section.name] -= 1
            return section.name
    return None


def _render_body(
    header: Sequence[str],
    sections: Sequence[_Section],
    kept: dict[str, int],
) -> str:
    lines = list(header)
    for section in sections:
        count = kept[section.name]
        if not count:
            continue
        lines.extend(["", section.title, ""])
        lines.extend(section.rows[:count])
    return "\n".join(lines) + "\n"


def _render_omissions(omissions: Sequence[OrientationOmission]) -> str:
    lines = ["### Omissions", ""]
    if not omissions:
        lines.append("- None")
    else:
        lines.extend(
            f"- `{omission.section}`: {omission.omitted_items} of {omission.total_items} "
            f"omitted ({omission.reason})"
            for omission in omissions
        )
    return "\n".join(lines) + "\n"


def _worst_case_omissions(sections: Sequence[_Section]) -> list[OrientationOmission]:
    """The largest omission block the sections could produce, for reservation."""
    worst: list[OrientationOmission] = []
    for section in sections:
        worst.extend(section.omissions)
        if section.rows:
            worst.append(
                OrientationOmission(
                    section=section.name,
                    omitted_items=len(section.rows),
                    total_items=len(section.rows),
                    reason=_OMISSION_TOKEN_BUDGET,
                )
            )
    return worst


def _actual_omissions(
    sections: Sequence[_Section],
    kept: dict[str, int],
) -> list[OrientationOmission]:
    omissions: list[OrientationOmission] = []
    for section in sections:
        omissions.extend(section.omissions)
        dropped = len(section.rows) - kept[section.name]
        if dropped > 0:
            omissions.append(
                OrientationOmission(
                    section=section.name,
                    omitted_items=dropped,
                    total_items=len(section.rows),
                    reason=_OMISSION_TOKEN_BUDGET,
                )
            )
    return omissions
