"""Deterministic onboarding guide rendering from graph artifacts."""

from __future__ import annotations

from archex.graph_artifact import ArchGraph, GraphNode, GraphNodeType


class OnboardingError(ValueError):
    """Raised when onboarding output cannot be rendered."""


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
    lines.extend(f"{index}. `{path}`" for index, path in enumerate(reading_order, start=1))
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
    return [node for node in graph.nodes if node.type == node_type]


def _language_lines(graph: ArchGraph) -> list[str]:
    if not graph.project.languages:
        return ["- None"]
    return [
        f"- `{language}`: {count} files"
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
    return [f"- `{name}`: {len(paths)} files" for name, paths in modules.items()]


def _node_path_lines(nodes: list[GraphNode], max_items: int) -> list[str]:
    if not nodes:
        return ["- None"]
    return [f"- `{node.path or node.label}`" for node in nodes[:max_items]]


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
        f"- `{node.path or node.label}`: {node.complexity.token_count} tokens, "
        f"{node.complexity.symbol_count} symbols"
        for node in nodes
    ]
