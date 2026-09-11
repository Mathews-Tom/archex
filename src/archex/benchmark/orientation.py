"""R24: deterministic orientation measurement for the onboarding profiles.

This harness answers one question about the compact profile: does it cost fewer
context tokens **without** making the files a task needs harder to reach? It is
offline, model-free and agent-free -- every number is recomputed from a graph
artifact and a frozen task population, so the report regenerates byte-for-byte.

Metric definitions, stated here because two of them are models rather than
measurements:

`exact_path_hits`
    An expected file whose exact repository-relative path appears as a handle
    (a backticked token) in the rendered view.
`locatable_files`
    Not named exactly, but some ancestor directory appears as a handle, so one
    directory listing or path-scoped query reaches it.
`unlocated_files`
    Neither. A repository-wide search is the only route left.
`locator_breadth`
    How many indexed files the matched handle stands for: 1 for an exact path,
    the file count under the nearest matched ancestor for a locatable file, and
    the whole indexed corpus for an unlocated one. Lower is better; it is what
    stops a view from buying completeness with uselessly coarse prefixes.
`modeled_exploration_calls`
    A **model**, not an observation: one call for the orientation view, one per
    distinct matched locator directory, and one per unlocated file. R20 records
    no orientation-phase call counts (its telemetry is whole-session), so no
    agent-observed call reduction is claimed anywhere in this report.
`omitted_items` / `reported_by_tool`
    Items a profile could have listed but did not. `reported_by_tool` is the
    load-bearing column: the compact profile reports its omissions in its own
    receipt, the full profile truncates silently and reports nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from archex.benchmark.loader import load_task
from archex.graph_artifact import ArchGraph, GraphNodeType
from archex.onboarding import (
    COMPACT_PROFILE,
    FULL_PROFILE,
    render_compact_orientation,
    render_onboarding_markdown,
)
from archex.reporting import count_tokens

#: Matches a rendered handle at any delimiter width, since a path containing a
#: backtick is rendered with a widened fence and padding by `render_handle`.
_HANDLE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")

#: Reported units per omission reason, so files, rows and directories are never summed.
_OMISSION_UNITS = {
    "token_budget": "rows",
    "section_cap": "items",
    "folded_directories": "directories",
    "unrenderable_path": "files",
}

#: Section titles the full profile caps at `max_files` without saying so.
_FULL_CAPPED_SECTIONS: dict[str, GraphNodeType | None] = {
    "Entry Points": GraphNodeType.ENTRY_POINT,
    "Public Interfaces": GraphNodeType.INTERFACE,
    "Complexity Hotspots": GraphNodeType.FILE,
    "Test Surface": GraphNodeType.TEST,
    "Configuration Surface": GraphNodeType.CONFIG,
}

_PATH_NODE_TYPES = (GraphNodeType.FILE, GraphNodeType.TEST, GraphNodeType.CONFIG)


class OrientationManifestError(ValueError):
    """Raised when the frozen orientation manifest is invalid."""


class OrientationManifest(BaseModel):
    """The frozen population and profile settings for one orientation run."""

    model_config = ConfigDict(extra="forbid")

    milestone: str
    repository: str
    tasks_dir: str
    task_ids: list[str] = Field(min_length=1)
    full_max_files: int = Field(gt=0)
    compact_token_budgets: list[int] = Field(min_length=1)


class OrientationOmissionMeasurement(BaseModel):
    """One section's omitted items, in a stated unit, and who reported them."""

    section: str
    omitted_items: int
    available_items: int
    unit: str
    reason: str
    reported_by_tool: bool


class OrientationTaskMeasurement(BaseModel):
    """One task's reachability under one rendered view."""

    task_id: str
    expected_files: int
    exact_path_hits: int
    locatable_files: int
    unlocated_files: int
    completeness: float
    mean_locator_breadth: float
    modeled_exploration_calls: int


class OrientationProfileMeasurement(BaseModel):
    """One profile's context cost, reachability, and omission honesty."""

    profile: str
    token_budget: int | None = None
    max_files: int | None = None
    context_tokens: int
    expected_files: int
    exact_path_hits: int
    locatable_files: int
    unlocated_files: int
    completeness: float
    mean_locator_breadth: float
    modeled_exploration_calls: int
    corpus_files: int
    files_named: int
    files_not_named: int
    self_reported_items: int
    self_reported_sections: int
    omissions: list[OrientationOmissionMeasurement] = Field(
        default_factory=list[OrientationOmissionMeasurement]
    )
    tasks: list[OrientationTaskMeasurement] = Field(
        default_factory=list[OrientationTaskMeasurement]
    )


class OrientationReport(BaseModel):
    """A regenerable comparison of the onboarding profiles over one graph."""

    milestone: str
    repository: str
    graph_revision: str | None
    """The commit whose tree the measured graph describes -- the corpus, not the code."""
    graph_archex_version: str
    """The archex version that exported the graph."""
    indexed_files: int
    graph_nodes: int
    graph_edges: int
    task_count: int
    task_ids: list[str]
    profiles: list[OrientationProfileMeasurement]


def load_orientation_manifest(path: Path) -> OrientationManifest:
    """Load and strictly validate the frozen orientation manifest."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise OrientationManifestError(f"Failed to parse YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OrientationManifestError(
            f"Expected a YAML mapping in {path}, got {type(raw).__name__}"
        )
    try:
        manifest = OrientationManifest.model_validate(cast("dict[str, object]", raw))
    except ValidationError as exc:
        raise OrientationManifestError(f"Invalid orientation manifest {path}: {exc}") from exc
    duplicates = [task_id for task_id, count in Counter(manifest.task_ids).items() if count > 1]
    if duplicates:
        raise OrientationManifestError(
            f"Duplicate task ids in {path}: {', '.join(sorted(duplicates))}"
        )
    return manifest


def load_orientation_tasks(
    manifest: OrientationManifest,
    root: Path,
) -> list[tuple[str, list[str]]]:
    """Resolve the manifest's frozen task ids to their expected-file lists."""
    tasks_dir = root / manifest.tasks_dir
    if not tasks_dir.is_dir():
        raise OrientationManifestError(f"Tasks directory not found: {tasks_dir}")
    resolved: list[tuple[str, list[str]]] = []
    for task_id in manifest.task_ids:
        task_path = tasks_dir / f"{task_id}.yaml"
        if not task_path.is_file():
            raise OrientationManifestError(f"Manifest task {task_id!r} has no file at {task_path}")
        task = load_task(task_path)
        if task.task_id != task_id:
            raise OrientationManifestError(
                f"{task_path} declares task_id {task.task_id!r}, expected {task_id!r}"
            )
        resolved.append((task_id, list(task.expected_files)))
    return resolved


def run_orientation_benchmark(
    graph: ArchGraph,
    tasks: list[tuple[str, list[str]]],
    manifest: OrientationManifest,
) -> OrientationReport:
    """Measure every profile in the manifest over the same graph and tasks."""
    corpus = _corpus_paths(graph)
    breadth = _directory_breadth(corpus)
    profiles = [_measure_full(graph, tasks, manifest.full_max_files, corpus, breadth)]
    profiles.extend(
        _measure_compact(graph, tasks, budget, corpus, breadth)
        for budget in manifest.compact_token_budgets
    )
    return OrientationReport(
        milestone=manifest.milestone,
        graph_revision=graph.metadata.commit_hash,
        graph_archex_version=graph.metadata.archex_version,
        repository=manifest.repository,
        indexed_files=len(corpus),
        graph_nodes=len(graph.nodes),
        graph_edges=len(graph.edges),
        task_count=len(tasks),
        task_ids=[task_id for task_id, _ in tasks],
        profiles=profiles,
    )


def _measure_full(
    graph: ArchGraph,
    tasks: list[tuple[str, list[str]]],
    max_files: int,
    corpus: list[str],
    breadth: dict[str, int],
) -> OrientationProfileMeasurement:
    content = render_onboarding_markdown(graph, max_files=max_files)
    omissions = _full_profile_omissions(graph, content)
    return _profile_measurement(
        profile=FULL_PROFILE,
        content=content,
        tasks=tasks,
        corpus=corpus,
        breadth=breadth,
        omissions=omissions,
        max_files=max_files,
        token_budget=None,
    )


def _measure_compact(
    graph: ArchGraph,
    tasks: list[tuple[str, list[str]]],
    token_budget: int,
    corpus: list[str],
    breadth: dict[str, int],
) -> OrientationProfileMeasurement:
    orientation = render_compact_orientation(graph, token_budget=token_budget)
    omissions = [
        OrientationOmissionMeasurement(
            section=omission.section,
            omitted_items=omission.omitted_items,
            available_items=omission.total_items,
            unit=_OMISSION_UNITS.get(omission.reason, "items"),
            reason=omission.reason,
            reported_by_tool=True,
        )
        for omission in orientation.receipt.omissions
    ]
    return _profile_measurement(
        profile=COMPACT_PROFILE,
        content=orientation.content,
        tasks=tasks,
        corpus=corpus,
        breadth=breadth,
        omissions=omissions,
        max_files=None,
        token_budget=token_budget,
    )


def _profile_measurement(
    *,
    profile: str,
    content: str,
    tasks: list[tuple[str, list[str]]],
    corpus: list[str],
    breadth: dict[str, int],
    omissions: list[OrientationOmissionMeasurement],
    max_files: int | None,
    token_budget: int | None,
) -> OrientationProfileMeasurement:
    handles = _handles(content)
    measurements = [
        _measure_task(task_id, expected, handles, corpus, breadth) for task_id, expected in tasks
    ]
    expected_total = sum(item.expected_files for item in measurements)
    exact = sum(item.exact_path_hits for item in measurements)
    locatable = sum(item.locatable_files for item in measurements)
    unlocated = sum(item.unlocated_files for item in measurements)
    breadth_total = sum(
        item.mean_locator_breadth * item.expected_files
        for item in measurements
        if item.expected_files
    )
    files_named = len(handles & set(corpus))
    return OrientationProfileMeasurement(
        profile=profile,
        token_budget=token_budget,
        max_files=max_files,
        context_tokens=count_tokens(content),
        expected_files=expected_total,
        exact_path_hits=exact,
        locatable_files=locatable,
        unlocated_files=unlocated,
        completeness=_ratio(exact + locatable, expected_total),
        mean_locator_breadth=round(breadth_total / expected_total, 4) if expected_total else 0.0,
        modeled_exploration_calls=sum(item.modeled_exploration_calls for item in measurements),
        corpus_files=len(corpus),
        files_named=files_named,
        files_not_named=len(corpus) - files_named,
        self_reported_items=sum(item.omitted_items for item in omissions if item.reported_by_tool),
        self_reported_sections=sum(1 for item in omissions if item.reported_by_tool),
        omissions=omissions,
        tasks=measurements,
    )


def _measure_task(
    task_id: str,
    expected: list[str],
    handles: set[str],
    corpus: list[str],
    breadth: dict[str, int],
) -> OrientationTaskMeasurement:
    exact = 0
    locatable = 0
    unlocated = 0
    breadths: list[int] = []
    locator_dirs: set[str] = set()
    for path in expected:
        if path in handles:
            exact += 1
            breadths.append(1)
            continue
        matched = _nearest_handle_ancestor(path, handles)
        if matched is None:
            unlocated += 1
            breadths.append(len(corpus))
            continue
        locatable += 1
        locator_dirs.add(matched)
        breadths.append(breadth.get(matched, len(corpus)))
    calls = 1 + len(locator_dirs) + unlocated
    return OrientationTaskMeasurement(
        task_id=task_id,
        expected_files=len(expected),
        exact_path_hits=exact,
        locatable_files=locatable,
        unlocated_files=unlocated,
        completeness=_ratio(exact + locatable, len(expected)),
        mean_locator_breadth=round(sum(breadths) / len(breadths), 4) if breadths else 0.0,
        modeled_exploration_calls=calls,
    )


def _nearest_handle_ancestor(path: str, handles: set[str]) -> str | None:
    """The most specific ancestor directory of `path` that the view named."""
    segments = path.split("/")[:-1]
    while segments:
        prefix = "/".join(segments)
        if f"{prefix}/" in handles or prefix in handles:
            return f"{prefix}/"
        segments = segments[:-1]
    return None


def _corpus_paths(graph: ArchGraph) -> list[str]:
    seen: dict[str, None] = {}
    for node in graph.nodes:
        if node.type in _PATH_NODE_TYPES and node.path:
            seen.setdefault(node.path, None)
    return sorted(seen)


def _directory_breadth(corpus: list[str]) -> dict[str, int]:
    """Indexed file count under every directory prefix in the corpus."""
    breadth: dict[str, int] = {}
    for path in corpus:
        segments = path.split("/")[:-1]
        for depth in range(1, len(segments) + 1):
            prefix = "/".join(segments[:depth]) + "/"
            breadth[prefix] = breadth.get(prefix, 0) + 1
    return breadth


def _full_profile_omissions(
    graph: ArchGraph,
    content: str,
) -> list[OrientationOmissionMeasurement]:
    """Count what the full profile truncated without reporting it.

    Counted in **distinct paths**, not rows, because the rendered rows are not
    distinct: interface nodes are per-symbol, so 40 interface rows can name far
    fewer than 40 files. Comparing an available path count against a row count
    would subtract one unit from another and overstate the omission. The
    reading-order section is deliberately excluded: a ranked top-N is not
    "omitting" the rest of the corpus in the same sense a truncated
    enumeration is, and charging it against all 1184 indexed files would
    inflate the total with a number the compact profile is never charged.
    """
    omissions: list[OrientationOmissionMeasurement] = []
    named = _handles(content)
    for section, node_type in _FULL_CAPPED_SECTIONS.items():
        if node_type is None:
            continue
        available_paths = _unique_paths_of_type(graph, node_type)
        printed = len(available_paths & named)
        available = len(available_paths)
        if available > printed:
            omissions.append(
                OrientationOmissionMeasurement(
                    section=section.lower().replace(" ", "_"),
                    omitted_items=available - printed,
                    available_items=available,
                    unit="files",
                    reason="section_cap",
                    reported_by_tool=False,
                )
            )
    return omissions


def _unique_paths_of_type(graph: ArchGraph, node_type: GraphNodeType) -> set[str]:
    return {node.path or node.label for node in graph.nodes if node.type is node_type}


def _handles(content: str) -> set[str]:
    """Every rendered fetch handle in a view, at whatever delimiter width it used."""
    return {match.group(2).strip() for match in _HANDLE.finditer(content)}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def format_orientation_markdown(report: OrientationReport) -> str:
    """Render the checked-in comparison report."""
    lines = [
        f"# Orientation profile comparison ({report.milestone})",
        "",
        "Deterministic, offline, model-free and agent-free: every number below is",
        "recomputed from one graph artifact and a frozen task population, so this",
        "report regenerates byte-for-byte from the same inputs.",
        "",
        "## Inputs",
        "",
        f"- Repository: `{report.repository}`",
        f"- Graph revision (corpus measured): `{report.graph_revision or 'none'}`",
        f"- Graph exported by archex `{report.graph_archex_version}`",
        f"- Indexed files: {report.indexed_files}",
        f"- Graph: {report.graph_nodes} nodes, {report.graph_edges} edges",
        f"- Tasks: {report.task_count} (frozen in the manifest)",
        "",
        "## Profiles",
        "",
        "| Profile | Budget | Context tokens | Exact paths | Locatable | Unlocated |"
        " Completeness | Mean locator breadth | Modeled calls | Files named |"
        " Files not named | Self-reported omissions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for profile in report.profiles:
        budget = (
            f"{profile.token_budget} tokens"
            if profile.token_budget is not None
            else f"max_files={profile.max_files}"
        )
        lines.append(
            f"| `{profile.profile}` | {budget} | {profile.context_tokens} |"
            f" {profile.exact_path_hits}/{profile.expected_files} |"
            f" {profile.locatable_files}/{profile.expected_files} |"
            f" {profile.unlocated_files}/{profile.expected_files} |"
            f" {profile.completeness:.4f} | {profile.mean_locator_breadth:.1f} |"
            f" {profile.modeled_exploration_calls} |"
            f" {profile.files_named}/{profile.corpus_files} |"
            f" {profile.files_not_named}/{profile.corpus_files} |"
            f" {profile.self_reported_items} in {profile.self_reported_sections} section(s) |"
        )
    lines.extend(["", "## Omissions per profile", ""])
    for profile in report.profiles:
        budget = (
            f"token_budget={profile.token_budget}"
            if profile.token_budget is not None
            else f"max_files={profile.max_files}"
        )
        lines.append(f"### `{profile.profile}` ({budget})")
        lines.append("")
        if not profile.omissions:
            lines.extend(["- None", ""])
            continue
        for omission in profile.omissions:
            reported = "reported by the tool" if omission.reported_by_tool else "**silent**"
            lines.append(
                f"- `{omission.section}`: {omission.omitted_items} of "
                f"{omission.available_items} {omission.unit} omitted "
                f"({omission.reason}, {reported})"
            )
        lines.append("")
    lines.extend(["## Metric definitions", ""])
    lines.extend(
        [
            "- **Exact paths** — expected files named by their exact repository-relative"
            " path in the view.",
            "- **Locatable** — not named exactly, but an ancestor directory is named, so"
            " one directory listing or path-scoped query reaches them.",
            "- **Unlocated** — neither; a repository-wide search is the only route left.",
            "- **Mean locator breadth** — indexed files the matched handle stands for"
            " (1 for an exact path, the corpus size for an unlocated file). Lower is"
            " better; it prevents buying completeness with uselessly coarse prefixes.",
            "- **Modeled calls** — a declared model, not an observation: one call for the"
            " view, one per distinct matched locator directory, one per unlocated file."
            " It charges one unit per locator directory regardless of how much that"
            " directory contains, so it penalises a view with finer prefixes; read it"
            " together with locator breadth, which moves the other way. R20's telemetry"
            " is whole-session and records no orientation-phase call counts, so no"
            " agent-observed call reduction is claimed here.",
            "- **Files named / not named** — indexed files whose exact path the view"
            " carries, measured identically for both profiles. This is the only"
            " omission figure comparable across profiles: an enumerating profile names"
            " more files, a clustering profile names fewer and points at prefixes"
            " instead.",
            "- **Self-reported omissions** — items the profile itself declares in its own"
            " receipt, in the unit each section counts (files, rows, or folded"
            " directories — never summed across units). The full profile declares"
            " nothing, which is the difference this column exists to show; the"
            " per-section lists above give the silent counts the harness had to"
            " reconstruct for it.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
