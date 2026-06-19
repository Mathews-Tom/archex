"""Competitive comparison report across retrieval, baseline, and compression lanes.

Renders the public head-to-head comparison as per-repo/task-family tables plus an
aggregate, never as an aggregate-only winner claim. Lanes are labeled by layer
type so a Headroom-style compression layer is never presented as a retrieval
engine: retrieval/baseline lanes fill the quality columns; compression lanes
contribute a compression ratio and are explicitly marked ``compression``.
"""

from __future__ import annotations

from archex.benchmark.headroom import load_headroom_results
from archex.benchmark.headtohead import (
    HeadToHeadManifestError,
    comparison_lane_layers,
    lane_label,
    result_provenance,
)
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    ComparisonLayerType,
    CompressionLayerResult,
    HeadToHeadManifest,
    Strategy,
)


def load_compression_results(
    manifest: HeadToHeadManifest, task_ids: list[str]
) -> list[CompressionLayerResult]:
    """Import compression-layer results for every artifact-backed layer in the manifest.

    Layers without an ``artifact_dir`` are skipped: local compression cannot be
    replayed from stored retrieval results. Returns the combined results.
    """
    results: list[CompressionLayerResult] = []
    for layer in manifest.compression_layers:
        if layer.artifact_dir is not None:
            results.extend(load_headroom_results(layer, task_ids))
    return results


_METRIC_HEADER = (
    "| Lane | Layer | Recall | Required-file recall | Region recall | Line recall | "
    "Precision | Context noise | Token eff. (compl.) | Compression ratio | Receipt acc. | "
    "Freshness correct | Cold-start ms | Warm p50 ms | Warm p95 ms |"
)
_METRIC_DIVIDER = "| " + " | ".join(["---"] * 15) + " |"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _num(value: float, *, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _optional_mean(values: list[float | None], *, digits: int = 2) -> str:
    present = [v for v in values if v is not None]
    return _num(_mean(present), digits=digits) if present else "n/a"


def _optional_rate(flags: list[bool | None]) -> str:
    present = [1.0 if flag else 0.0 for flag in flags if flag is not None]
    return _num(_mean(present)) if present else "n/a"


def _gated_rate(flags: list[bool], measured: list[bool]) -> str:
    present = [1.0 if flag else 0.0 for flag, ok in zip(flags, measured, strict=True) if ok]
    return _num(_mean(present)) if present else "n/a"


def _warm_latency(results: list[BenchmarkResult]) -> list[float]:
    return [r.warm_latency_ms or r.wall_time_ms for r in results]


def _retrieval_row(label: str, layer: ComparisonLayerType, results: list[BenchmarkResult]) -> str:
    latencies = _warm_latency(results)
    cells = [
        label,
        layer.value,
        _num(_mean([r.recall for r in results])),
        _num(_mean([r.required_file_recall for r in results])),
        _optional_mean([r.region_recall for r in results]),
        _optional_mean([r.line_recall for r in results]),
        _num(_mean([r.precision for r in results])),
        _optional_mean([r.context_noise_ratio for r in results]),
        _num(_mean([r.token_efficiency_with_completion for r in results])),
        _optional_mean([r.bundle_compression_ratio for r in results]),
        _optional_rate([r.receipt_accuracy for r in results]),
        _gated_rate(
            [r.freshness_correct for r in results],
            [r.freshness_measured for r in results],
        ),
        _num(_mean([r.cold_start_ms for r in results]), digits=0),
        _num(_percentile(latencies, 0.50), digits=0),
        _num(_percentile(latencies, 0.95), digits=0),
    ]
    return "| " + " | ".join(cells) + " |"


def _compression_row(
    label: str, layer: ComparisonLayerType, results: list[CompressionLayerResult]
) -> str:
    # Compression lanes are not retrieval engines: every retrieval-quality column
    # is n/a; the lane contributes only a compression ratio.
    cells = [
        label,
        layer.value,
        *(["n/a"] * 7),
        _num(_mean([r.bundle_compression_ratio for r in results])),
        *(["n/a"] * 5),
    ]
    return "| " + " | ".join(cells) + " |"


def _retrieval_lanes(
    manifest: HeadToHeadManifest, reports: list[BenchmarkReport]
) -> dict[str, list[BenchmarkResult]]:
    comparison: set[Strategy] = {
        Strategy.ARCHEX_QUERY,
        Strategy.EXTERNAL_MCP,
        manifest.raw_read_strategy,
        *manifest.archex.candidate_strategies,
    }
    lanes: dict[str, list[BenchmarkResult]] = {}
    for report in reports:
        for result in report.results:
            if result.strategy in comparison:
                lanes.setdefault(lane_label(result), []).append(result)
    return lanes


def _compression_lanes(
    results: list[CompressionLayerResult],
) -> dict[str, list[CompressionLayerResult]]:
    lanes: dict[str, list[CompressionLayerResult]] = {}
    for result in results:
        lanes.setdefault(result.lane_label, []).append(result)
    return lanes


def _ordered_retrieval_labels(
    manifest: HeadToHeadManifest, lanes: dict[str, list[BenchmarkResult]]
) -> list[str]:
    preferred = ["archex", *(s.value for s in manifest.archex.candidate_strategies)]
    preferred += [tool.name for tool in manifest.external_tools]
    preferred += [lane.name.value for lane in manifest.graphify_lanes]
    preferred.append("raw-ripgrep/read")
    ordered = [label for label in preferred if label in lanes]
    ordered += sorted(label for label in lanes if label not in preferred)
    return ordered


def _compression_provenance(results: list[CompressionLayerResult]) -> str:
    provenance = results[0].provenance
    layer = provenance.get("layer", "compression")
    version = provenance.get("version", "")
    source_lane = provenance.get("source_lane", "")
    run_mode = provenance.get("run_mode", "")
    return f"layer={layer}; version={version}; source_lane={source_lane}; run_mode={run_mode}"


def _metric_table(
    retrieval_order: list[str],
    retrieval_lanes: dict[str, list[BenchmarkResult]],
    layers: dict[str, ComparisonLayerType],
    compression_order: list[str],
    compression_lanes: dict[str, list[CompressionLayerResult]],
) -> list[str]:
    rows = [_METRIC_HEADER, _METRIC_DIVIDER]
    for label in retrieval_order:
        layer = layers.get(label, ComparisonLayerType.RETRIEVAL)
        rows.append(_retrieval_row(label, layer, retrieval_lanes[label]))
    for label in compression_order:
        layer = layers.get(label, ComparisonLayerType.COMPRESSION)
        rows.append(_compression_row(label, layer, compression_lanes[label]))
    return rows


def _task_family(report: BenchmarkReport) -> str:
    for result in report.results:
        if result.category is not None:
            return result.category.value
    return "unknown"


def format_competitive_markdown(
    manifest: HeadToHeadManifest,
    reports: list[BenchmarkReport],
    compression_results: list[CompressionLayerResult] | None = None,
) -> str:
    """Render the competitive comparison: per-repo/task-family tables plus aggregate.

    ``compression_results`` carries optional Headroom-style compression lanes. The
    report never emits a single aggregate-only winner; every lane and cell is shown.
    """
    if not reports:
        return "No competitive benchmark results."

    compression_results = compression_results or []
    layers = comparison_lane_layers(manifest)
    retrieval_lanes = _retrieval_lanes(manifest, reports)
    compression_lanes = _compression_lanes(compression_results)
    retrieval_order = _ordered_retrieval_labels(manifest, retrieval_lanes)
    compression_order = [
        label
        for label in (mode.value for layer in manifest.compression_layers for mode in layer.modes)
        if label in compression_lanes
    ]
    compression_order += sorted(
        label for label in compression_lanes if label not in compression_order
    )

    required_lanes = {"archex", manifest.external_tools[0].name, "raw-ripgrep/read"}
    missing_lanes = sorted(required_lanes.difference(retrieval_lanes))
    if missing_lanes:
        raise HeadToHeadManifestError(
            "Competitive report is missing lane(s): " + ", ".join(missing_lanes)
        )

    task_repo = {report.task_id: report.repo for report in reports}
    lines: list[str] = [
        "# archex Competitive Comparison",
        "",
        f"Manifest: `{manifest.name}`",
        f"Tasks: `{len(reports)}` external-repo tasks across "
        f"`{len({report.repo for report in reports})}` repos",
        f"Hardware notes: {manifest.hardware_notes}",
        "",
        "Headroom is evaluated as a compression / context-management layer, not a "
        "retrieval engine. Compression lanes contribute a compression ratio only; "
        "retrieval-quality columns are `n/a` for them.",
        "",
        "Graphify is evaluated as a graph / memory layer. "
        "`graphify_build_plus_query` includes graph construction/setup plus the first "
        "graph-backed answer; `graphify_query_warm` measures only the warm graph-query "
        "path against a prebuilt graph. These lanes are reported separately and are not "
        "framed as direct retrieval-equivalent wins.",
        "",
        "Metrics are reported per repo/task family and in aggregate. No aggregate-only "
        "winner is claimed; every lane and cell is shown.",
        "",
        "## Lanes and layer types",
        "",
        "| Lane | Layer type | Provenance |",
        "| --- | --- | --- |",
    ]
    for label in retrieval_order:
        layer = layers.get(label, ComparisonLayerType.RETRIEVAL)
        provenance = result_provenance(retrieval_lanes[label][0], manifest)
        lines.append(f"| {label} | {layer.value} | {provenance} |")
    for label in compression_order:
        layer = layers.get(label, ComparisonLayerType.COMPRESSION)
        provenance = _compression_provenance(compression_lanes[label])
        lines.append(f"| {label} | {layer.value} | {provenance} |")

    lines.extend(["", f"## Aggregate ({len(reports)} tasks)", ""])
    lines.extend(
        _metric_table(
            retrieval_order, retrieval_lanes, layers, compression_order, compression_lanes
        )
    )

    task_family = {report.task_id: _task_family(report) for report in reports}

    families = sorted({task_family[report.task_id] for report in reports})
    lines.extend(["", "## By task family", ""])
    for family in families:
        family_reports = [report for report in reports if task_family[report.task_id] == family]
        family_retrieval = _retrieval_lanes(manifest, family_reports)
        family_retrieval_order = _ordered_retrieval_labels(manifest, family_retrieval)
        family_compression = _compression_lanes(
            [r for r in compression_results if task_family.get(r.task_id) == family]
        )
        family_compression_order = [
            label for label in compression_order if label in family_compression
        ]
        lines.extend([f"### `{family}` ({len(family_reports)} tasks)", ""])
        lines.extend(
            _metric_table(
                family_retrieval_order,
                family_retrieval,
                layers,
                family_compression_order,
                family_compression,
            )
        )
        lines.append("")

    repos = sorted({report.repo for report in reports})
    lines.extend(["", "## By repo", ""])
    for repo in repos:
        repo_reports = [report for report in reports if report.repo == repo]
        repo_retrieval = _retrieval_lanes(manifest, repo_reports)
        repo_retrieval_order = _ordered_retrieval_labels(manifest, repo_retrieval)
        repo_compression = _compression_lanes(
            [r for r in compression_results if task_repo.get(r.task_id) == repo]
        )
        repo_compression_order = [label for label in compression_order if label in repo_compression]
        lines.extend([f"### `{repo}` ({len(repo_reports)} tasks)", ""])
        lines.extend(
            _metric_table(
                repo_retrieval_order,
                repo_retrieval,
                layers,
                repo_compression_order,
                repo_compression,
            )
        )
        lines.append("")

    lines.extend(
        _operational_dimensions(
            manifest,
            retrieval_order,
            retrieval_lanes,
            compression_order,
            layers,
        )
    )
    return "\n".join(lines)


def _operational_dimensions(
    manifest: HeadToHeadManifest,
    retrieval_order: list[str],
    retrieval_lanes: dict[str, list[BenchmarkResult]],
    compression_order: list[str],
    layers: dict[str, ComparisonLayerType],
) -> list[str]:
    tool_by_name = {tool.name: tool for tool in manifest.external_tools}
    graphify_by_name = {lane.name.value: lane for lane in manifest.graphify_lanes}
    lines = [
        "## Operational dimensions",
        "",
        "| Lane | Layer | Local/offline posture | Setup steps | Embedder / backend |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in retrieval_order:
        layer = layers.get(label, ComparisonLayerType.RETRIEVAL)
        if label == "archex":
            posture = "local models only" if manifest.archex.local_models_only else "configurable"
            steps = "index build"
            backend = manifest.archex.embedder
        elif label == "raw-ripgrep/read":
            posture = "local files only"
            steps = "none"
            backend = "n/a"
        elif label in tool_by_name:
            tool = tool_by_name[label]
            posture = f"operator-configured (embedder={tool.embedder})"
            steps = f"{len(tool.bootstrap_commands)} bootstrap command(s)"
            backend = tool.embedder
        elif label in graphify_by_name:
            lane = graphify_by_name[label]
            provenance = retrieval_lanes[label][0].provenance
            posture = provenance.get(
                "graphify_local_offline_posture",
                lane.operational_notes or "graph / memory layer",
            )
            steps = (
                "graph build + first query"
                if lane.includes_build_cost
                else "warm query on prebuilt graph"
            )
            backend = provenance.get("graphify_backend", "n/a")
        else:
            posture = "local models only"
            steps = "index build"
            backend = manifest.archex.embedder
        lines.append(f"| {label} | {layer.value} | {posture} | {steps} | {backend} |")
    for label in compression_order:
        layer = layers.get(label, ComparisonLayerType.COMPRESSION)
        lines.append(f"| {label} | {layer.value} | post-selection compression layer | n/a | n/a |")
    lines.extend(
        [
            "",
            "Language coverage and qualitative operational complexity are documented in "
            "`docs/ARCHEX_VS_COCOINDEX.md`; they are not claimed as benchmark numbers here.",
        ]
    )
    return lines
