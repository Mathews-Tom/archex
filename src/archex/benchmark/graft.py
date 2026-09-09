"""Pinned Graft adapter for the graph-memory comparison lanes.

Graft (`@nanonets/graft`) is a graph-memory tool: `graft build` writes a
persistent structural code graph, `graft ask` answers against it. It joins the
public head-to-head harness through the tool-neutral contract in
:mod:`archex.benchmark.graph_memory`, plus the pinned identity and output
normalization here.

Everything in this module follows the frozen protocol in
`benchmarks/preregistrations/R19-graft-graph-memory-comparison.md`:

* structural mode only — `build` without `--deep`, so no model is called, no API
  key is required, and no spend occurs;
* the graph directory lives outside the task repository and `--no-gitignore
  --no-ignore` are passed, so the checkout the other lanes measure stays pristine;
* every measured query passes `--no-refresh`, because a default `graft ask`
  silently repairs graph drift and would fold synchronization into query latency;
* freshness is read only from `check --json`'s `graph` section, because `context`
  is permanently absent in structural mode;
* rank is the emitted `hits` order, never a re-sort by `hits[].score`, which is
  not monotonically descending;
* a hit's returned file is the path component of its `pointer`. Symbol hits
  (`<path>:L<start>-L<end>`) carry source; whole-file hits (bare `<path>`) carry
  none even under `--source`, so they count toward required-file recall but
  contribute no returned source.

The adapter fails closed on any artifact that contradicts those facts: no source
is inferred, no cell is dropped, and a cell that could not be measured is
retained as an explicit failure instead of a missing row.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, cast

from pydantic import Field, ValidationError

from archex.benchmark.graph_memory import (
    GraphMemoryAdapterError,
    GraphMemoryArtifact,
    GraphMemoryUnavailableError,
    artifact_digest,
    result_from_graph_memory_artifact,
    validate_graph_memory_cost_semantics,
    validate_graph_memory_identity,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from archex.benchmark.models import (
        BenchmarkResult,
        BenchmarkTask,
        GraphMemoryLaneConfig,
    )

GRAFT_PACKAGE = "@nanonets/graft"
"""Released npm package the comparison is pinned to."""

GRAFT_NPM_INTEGRITY = (
    "sha512-L3E5F1aDYJDCARgfR7O2VaMt8xwO1XNYyHiW2n1WhKnj87gPqoxoZJGNbGXfw6XeA9JSJX3"
    "naA36RZ+jDf4AcQ=="
)
"""npm distribution integrity of `@nanonets/graft@0.16.0`."""

GRAFT_SOURCE_COMMIT = "aa1e2bb0f6326068ac64886da1e67fa25a7804de"
"""Package `gitHead`, equal to the commit tagged `v0.16.0`."""

GRAFT_RESULT_CARDINALITY = 10
"""Frozen `-n` for every measured query; matches the external retrieval lane's limit."""

GRAFT_RANK_BASIS = "emitted_hits_order"
"""Only valid rank basis: `hits[].score` is not monotonically descending."""

GRAFT_FRESHNESS_SOURCE = "check_json_graph"
"""Only valid freshness source: `check --json`'s `context` section is a `--deep` artifact."""

GRAFT_DEPTH_TIER_LANGUAGES = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "tsx",
        "jsx",
        "go",
        "java",
        "kotlin",
        "swift",
        "php",
        "r",
    }
)
"""Languages Graft extracts with a hand-written (native depth-tier) extractor."""


class GraftExtractionTier(StrEnum):
    """Which Graft extractor tier covered a task's languages.

    ``depth`` is the hand-written per-language extractor. ``breadth`` is the
    language-agnostic WASM extractor: signature-only, with no receiver typing.
    The two are never presented as equivalent coverage.
    """

    DEPTH = "depth"
    BREADTH = "breadth"


class GraftCellStatus(StrEnum):
    """Whether a planned Graft cell produced a measurement or a recorded failure."""

    OK = "ok"
    FAILED = "failed"


class GraftArtifact(GraphMemoryArtifact):
    """Schema for one ``{task_id}.json`` Graft lane artifact.

    Adds the pinned released-package identity, the protocol posture that had to
    hold while the cell ran, and the returned-unit split that keeps whole-file
    hits out of the returned-source counts.
    """

    graft_package: str = GRAFT_PACKAGE
    graft_version: str
    npm_integrity: str
    source_commit: str
    status: GraftCellStatus = GraftCellStatus.OK
    failure_reason: str | None = None
    timing_mode: str
    extraction_tier: GraftExtractionTier
    query_mode: str
    rank_basis: str = GRAFT_RANK_BASIS
    freshness_source: str = GRAFT_FRESHNESS_SOURCE
    result_cardinality: int = Field(ge=1)
    deep_summaries: bool = False
    no_refresh: bool = True
    graph_dir_outside_repo: bool = True
    telemetry_disabled: bool = True
    returned_units: int = Field(ge=0)
    symbol_hits: int = Field(ge=0)
    whole_file_hits: int = Field(ge=0)
    returned_source_units: int = Field(ge=0)
    graph_ok: bool = False
    graph_nodes: int = Field(default=0, ge=0)
    output_digest: str


@dataclass(frozen=True)
class GraftHit:
    """One ``hits[]`` entry, at the rank Graft emitted it."""

    rank: int
    pointer: str
    path: str
    start_line: int | None
    end_line: int | None
    title: str | None
    code: str | None

    @property
    def is_whole_file(self) -> bool:
        """A bare ``<path>`` pointer names a file, not a symbol span."""
        return self.start_line is None


@dataclass(frozen=True)
class GraftAskOutput:
    """Normalized `graft ask --json` output for one measured query."""

    query_mode: str
    hits: tuple[GraftHit, ...]
    result_files: tuple[str, ...]
    digest: str

    @property
    def symbol_hits(self) -> int:
        return sum(1 for hit in self.hits if not hit.is_whole_file)

    @property
    def whole_file_hits(self) -> int:
        return sum(1 for hit in self.hits if hit.is_whole_file)

    @property
    def source_bearing_hits(self) -> tuple[GraftHit, ...]:
        return tuple(hit for hit in self.hits if not hit.is_whole_file and hit.code)


@dataclass(frozen=True)
class GraftGraphFreshness:
    """The `graph` section of `graft check --json`; the only valid freshness signal."""

    ok: bool
    missing: bool
    nodes: int


def graft_extraction_tier(languages: Sequence[str] | None) -> GraftExtractionTier:
    """Return the tier Graft uses for ``languages``.

    Any language outside the hand-written depth tier drops the whole cell to the
    breadth tier, because that is the coverage the answer was actually built from.
    An unlabeled task cannot claim depth coverage.
    """
    normalized = [language.strip().lower() for language in languages or [] if language.strip()]
    if not normalized:
        return GraftExtractionTier.BREADTH
    if all(language in GRAFT_DEPTH_TIER_LANGUAGES for language in normalized):
        return GraftExtractionTier.DEPTH
    return GraftExtractionTier.BREADTH


def _decode_json_object(raw: bytes, *, what: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GraphMemoryAdapterError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GraphMemoryAdapterError(f"{what} must be a JSON object")
    return cast("dict[str, object]", payload)


def _split_pointer(pointer: str) -> tuple[str, int | None, int | None]:
    """Split a Graft pointer into its path and optional 1-indexed line span."""
    path, separator, span = pointer.rpartition(":")
    if not separator:
        return pointer, None, None
    start, dash, end = span.partition("-")
    if not dash or not start.startswith("L") or not end.startswith("L"):
        return pointer, None, None
    try:
        return path, int(start[1:]), int(end[1:])
    except ValueError:
        return pointer, None, None


def parse_graft_ask_output(raw: bytes) -> GraftAskOutput:
    """Normalize `graft ask --json` bytes into ranked hits and returned files.

    Rank is the emitted order. A hit whose pointer names no repository path is a
    non-attributable answer and fails the cell rather than being inferred away.
    """
    payload = _decode_json_object(raw, what="graft ask output")
    raw_hits = payload.get("hits", [])
    if not isinstance(raw_hits, list):
        raise GraphMemoryAdapterError("graft ask output 'hits' must be a JSON array")

    hits: list[GraftHit] = []
    result_files: list[str] = []
    for rank, entry in enumerate(cast("list[object]", raw_hits), start=1):
        if not isinstance(entry, dict):
            raise GraphMemoryAdapterError(f"graft ask hit {rank} must be a JSON object")
        hit_fields = cast("dict[str, object]", entry)
        raw_pointer = hit_fields.get("pointer")
        pointer = raw_pointer.strip() if isinstance(raw_pointer, str) else ""
        if not pointer:
            raise GraphMemoryAdapterError(
                f"graft ask hit {rank} has no pointer; source and paths are never inferred"
            )
        path, start_line, end_line = _split_pointer(pointer)
        if not path:
            raise GraphMemoryAdapterError(
                f"graft ask hit {rank} pointer {pointer!r} names no repository path"
            )
        code = hit_fields.get("code")
        title = hit_fields.get("title")
        hits.append(
            GraftHit(
                rank=rank,
                pointer=pointer,
                path=path,
                start_line=start_line,
                end_line=end_line,
                title=title if isinstance(title, str) else None,
                code=code if isinstance(code, str) and code else None,
            )
        )
        if path not in result_files:
            result_files.append(path)

    query_mode = payload.get("mode")
    return GraftAskOutput(
        query_mode=query_mode if isinstance(query_mode, str) and query_mode else "unknown",
        hits=tuple(hits),
        result_files=tuple(result_files),
        digest=artifact_digest(raw),
    )


def parse_graft_check_output(raw: bytes) -> GraftGraphFreshness:
    """Read freshness from the `graph` section of `graft check --json`."""
    payload = _decode_json_object(raw, what="graft check output")
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        raise GraphMemoryAdapterError(
            "graft check output has no 'graph' section; 'context' is a --deep artifact "
            "and is never a valid freshness source here"
        )
    graph_fields = cast("dict[str, object]", graph)
    nodes = graph_fields.get("nodes")
    return GraftGraphFreshness(
        ok=graph_fields.get("ok") is True,
        missing=graph_fields.get("missing") is not False,
        nodes=nodes if isinstance(nodes, int) and not isinstance(nodes, bool) else 0,
    )


def _validate_artifact(
    config: GraphMemoryLaneConfig,
    artifact: GraftArtifact,
    *,
    source: str,
) -> None:
    lane = config.name
    validate_graph_memory_identity(
        config,
        artifact,
        source=source,
        package=artifact.graft_package,
        version=artifact.graft_version,
    )
    if artifact.npm_integrity != GRAFT_NPM_INTEGRITY:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} npm_integrity {artifact.npm_integrity!r} "
            f"does not match the pinned released artifact"
        )
    if artifact.source_commit != GRAFT_SOURCE_COMMIT:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} source_commit {artifact.source_commit!r} "
            f"does not match the pinned {GRAFT_SOURCE_COMMIT!r}"
        )
    if artifact.timing_mode != config.mode.value:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} timing_mode {artifact.timing_mode!r} "
            f"does not match lane mode {config.mode.value!r}"
        )
    if artifact.rank_basis != GRAFT_RANK_BASIS:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} rank_basis must be {GRAFT_RANK_BASIS!r}; "
            "hits[].score is not monotonically descending"
        )
    if artifact.freshness_source != GRAFT_FRESHNESS_SOURCE:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} freshness_source must be {GRAFT_FRESHNESS_SOURCE!r}"
        )
    if artifact.result_cardinality != GRAFT_RESULT_CARDINALITY:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} result_cardinality "
            f"{artifact.result_cardinality} does not match the frozen "
            f"{GRAFT_RESULT_CARDINALITY}"
        )
    if artifact.deep_summaries:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} ran paid --deep summaries, which are out of scope"
        )
    if not artifact.no_refresh:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} must be measured with --no-refresh; a default "
            "ask repairs graph drift and folds synchronization into query latency"
        )
    if not artifact.graph_dir_outside_repo:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} must keep its graph directory outside the "
            "task repository so the measured checkout stays pristine"
        )
    if not artifact.telemetry_disabled:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} must run with telemetry disabled "
            "(CI=1 and DO_NOT_TRACK=1)"
        )
    if artifact.symbol_hits + artifact.whole_file_hits != artifact.returned_units:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} returned_units {artifact.returned_units} does not "
            f"equal symbol_hits {artifact.symbol_hits} + whole_file_hits "
            f"{artifact.whole_file_hits}"
        )
    if artifact.returned_source_units > artifact.symbol_hits:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} reports more returned_source_units than symbol "
            "hits; whole-file hits carry no source even under --source"
        )
    if len(artifact.output_digest) != 64 or any(
        char not in "0123456789abcdef" for char in artifact.output_digest
    ):
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} output_digest must be a lowercase SHA-256 digest"
        )

    if artifact.status is GraftCellStatus.FAILED:
        _validate_failed_cell(artifact, lane=lane, source=source)
        return
    if artifact.failure_reason is not None:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} is status 'ok' but records a failure_reason"
        )
    if bool(artifact.result_files) != (artifact.returned_units > 0):
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} returned_units {artifact.returned_units} and "
            f"{len(artifact.result_files)} returned file(s) disagree"
        )
    validate_graph_memory_cost_semantics(config, artifact, source=source)


def _validate_failed_cell(artifact: GraftArtifact, *, lane: str, source: str) -> None:
    """A recorded failure is retained and counted; it must not look like a measurement."""
    if not (artifact.failure_reason or "").strip():
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} is status 'failed' but records no failure_reason"
        )
    if artifact.result_files or artifact.required_files_present:
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} is status 'failed' but reports returned files"
        )
    scored = (
        artifact.recall,
        artifact.precision,
        artifact.required_file_recall,
        artifact.token_efficiency,
        artifact.cold_start_ms,
        artifact.warm_latency_ms,
    )
    if any(value != 0.0 for value in scored):
        raise GraphMemoryAdapterError(
            f"graft lane {lane!r} {source} is status 'failed' but reports non-zero metrics"
        )


def _extra_provenance(artifact: GraftArtifact) -> dict[str, str]:
    return {
        "graft_package": artifact.graft_package,
        "graft_version": artifact.graft_version,
        "graft_npm_integrity": artifact.npm_integrity,
        "graft_source_commit": artifact.source_commit,
        "graft_status": artifact.status.value,
        "graft_timing_mode": artifact.timing_mode,
        "graft_extraction_tier": artifact.extraction_tier.value,
        "graft_query_mode": artifact.query_mode,
        "graft_rank_basis": artifact.rank_basis,
        "graft_freshness_source": artifact.freshness_source,
        "graft_result_cardinality": str(artifact.result_cardinality),
        "graft_deep_summaries": str(artifact.deep_summaries).lower(),
        "graft_no_refresh": str(artifact.no_refresh).lower(),
        "graft_telemetry_disabled": str(artifact.telemetry_disabled).lower(),
        "graft_returned_units": str(artifact.returned_units),
        "graft_symbol_hits": str(artifact.symbol_hits),
        "graft_whole_file_hits": str(artifact.whole_file_hits),
        "graft_returned_source_units": str(artifact.returned_source_units),
        "graft_graph_nodes": str(artifact.graph_nodes),
        "graft_output_digest": artifact.output_digest,
        **(
            {"graft_failure_reason": artifact.failure_reason}
            if artifact.failure_reason is not None
            else {}
        ),
    }


def _result_from_artifact(
    config: GraphMemoryLaneConfig,
    artifact: GraftArtifact,
    *,
    run_mode: str,
    artifact_path: Path | None = None,
    digest: str | None = None,
) -> BenchmarkResult:
    return result_from_graph_memory_artifact(
        config,
        artifact,
        run_mode=run_mode,
        package=artifact.graft_package,
        version=artifact.graft_version,
        extra_provenance=_extra_provenance(artifact),
        artifact_path=artifact_path,
        digest=digest,
    )


def load_graft_artifact(
    config: GraphMemoryLaneConfig,
    *,
    task_id: str,
    artifact_dir: Path,
    expected_tier: GraftExtractionTier | None = None,
) -> BenchmarkResult:
    """Import one operator-produced Graft lane artifact for ``task_id``."""
    artifact_path = artifact_dir / f"{task_id}.json"
    if not artifact_path.is_file():
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} artifact not found: {artifact_path}"
        )
    raw = artifact_path.read_bytes()
    try:
        artifact = GraftArtifact.model_validate_json(raw)
    except ValidationError as exc:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} artifact {artifact_path} is invalid: {exc}"
        ) from exc
    if artifact.task_id != task_id:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} artifact {artifact_path} has task_id "
            f"{artifact.task_id!r}, expected {task_id!r}"
        )
    _validate_artifact(config, artifact, source=f"artifact {artifact_path}")
    if expected_tier is not None and artifact.extraction_tier is not expected_tier:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} artifact {artifact_path} records extraction_tier "
            f"{artifact.extraction_tier.value!r}, but the task's languages resolve to "
            f"{expected_tier.value!r}; a mislabeled tier corrupts the depth-tier subset"
        )
    return _result_from_artifact(
        config,
        artifact,
        run_mode="artifact",
        artifact_path=artifact_path,
        digest=artifact_digest(raw),
    )


def run_graft_lane(
    config: GraphMemoryLaneConfig,
    *,
    task: BenchmarkTask,
    repo_path: Path | None,
    graph_dir: Path | None = None,
) -> BenchmarkResult:
    """Produce one Graft lane result via artifact import or local command mode.

    Local mode runs the configured operator command, which owns the frozen
    protocol and writes one Graft artifact JSON to stdout. ``graph_dir`` is
    passed through so the warm lane queries the graph the cold lane built, and
    it must live outside ``repo_path``.
    """
    if config.artifact_dir is not None:
        return load_graft_artifact(
            config,
            task_id=task.task_id,
            artifact_dir=Path(config.artifact_dir),
            expected_tier=graft_extraction_tier(task.languages),
        )
    if which(config.command) is None:
        raise GraphMemoryUnavailableError(
            f"graft lane {config.name!r} command {config.command!r} not found; "
            "set graph_memory_lanes[].artifact_dir to import operator artifacts instead"
        )
    if repo_path is None:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} local execution requires repo_path"
        )
    if graph_dir is None:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} local execution requires graph_dir outside the "
            "task repository"
        )
    if (
        graph_dir.resolve() == repo_path.resolve()
        or repo_path.resolve() in graph_dir.resolve().parents
    ):
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} graph_dir {graph_dir} is inside the task "
            "repository; the measured checkout must stay pristine"
        )

    payload = {
        "task": task.model_dump(mode="json"),
        "repo_path": str(repo_path),
        "graph_dir": str(graph_dir),
        "lane": config.name,
        "tool": config.tool.value,
        "mode": config.mode.value,
        "graft": {
            "package_name": config.package_name,
            "version": config.version,
            "npm_integrity": GRAFT_NPM_INTEGRITY,
            "source_commit": GRAFT_SOURCE_COMMIT,
            "result_cardinality": GRAFT_RESULT_CARDINALITY,
        },
    }
    command = [config.command, *config.args]
    env = {**os.environ, "CI": "1", "DO_NOT_TRACK": "1", **config.env}
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} timed out after {config.timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} failed with exit {completed.returncode}: {detail}"
        )
    try:
        artifact = GraftArtifact.model_validate_json(completed.stdout)
    except ValidationError as exc:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} local output is invalid: {exc}"
        ) from exc
    if artifact.task_id != task.task_id:
        raise GraphMemoryAdapterError(
            f"graft lane {config.name!r} local output task_id {artifact.task_id!r} "
            f"does not match benchmark task {task.task_id!r}"
        )
    _validate_artifact(config, artifact, source="local output")
    return _result_from_artifact(config, artifact, run_mode="local")
