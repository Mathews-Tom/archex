"""Tests for head-to-head benchmark manifest loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.benchmark.headtohead import (
    HeadToHeadManifestError,
    comparison_lane_layers,
    load_headtohead_manifest,
    load_headtohead_tasks,
)
from archex.benchmark.models import ComparisonLayerType


def _write_task(path: Path, task_id: str, repo: str, category: str) -> None:
    path.write_text(
        f'''\
task_id: {task_id}
repo: "{repo}"
commit: abc123
category: {category}
question: "How does this work?"
expected_files:
  - src/main.py
''',
        encoding="utf-8",
    )


def _manifest_text(
    task_subset: list[str],
    *,
    candidate_strategies: list[str] | None = None,
    extra: str = "",
) -> str:
    tasks_yaml = "\n".join(f"  - {task_id}" for task_id in task_subset)
    candidates_yaml = ""
    if candidate_strategies:
        rendered = "\n".join(f"    - {name}" for name in candidate_strategies)
        candidates_yaml = f"  candidate_strategies:\n{rendered}\n"
    return f"""\
manifest_version: 1
name: public-c1-comparison
hardware_notes: "Apple M1 Pro, local models only"
task_subset:
{tasks_yaml}
archex:
  strategy: archex_query
{candidates_yaml}  embedder: jina-v2
  local_models_only: true
external_tools:
  - name: ccc
    version: "0.2.35"
    command: ccc
    args: [mcp]
    embedder: Snowflake/snowflake-arctic-embed-xs
    bootstrap_commands:
      - command: ccc
        args: [init, -f]
      - command: ccc
        args: [index]
raw_read_strategy: raw_ripgrep
{extra}"""


_HEADROOM_LAYER_YAML = """\
compression_layers:
  - name: headroom
    version: "0.4.1"
    command: headroom
    args: [compress]
    modes:
      - headroom_only_on_raw_context
      - archex_plus_headroom
    compression_settings:
      profile: balanced
"""


def test_load_headtohead_manifest_accepts_pinned_external_tool(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(_manifest_text(["external_task"]), encoding="utf-8")

    manifest = load_headtohead_manifest(manifest_path)

    assert manifest.name == "public-c1-comparison"
    assert manifest.external_tools[0].version == "0.2.35"
    assert manifest.external_tools[0].embedder == "Snowflake/snowflake-arctic-embed-xs"
    assert len(manifest.external_tools[0].bootstrap_commands) == 2


def test_load_headtohead_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(["external_task"]) + "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(HeadToHeadManifestError, match="unknown field 'unexpected'"):
        load_headtohead_manifest(manifest_path)


def test_load_headtohead_manifest_rejects_unpinned_tool_version(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(["external_task"]).replace('version: "0.2.35"', "version: latest"),
        encoding="utf-8",
    )

    with pytest.raises(HeadToHeadManifestError, match="must pin an exact released version"):
        load_headtohead_manifest(manifest_path)


def test_load_headtohead_tasks_selects_external_subset_in_order(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_task(tasks_dir / "a.yaml", "task_a", "owner/repo", "external-framework")
    _write_task(tasks_dir / "b.yaml", "task_b", "owner/repo", "external-large")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(_manifest_text(["task_b", "task_a"]), encoding="utf-8")

    _, tasks = load_headtohead_tasks(manifest_path, tasks_dir)

    assert [task.task_id for task in tasks] == ["task_b", "task_a"]


def test_load_headtohead_tasks_rejects_self_repo_subset(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_task(tasks_dir / "self.yaml", "self_task", ".", "self")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(_manifest_text(["self_task"]), encoding="utf-8")

    with pytest.raises(HeadToHeadManifestError, match="exclude self-repo"):
        load_headtohead_tasks(manifest_path, tasks_dir)


def test_load_headtohead_manifest_accepts_pinned_compression_layer(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(["external_task"], extra=_HEADROOM_LAYER_YAML),
        encoding="utf-8",
    )

    manifest = load_headtohead_manifest(manifest_path)

    assert manifest.external_tools[0].layer_type is ComparisonLayerType.RETRIEVAL
    layer = manifest.compression_layers[0]
    assert layer.name == "headroom"
    assert layer.version == "0.4.1"
    assert layer.layer_type is ComparisonLayerType.COMPRESSION
    assert layer.compression_settings == {"profile": "balanced"}


def test_load_headtohead_manifest_rejects_unpinned_compression_version(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(
            ["external_task"],
            extra=_HEADROOM_LAYER_YAML.replace('version: "0.4.1"', "version: latest"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HeadToHeadManifestError,
        match="compression_layers.headroom.version must pin an exact released version",
    ):
        load_headtohead_manifest(manifest_path)


def test_load_headtohead_manifest_rejects_compression_layer_labeled_retrieval(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(
            ["external_task"],
            extra=_HEADROOM_LAYER_YAML.replace(
                "    command: headroom\n",
                "    command: headroom\n    layer_type: retrieval\n",
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HeadToHeadManifestError,
        match="compression_layers.headroom.layer_type must be compression",
    ):
        load_headtohead_manifest(manifest_path)


def test_load_headtohead_manifest_rejects_external_tool_labeled_compression(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(["external_task"]).replace(
            "    command: ccc\n",
            "    command: ccc\n    layer_type: compression\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HeadToHeadManifestError,
        match="external_tools.ccc.layer_type must be retrieval",
    ):
        load_headtohead_manifest(manifest_path)


def test_load_headtohead_manifest_accepts_archex_candidate_lanes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(
            ["external_task"],
            candidate_strategies=["archex_query_compressed", "archex_query_efficiency_packed"],
        ),
        encoding="utf-8",
    )

    manifest = load_headtohead_manifest(manifest_path)

    assert [s.value for s in manifest.archex.candidate_strategies] == [
        "archex_query_compressed",
        "archex_query_efficiency_packed",
    ]


def test_load_headtohead_manifest_rejects_non_archex_candidate(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(["external_task"], candidate_strategies=["raw_ripgrep"]),
        encoding="utf-8",
    )

    with pytest.raises(HeadToHeadManifestError, match="must be an archex_query lane"):
        load_headtohead_manifest(manifest_path)


def test_comparison_lane_layers_labels_each_lane(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _manifest_text(
            ["external_task"],
            candidate_strategies=["archex_query_compressed"],
            extra=_HEADROOM_LAYER_YAML,
        ),
        encoding="utf-8",
    )

    layers = comparison_lane_layers(load_headtohead_manifest(manifest_path))

    assert layers["archex"] is ComparisonLayerType.RETRIEVAL
    assert layers["archex_query_compressed"] is ComparisonLayerType.RETRIEVAL
    assert layers["ccc"] is ComparisonLayerType.RETRIEVAL
    assert layers["headroom_only_on_raw_context"] is ComparisonLayerType.COMPRESSION
    assert layers["archex_plus_headroom"] is ComparisonLayerType.COMPRESSION
    assert layers["raw-ripgrep/read"] is ComparisonLayerType.BASELINE
