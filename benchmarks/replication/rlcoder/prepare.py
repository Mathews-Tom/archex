#!/usr/bin/env python
"""Materialise RLCoder's own reference setup, pinned, for the S0 replication gate.

This clones the upstream harness at a fixed commit, applies the four permitted
portability edits enumerated in `.docs/spikes/S0-replication-gate.md`, builds the
tree-sitter parser the metric needs for this platform, and downloads the pinned
dataset and model weights. It runs nothing and measures nothing.

The upstream code is never vendored into this repository. What is checked in is
the pin, the patch, and the commands.

Usage:
    python benchmarks/replication/rlcoder/prepare.py --work-dir /tmp/s0-rlcoder
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS_REPO = "https://github.com/DeepSoftwareAnalytics/RLCoder"
HARNESS_COMMIT = "164d8d88cde324a38f5da70c4f858cc4679ef08e"
DATASET_REPO = "nov3630/Data4RLCoder"
DATASET_SPLIT = "repoeval/line_level"
RETRIEVER_RL = "nov3630/RLRetriever"
RETRIEVER_BASE = "microsoft/unixcoder-base"
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
# Fetched revisions, not just recorded ones. Without an explicit revision a rerun
# silently takes whatever is at HEAD and reports success against a different SHA.
DATASET_REVISION = "cb9639f20f2374d75e5b0b8e8650f1b20802bf9f"
MODEL_REVISIONS = {
    "retriever_rl": "ec587f5d8635462fff3cfb06f9a946148acda08b",
    "retriever_base": "5604afdc964f6c53782a6813140ade5216b99006",
    "generator": "c919139c3a9b4070729c8b2cca4847ab29ca8d94",
}
TREE_SITTER_PYTHON = "https://github.com/tree-sitter/tree-sitter-python"
TREE_SITTER_PYTHON_TAG = "v0.20.4"
PATCH = Path(__file__).with_name("portability.patch")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a command, failing loudly. A silent prepare step is worse than none."""
    result = subprocess.run(command, cwd=cwd, check=False)  # noqa: S603
    if result.returncode != 0:
        msg = f"command failed with {result.returncode}: {' '.join(command)}"
        raise SystemExit(msg)


def clone_harness(work_dir: Path) -> Path:
    harness = work_dir / "RLCoder"
    if harness.exists():
        shutil.rmtree(harness)
    _run(["git", "clone", "--quiet", HARNESS_REPO, str(harness)])
    _run(["git", "checkout", "--quiet", HARNESS_COMMIT], cwd=harness)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=harness, text=True).strip()  # noqa: S603, S607
    if head != HARNESS_COMMIT:
        msg = f"harness checkout drifted: expected {HARNESS_COMMIT}, got {head}"
        raise SystemExit(msg)
    _run(["git", "apply", str(PATCH)], cwd=harness)
    return harness


def build_parser(harness: Path, work_dir: Path) -> Path:
    """Build the tree-sitter Python parser for this platform.

    The upstream repository ships prebuilt x86-64 ELF objects, which will not
    load on other platforms. The grammar and its tag are the same; only the
    object is rebuilt, so the metric is unchanged.
    """
    from tree_sitter import Language

    grammar = work_dir / "tree-sitter-python"
    if not grammar.exists():
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                TREE_SITTER_PYTHON_TAG,
                TREE_SITTER_PYTHON,
                str(grammar),
            ]
        )
    target = harness / "utils" / "build" / "python-lang-parser.so"
    Language.build_library(str(target), [str(grammar)])
    return target


def fetch_assets(work_dir: Path, *, allow_unpinned: bool) -> dict[str, str]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    def _verify(role: str, actual: str, expected: str | None) -> str:
        if expected is not None and actual != expected:
            msg = f"{role} resolved to {actual}, expected the pinned {expected}"
            raise SystemExit(msg)
        return actual

    dataset_revision = None if allow_unpinned else DATASET_REVISION
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=dataset_revision,
        local_dir=str(work_dir / "data"),
        allow_patterns=[f"{DATASET_SPLIT}/*"],
    )
    pins: dict[str, str] = {
        "dataset": _verify("dataset", str(api.dataset_info(DATASET_REPO).sha), dataset_revision)
    }
    for role, repo in (
        ("retriever_rl", RETRIEVER_RL),
        ("retriever_base", RETRIEVER_BASE),
        ("generator", GENERATOR),
    ):
        expected = None if allow_unpinned else MODEL_REVISIONS[role]
        snapshot_download(
            repo,
            revision=expected,
            local_dir=str(work_dir / "models" / repo.split("/")[-1]),
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
        )
        pins[role] = _verify(role, str(api.model_info(repo).sha), expected)
    return pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Fetch HuggingFace assets at HEAD instead of the recorded revisions.",
    )
    args = parser.parse_args()

    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    harness = clone_harness(work_dir)
    build_parser(harness, work_dir)
    pins = fetch_assets(work_dir, allow_unpinned=args.allow_unpinned)
    pins["harness"] = HARNESS_COMMIT

    rendered = json.dumps(pins, indent=2, sort_keys=True)
    (work_dir / "pins.json").write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
