"""M5 usability evidence regression test.

Mirrors `scripts/m5_explorer_usability_evidence.py` as a standing regression
test: proves the explorer's Diff Review and Target Neighborhood views
surface the objectively correct file/importers for
`tests/fixtures/impact_diff`'s documented `hub.py` scenario (see
`docs/EXPLORER_USABILITY_EVIDENCE.md` for the full protocol and measured
evidence).
"""

from __future__ import annotations

import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from archex.api import index_repository
from archex.config import load_config, load_index_config
from archex.explorer.loader import ExplorerData
from archex.explorer.server import ExplorerServer, create_server
from archex.graph_artifact import build_arch_graph_from_store
from archex.models import RepoSource
from archex.report.artifact import build_analysis_artifact

EXPECTED_IMPORTERS = frozenset({"leaf.py", "consumer_a.py", "consumer_b.py", "consumer_c.py"})


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


@pytest.fixture
def running_explorer_for_impact_diff(impact_diff_repo: Path) -> Iterator[ExplorerServer]:
    _edit_hub(impact_diff_repo)
    repo_source = RepoSource(local_path=str(impact_diff_repo))
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)
    store = index_repository(repo_source, config=config, index_config=index_config)
    try:
        graph = build_arch_graph_from_store(store, repo_root=impact_diff_repo)
    finally:
        store.close()
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    data = ExplorerData(artifact=artifact, graph=graph)

    server = create_server(data, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        return response.read().decode("utf-8")


def test_diff_review_surfaces_the_correct_changed_file(
    running_explorer_for_impact_diff: ExplorerServer,
) -> None:
    server = running_explorer_for_impact_diff
    port = server.server_address[1]

    html = _get(f"http://127.0.0.1:{port}/view/diff?token={server.token}")

    assert "hub.py" in html


def test_target_neighborhood_surfaces_all_real_importers(
    running_explorer_for_impact_diff: ExplorerServer,
) -> None:
    server = running_explorer_for_impact_diff
    port = server.server_address[1]

    html = _get(f"http://127.0.0.1:{port}/view/neighborhood?node=file:hub.py&token={server.token}")

    for importer in EXPECTED_IMPORTERS:
        assert importer in html
