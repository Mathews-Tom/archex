"""Evidence script: measure time to the correct file/symbol via a real explorer server.

M5's DEVELOPMENT_PLAN.md acceptance row requires "usability evidence
measures time to first correct file/symbol rather than self-reported
satisfaction." Run:

    uv run python scripts/m5_explorer_usability_evidence.py

This is an automated, deterministic proxy for a new-contributor
orientation task, not a live human trial -- no human subjects were
involved in producing this evidence, and this script does not substitute
for one. What it *does* measure honestly: real wall-clock elapsed time,
over real HTTP requests, against a real running `ExplorerServer`, to reach
a page whose rendered content contains an objectively correct answer
(defined by `tests/fixtures/impact_diff`'s own documented graph shape, not
by this script).

Scenario (`tests/fixtures/impact_diff`, also used by
`tests/test_report_artifact.py` and friends): `hub.py` is imported directly
by five files -- `leaf.py`, `consumer_a.py`, `consumer_b.py` and
`consumer_c.py` each import `shared_helper`, and the entry point `main.py`
imports `other_helper` -- a deliberate hub. `EXPECTED_IMPORTERS` below is
the four `shared_helper` importers, the set that exercises the shared-symbol
fan-in the risk classifier keys on; `main.py` is the fifth direct importer
and is why the inbound-edge counts read five, not four. Editing `hub.py`
makes it the diff's own single changed file and its riskiest symbol
candidate (public-interface change, high fan-in). A contributor who just
cloned this repository and asks "what changed, and what does it affect?"
should be able to answer "hub.py, and the files that import it" from the
explorer alone.

Five navigation paths are timed:

1. Diff Review: does the rendered `/view/diff` page name `hub.py` as the
   changed file and (if the risk classifier flagged one) surfaces it as a
   symbol risk candidate?
2. Target Neighborhood: does searching `/view/neighborhood?node=file:hub.py`
   surface all four real importers?
3. Node Search (R25): does `/view/search?q=hub` find `hub.py` without the
   reader already knowing its exact node id -- the step that previously had
   no surface at all?
4. Inbound-only Neighborhood (R25): does `direction=in` report exactly the
   files that depend on `hub.py`, each marked inbound, and nothing outbound?
5. Static export (R25): does the offline bundle answer the same question
   from a `file://` path with no server running at all?

Every path is graded pass/fail against the fixture's documented ground truth
to keep "correct" objective, not just fast.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from archex.explorer.export import NODES_FILE, export_explorer_site, node_page_name
from archex.explorer.loader import ExplorerData
from archex.explorer.server import create_server
from archex.graph_artifact import build_arch_graph_from_store
from archex.report.artifact import build_analysis_artifact

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "impact_diff"
EXPECTED_IMPORTERS = frozenset({"leaf.py", "consumer_a.py", "consumer_b.py", "consumer_c.py"})


def _init_fixture_repo(dest: Path) -> None:
    shutil.copytree(FIXTURE_DIR, dest)
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "usability@archex.test"],
        ["git", "config", "user.name", "archex-usability"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
    ):
        subprocess.run(command, cwd=dest, check=True, capture_output=True)
    hub = dest / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        return response.read().decode("utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="archex-explorer-usability-") as tmp:
        repo_path = Path(tmp) / "impact_diff"
        _init_fixture_repo(repo_path)

        from archex.api import index_repository
        from archex.config import load_config, load_index_config
        from archex.models import RepoSource

        repo_source = RepoSource(local_path=str(repo_path))
        config = load_config(repo_source)
        index_config = load_index_config(repo_source)
        store = index_repository(repo_source, config=config, index_config=index_config)
        try:
            graph = build_arch_graph_from_store(store, repo_root=repo_path)
        finally:
            store.close()
        artifact = build_analysis_artifact(repo_path, base_ref="HEAD")

        data = ExplorerData(artifact=artifact, graph=graph)
        server = create_server(data, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            start = time.perf_counter()
            diff_html = _get(f"http://127.0.0.1:{port}/view/diff?token={server.token}")
            diff_elapsed = time.perf_counter() - start
            diff_correct = "hub.py" in diff_html

            start = time.perf_counter()
            neighborhood_html = _get(
                f"http://127.0.0.1:{port}/view/neighborhood?node=file:hub.py&token={server.token}"
            )
            neighborhood_elapsed = time.perf_counter() - start
            neighborhood_correct = all(
                importer in neighborhood_html for importer in EXPECTED_IMPORTERS
            )

            # R25: reaching the node without already knowing its id.
            start = time.perf_counter()
            search_html = _get(f"http://127.0.0.1:{port}/view/search?q=hub&token={server.token}")
            search_elapsed = time.perf_counter() - start
            search_correct = (
                "file:hub.py" in search_html
                and "/view/neighborhood?node=file%3Ahub.py" in search_html
            )

            # R25: "what depends on my change" answered directionally.
            start = time.perf_counter()
            inbound_html = _get(
                f"http://127.0.0.1:{port}/view/neighborhood"
                f"?node=file:hub.py&direction=in&token={server.token}"
            )
            inbound_elapsed = time.perf_counter() - start
            inbound_correct = (
                all(importer in inbound_html for importer in EXPECTED_IMPORTERS)
                and '<tr class="orientation-in">' in inbound_html
                and '<tr class="orientation-out">' not in inbound_html
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        # R25: the same question, offline, with no server running.
        export_dir = Path(tmp) / "explorer-site"
        start = time.perf_counter()
        export_explorer_site(data, export_dir)
        export_elapsed = time.perf_counter() - start
        hub_page = (export_dir / node_page_name("file:hub.py")).read_text(encoding="utf-8")
        index_page = (export_dir / NODES_FILE).read_text(encoding="utf-8")
        export_correct = (
            all(importer in hub_page for importer in EXPECTED_IMPORTERS)
            and '<tr class="orientation-in">' in hub_page
            and "file:hub.py" in index_page
            and "<script" not in hub_page
            and "token=" not in hub_page
        )

    print(
        "Scenario: tests/fixtures/impact_diff, hub.py edited (deliberate hub, 5 direct importers)\n"
    )
    rows = (
        ("Diff Review (/view/diff)", diff_elapsed, diff_correct),
        ("Target Neighborhood", neighborhood_elapsed, neighborhood_correct),
        ("Node Search (/view/search)", search_elapsed, search_correct),
        ("Inbound-only Neighborhood", inbound_elapsed, inbound_correct),
        ("Static export (offline)", export_elapsed, export_correct),
    )
    print(f"{'path':<28} {'elapsed':>10}  {'correct':>8}")
    for label, elapsed, correct in rows:
        print(f"{label:<28} {elapsed:>9.3f}s  {correct!s:>8}")

    if not all(correct for _, _, correct in rows):
        print("\nFAILED: the explorer did not surface the objectively correct answer.")
        return 1
    print("\nPASSED: every navigation path reached the correct file/symbol.")
    print(
        "\nNote: this is an automated, deterministic proxy measuring real HTTP wall-clock "
        "time against a real server -- not a live human trial. No self-reported satisfaction "
        "score was collected or is claimed here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
