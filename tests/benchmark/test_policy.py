import os
from pathlib import Path

from archex.benchmark.models import BenchmarkProvenance, BenchmarkReport


def test_sealed_corpus_boundaries_no_benchmark_specialization() -> None:
    """Ensure production code does not hardcode benchmark-specific vocabulary."""
    # Production code must never specialize behavior to known benchmark tasks/repos.
    banned_terms = [
        "swe_bench",
        "swebench",
        "human_eval",
        "humaneval",
        "mbpp",
        "bird",
        "spider",
        "defects4j",
    ]

    src_dir = Path("src/archex")
    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = Path(root) / file

            # The benchmark engine itself is allowed to know about benchmarks.
            if "benchmark" in path.parts:
                continue

            content = path.read_text().lower()
            for term in banned_terms:
                assert term not in content, (
                    f"Sealed corpus violation: benchmark-specific term '{term}' "
                    f"found in production file {path}. Production behavior must "
                    f"generalize heuristically, not specialize to benchmarks."
                )


def test_benchmark_report_provenance_populated() -> None:
    """Ensure BenchmarkReport can record full reproducible provenance."""
    prov = BenchmarkProvenance(
        archex_version="1.0.0",
        generation_time="2026-07-11T00:00:00Z",
        hardware="darwin-arm64",
        config={"feature_flag": True},
        sample_count=5,
        commit="a1b2c3d4",
    )
    report = BenchmarkReport(
        task_id="t1",
        repo="r1",
        question="q1",
        results=[],
        baseline_tokens=100,
        provenance=prov,
    )

    assert report.provenance is not None
    assert report.provenance.archex_version == "1.0.0"
    assert report.provenance.hardware == "darwin-arm64"
    assert report.provenance.commit == "a1b2c3d4"
    assert report.provenance.sample_count == 5
