from pathlib import Path

from archex.api import query
from archex.models import Config, PipelineTiming, RepoSource


class TestAPITiming:
    def test_query_timing_phases_populated(self, python_simple_repo: Path) -> None:
        """Ensure all query phases are measured and populate the timing object."""
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=False)
        timing = PipelineTiming()

        # Rely on actual work taking > 0ms
        bundle = query(source, "authentication login", config=config, timing=timing)
        assert bundle is not None

        assert timing.total_ms > 0
        assert timing.acquire_ms >= 0
        assert timing.parse_ms >= 0
        assert timing.index_ms >= 0
        assert timing.search_ms >= 0
        assert timing.assemble_ms >= 0
        assert timing.strategy != ""

    def test_cached_query_timing_phases(self, python_simple_repo: Path) -> None:
        """Ensure cached queries also populate timing phases properly."""
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=True)

        # Warm up cache
        query(source, "user session", config=config)

        # Now run timed cached query
        timing = PipelineTiming()
        bundle = query(source, "user session", config=config, timing=timing)
        assert bundle is not None

        assert timing.total_ms > 0
        assert timing.acquire_ms >= 0
        assert timing.index_ms >= 0
        assert timing.search_ms >= 0
        assert timing.strategy in ("cached", "passthrough")

    def test_cached_search_phase_totals_cover_at_least_95_percent_of_runtime(
        self, python_simple_repo: Path
    ) -> None:
        """M1 acceptance: runtime phases must account for >= 95% of measured total.

        A small ``token_budget`` forces the cached search path (rather than
        passthrough, which trivially covers 100% by construction) so this
        exercises the phase most prone to under-measurement: loading a cached
        index (BM25/graph/chunk hydration) between the cache-hit check and the
        search phase.
        """
        source = RepoSource(local_path=str(python_simple_repo))
        config = Config(cache=True)

        # Warm up cache, then force the non-passthrough cached search path.
        query(source, "user session", config=config, token_budget=200)
        timing = PipelineTiming()
        bundle = query(source, "user session", config=config, timing=timing, token_budget=200)
        assert bundle is not None
        assert timing.strategy == "cached"

        phase_sum = (
            timing.acquire_ms
            + timing.parse_ms
            + timing.index_ms
            + timing.search_ms
            + timing.assemble_ms
        )
        assert timing.total_ms > 0
        assert phase_sum / timing.total_ms >= 0.95
