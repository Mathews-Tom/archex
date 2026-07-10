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
