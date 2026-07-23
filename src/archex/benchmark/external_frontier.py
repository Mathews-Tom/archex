"""M3 candidate lane matrix: default, cAST, fast, balanced, and only-if-defined symbolic-rerank.

Each lane pairs a benchmark ``Strategy`` with the retrieval-options override
it needs. ``default`` and ``cast`` both run the ``archex_query`` strategy but
differ in ``--chunker`` -- a *global* CLI option, not a per-strategy one --
so they require two separate ``archex benchmark run`` invocations against
two separate evidence directories, compared the same way
``format_chunker_frontier_table`` already compares any two chunker runs.
This module defines the lane matrix and its CLI recipe; an operator
orchestration script (see the M3 promotion-gate stack) drives the actual
invocations and diffs their evidence.

The symbolic-rerank lane is **only-if-defined**: it is never included by
default, so default/cAST/profile evidence is never blocked on it per the M3
constraint "do not block baseline/cAST/profile evidence on undefined
symbolic rerank." An operator opts in explicitly via
``include_symbolic_rerank=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from archex.benchmark.models import BenchmarkRetrievalOptions, ChunkerName, Strategy

#: Profile lanes force their own vector/rerank/module-prefilter toggles via
#: `index_config_for_profile`, but `benchmark_index_config` re-applies
#: certain global CLI flags (splade, module_prefilter, rerank_model) on top
#: of *any* strategy's base IndexConfig regardless of profile -- silently
#: breaking a profile's cost/quality guarantee for the comparison. See
#: `profile_purity_violations`.

_PROFILE_STRATEGIES = frozenset(
    {Strategy.ARCHEX_QUERY_PROFILE_FAST, Strategy.ARCHEX_QUERY_PROFILE_BALANCED}
)


class LaneName(StrEnum):
    """M3 candidate lane identifiers, also used as the lane's evidence subdirectory name."""

    DEFAULT = "default"
    CAST = "cast"
    PROFILE_FAST = "profile_fast"
    PROFILE_BALANCED = "profile_balanced"
    SYMBOLIC_RERANK = "symbolic_rerank"


@dataclass(frozen=True)
class ExternalFrontierLane:
    """One M3 candidate lane: a strategy, an optional chunker override, and its evidence name."""

    name: LaneName
    strategy: Strategy
    chunker: ChunkerName | None = None

    def cli_run_args(self) -> list[str]:
        """Return the ``archex benchmark run`` argv fragment this lane needs.

        Callers append ``--tasks-dir``/``--output``/corpus-access flags
        themselves; this only covers the lane-specific strategy/chunker
        selection so every lane shares one invocation recipe shape.
        """
        args = ["--strategy", self.strategy.value]
        if self.chunker is not None:
            args += ["--chunker", self.chunker]
        return args


DEFAULT_LANE = ExternalFrontierLane(
    name=LaneName.DEFAULT, strategy=Strategy.ARCHEX_QUERY, chunker="default"
)
CAST_LANE = ExternalFrontierLane(name=LaneName.CAST, strategy=Strategy.ARCHEX_QUERY, chunker="cast")
PROFILE_FAST_LANE = ExternalFrontierLane(
    name=LaneName.PROFILE_FAST, strategy=Strategy.ARCHEX_QUERY_PROFILE_FAST
)
PROFILE_BALANCED_LANE = ExternalFrontierLane(
    name=LaneName.PROFILE_BALANCED, strategy=Strategy.ARCHEX_QUERY_PROFILE_BALANCED
)
SYMBOLIC_RERANK_LANE = ExternalFrontierLane(
    name=LaneName.SYMBOLIC_RERANK, strategy=Strategy.ARCHEX_QUERY_SYMBOLIC_RERANK
)


def build_external_frontier_lanes(
    *, include_symbolic_rerank: bool = False
) -> list[ExternalFrontierLane]:
    """Return the M3 candidate lane matrix.

    ``include_symbolic_rerank`` defaults to ``False``: the symbolic-rerank
    lane is only-if-defined, so the default/cAST/profile lanes never depend
    on it being configured.
    """
    lanes = [DEFAULT_LANE, CAST_LANE, PROFILE_FAST_LANE, PROFILE_BALANCED_LANE]
    if include_symbolic_rerank:
        lanes.append(SYMBOLIC_RERANK_LANE)
    return lanes


def lane_strategies(lanes: list[ExternalFrontierLane]) -> list[Strategy]:
    """Return the distinct strategies a lane matrix touches, in lane order."""
    seen: list[Strategy] = []
    for lane in lanes:
        if lane.strategy not in seen:
            seen.append(lane.strategy)
    return seen


def profile_purity_violations(
    options: BenchmarkRetrievalOptions,
    lanes: list[ExternalFrontierLane],
) -> list[str]:
    """Return global retrieval-option fields that would corrupt a profile lane's purity.

    Empty when no lane in ``lanes`` is a profile lane, or when none of the
    purity-sensitive global options are set. A non-empty result means the
    fast/balanced comparison in this run cannot be trusted as measuring the
    product's actual fast/balanced profiles.
    """
    if not any(lane.strategy in _PROFILE_STRATEGIES for lane in lanes):
        return []
    violations: list[str] = []
    if options.splade:
        violations.append("splade")
    if options.module_prefilter:
        violations.append("module_prefilter")
    if options.rerank_model is not None:
        violations.append("rerank_model")
    return violations
