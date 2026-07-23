"""Deterministic fixed-agent downstream trajectory accounting for M3 candidate lanes.

The M3 evaluation program requires comparing candidate retrieval paths on
downstream agent outcomes, not just retrieval metrics -- but running a real
LLM-driven coding agent per candidate lane would make results
non-deterministic, network-dependent, and expensive to reproduce locally.
This module models one **fixed** (unchanging across every lane), fully
deterministic, offline downstream agent so any outcome difference between
lanes is attributable to retrieval, not agent variance.

The model: given a bundle's missing required files (already computed by
``compute_bundle_completion_penalty`` for every ``archex_query``-family
result), a fixed agent would need to *search* for each one before it could
read it -- unlike the existing oracle-style ``post_bundle_read_turns``
count, which assumes the agent already knows exactly which file to open.
Each missing file costs one search turn, up to a bounded budget: a proxy
for finite agent patience, not an infinite-retry oracle.

This intentionally does **not** change ``task_completion_result`` or any
existing pass/fail semantics -- ``post_bundle_search_turns`` is a purely
additive trajectory-cost metric alongside the existing completion fields.
"""

from __future__ import annotations

#: Bounded search budget: a fixed agent that cannot locate a required file
#: within this many search turns is modeled as exhausting its patience for
#: that file rather than searching indefinitely.
FIXED_AGENT_MAX_SEARCH_TURNS = 3


def compute_fixed_agent_search_turns(
    missing_files: list[str],
    *,
    max_search_turns: int = FIXED_AGENT_MAX_SEARCH_TURNS,
) -> int:
    """Return the fixed agent's search-turn cost for locating missing required files.

    One search turn per missing file, capped at ``max_search_turns``. Zero
    when nothing is missing.
    """
    return min(len(missing_files), max_search_turns)
