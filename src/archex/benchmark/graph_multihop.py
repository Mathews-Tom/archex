"""Bounded multi-hop dependency-graph expansion for the graph-multihop lane.

Explores the file dependency graph outward from seed files with explicit caps so
expansion can never flood the bundle: an edge-confidence threshold suppresses
low-confidence edges, a per-hop frontier cap limits breadth, a hop cap limits
depth, and a token budget limits total expansion cost. Every decision — each
expanded file and each cut (low-confidence / frontier / budget) — is recorded so
receipts and provenance can explain the expansion. Pure and deterministic: it
operates on a plain edge list, so it needs neither a graph object nor a store.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class GraphEdge:
    """A directed dependency edge with an extraction-confidence score in [0, 1]."""

    source: str
    target: str
    confidence: float


@dataclass(frozen=True)
class MultihopCaps:
    """Hard caps that bound multi-hop expansion against dependency flooding."""

    hop_cap: int = 2
    frontier_cap: int = 8
    confidence_threshold: float = 0.5
    token_budget: int = 1_000_000


class ExpansionAction(StrEnum):
    """Outcome recorded for one candidate file during expansion."""

    EXPANDED = "expanded"
    SUPPRESSED_LOW_CONFIDENCE = "suppressed_low_confidence"
    CUT_FRONTIER = "cut_frontier"
    CUT_BUDGET = "cut_budget"


@dataclass(frozen=True)
class ExpansionDecision:
    """One candidate file's expansion outcome, with the edge that proposed it."""

    file: str
    hop: int
    via: str
    confidence: float
    tokens: int
    action: ExpansionAction


@dataclass(frozen=True)
class MultihopResult:
    """Files added by expansion plus the full ordered decision log."""

    added_files: list[str]
    decisions: list[ExpansionDecision]
    hops_run: int

    def action_count(self, action: ExpansionAction) -> int:
        """Number of decisions with the given action."""
        return sum(1 for decision in self.decisions if decision.action is action)

    def expanded_decisions(self) -> list[ExpansionDecision]:
        """Decisions that added a file to the bundle, in expansion order."""
        return [d for d in self.decisions if d.action is ExpansionAction.EXPANDED]


def _undirected_adjacency(edges: list[GraphEdge]) -> dict[str, dict[str, float]]:
    """Build symmetric adjacency keeping the max confidence per (node, neighbour)."""
    adjacency: dict[str, dict[str, float]] = {}
    for edge in edges:
        if edge.source == edge.target:
            continue
        for a, b in ((edge.source, edge.target), (edge.target, edge.source)):
            neighbours = adjacency.setdefault(a, {})
            if edge.confidence > neighbours.get(b, float("-inf")):
                neighbours[b] = edge.confidence
    return adjacency


def graph_multihop_expand(
    edges: list[GraphEdge],
    seed_files: list[str],
    file_tokens: dict[str, int],
    caps: MultihopCaps,
) -> MultihopResult:
    """Expand from seeds hop by hop under confidence, frontier, hop, and budget caps.

    Candidates each hop are ranked by edge confidence (then file name) for
    determinism. A candidate below the confidence threshold is suppressed; beyond
    the per-hop frontier cap it is a frontier cut; if it would exceed the token
    budget it is a budget cut. Each candidate is decided exactly once (decided
    files are never reconsidered), so expansion always terminates and never floods.
    Only expanded files advance the frontier to the next hop.
    """
    adjacency = _undirected_adjacency(edges)
    visited: set[str] = set(seed_files)
    frontier: list[str] = list(dict.fromkeys(seed_files))
    spent = 0
    added: list[str] = []
    decisions: list[ExpansionDecision] = []
    hops_run = 0

    for hop in range(1, caps.hop_cap + 1):
        if not frontier:
            break
        # Best (max-confidence) proposing edge per unvisited neighbour this hop.
        candidates: dict[str, tuple[str, float]] = {}
        for node in frontier:
            for neighbour, confidence in adjacency.get(node, {}).items():
                if neighbour in visited:
                    continue
                best = candidates.get(neighbour)
                if best is None or confidence > best[1]:
                    candidates[neighbour] = (node, confidence)
        if not candidates:
            break
        hops_run = hop
        ranked = sorted(candidates.items(), key=lambda item: (-item[1][1], item[0]))
        accepted_this_hop = 0
        next_frontier: list[str] = []
        for neighbour, (via, confidence) in ranked:
            visited.add(neighbour)  # decided this hop; never reconsider
            tokens = file_tokens.get(neighbour, 0)
            if confidence < caps.confidence_threshold:
                action = ExpansionAction.SUPPRESSED_LOW_CONFIDENCE
            elif accepted_this_hop >= caps.frontier_cap:
                action = ExpansionAction.CUT_FRONTIER
            elif spent + tokens > caps.token_budget:
                action = ExpansionAction.CUT_BUDGET
            else:
                action = ExpansionAction.EXPANDED
                spent += tokens
                accepted_this_hop += 1
                added.append(neighbour)
                next_frontier.append(neighbour)
            decisions.append(
                ExpansionDecision(
                    file=neighbour,
                    hop=hop,
                    via=via,
                    confidence=confidence,
                    tokens=tokens,
                    action=action,
                )
            )
        frontier = next_frontier

    return MultihopResult(added_files=added, decisions=decisions, hops_run=hops_run)
