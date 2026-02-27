"""Architecture decision inference: derive ArchDecision records from patterns and structure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.exceptions import ProviderError
from archex.models import ArchDecision, DetectedPattern, Interface, Module

if TYPE_CHECKING:
    from archex.providers.base import LLMProvider

# Alternatives and implications for known pattern names
_PATTERN_ALTERNATIVES: dict[str, list[str]] = {
    "middleware_chain": ["Direct function composition", "Decorator pattern", "Event-driven hooks"],
    "factory": ["Direct instantiation", "Dependency injection", "Service locator"],
    "singleton": ["Module-level state", "Dependency injection", "Context objects"],
    "observer": ["Direct callbacks", "Event bus", "Reactive streams"],
    "strategy": ["Conditional branching", "Function dispatch table", "Polymorphism"],
    "decorator": ["Inheritance", "Middleware chain", "Composition"],
    "repository": ["Direct database access", "Active Record", "Query builder"],
    "adapter": ["Direct integration", "Façade", "Proxy"],
    "command": ["Direct method calls", "Event sourcing", "Message queue"],
    "template_method": ["Composition", "Strategy pattern", "Hooks"],
}

_PATTERN_IMPLICATIONS: dict[str, list[str]] = {
    "middleware_chain": [
        "Easy to add new middleware",
        "Processing order is explicit",
        "Each middleware is independently testable",
    ],
    "factory": [
        "Object creation is centralized",
        "Easy to swap implementations",
        "Decouples creation from usage",
    ],
    "singleton": [
        "Global state is shared",
        "Harder to test in isolation",
        "Reduces instantiation overhead",
    ],
    "observer": [
        "Loose coupling between components",
        "Supports multiple subscribers",
        "Order of notification is non-deterministic",
    ],
    "strategy": [
        "Algorithms are interchangeable at runtime",
        "Each strategy is independently testable",
        "Client code is decoupled from implementation details",
    ],
    "decorator": [
        "Behavior can be composed at runtime",
        "Open/closed principle respected",
        "Deep decorator chains can be hard to debug",
    ],
    "repository": [
        "Data access logic is centralized",
        "Easy to switch storage backends",
        "Supports unit testing with mock repositories",
    ],
    "adapter": [
        "Legacy or third-party interfaces are normalized",
        "Integration points are isolated",
        "Adds an indirection layer",
    ],
    "command": [
        "Operations are encapsulated as objects",
        "Supports undo/redo",
        "Commands can be queued or logged",
    ],
    "template_method": [
        "Common algorithm structure is shared",
        "Subclasses control variation points",
        "Inheritance-based extensibility",
    ],
}

_DECISION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string"},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "implications": {"type": "array", "items": {"type": "string"}},
        "source": {"type": "string", "enum": ["structural", "llm_inferred"]},
    },
    "required": ["decision", "alternatives", "evidence", "implications", "source"],
    "additionalProperties": False,
}

_LLM_SYSTEM_PROMPT = (
    "You are an expert software architect. Analyze the provided architectural pattern evidence "
    "and explain the trade-off rationale. Return structured JSON matching the schema."
)


def _build_decision_from_pattern(pattern: DetectedPattern) -> ArchDecision:
    evidence_list = [
        f"{e.file_path}:{e.start_line}-{e.end_line} ({e.symbol})" for e in pattern.evidence
    ]
    alternatives = _PATTERN_ALTERNATIVES.get(
        pattern.name,
        ["Alternative approach A", "Alternative approach B"],
    )
    implications = _PATTERN_IMPLICATIONS.get(
        pattern.name,
        ["Affects maintainability", "Affects testability"],
    )
    decision_text = (
        f"Uses {pattern.display_name} pattern for {pattern.description.lower().rstrip('.')}"
    )
    return ArchDecision(
        decision=decision_text,
        alternatives=alternatives,
        evidence=evidence_list,
        implications=implications,
        source="structural",
    )


def _build_llm_prompt(pattern: DetectedPattern, modules: list[Module]) -> str:
    evidence_lines = "\n".join(
        f"  - {e.file_path}:{e.start_line}-{e.end_line} | {e.symbol}: {e.explanation}"
        for e in pattern.evidence
    )
    module_names = ", ".join(m.name for m in modules) if modules else "none"
    return (
        f"Pattern: {pattern.display_name} (confidence: {pattern.confidence:.2f})\n"
        f"Category: {pattern.category}\n"
        f"Description: {pattern.description}\n"
        f"Evidence:\n{evidence_lines}\n"
        f"Modules present: {module_names}\n\n"
        "Provide an ArchDecision JSON with decision, alternatives, evidence (file paths), "
        "implications, and source='llm_inferred'."
    )


def infer_decisions(
    patterns: list[DetectedPattern],
    modules: list[Module],
    interfaces: list[Interface],
    provider: LLMProvider | None = None,
) -> list[ArchDecision]:
    """Infer architectural decisions from detected patterns.

    Structural decisions are always generated for patterns with confidence >= 0.5.
    When a provider is supplied, each decision is enriched via LLM.
    """
    decisions: list[ArchDecision] = []

    for pattern in patterns:
        if pattern.confidence < 0.5:
            continue

        structural = _build_decision_from_pattern(pattern)

        if provider is None:
            decisions.append(structural)
            continue

        prompt = _build_llm_prompt(pattern, modules)
        try:
            raw = provider.complete_structured(
                prompt=prompt,
                schema=_DECISION_SCHEMA,
                system=_LLM_SYSTEM_PROMPT,
            )
            enriched = ArchDecision(
                decision=str(raw.get("decision", structural.decision)),
                alternatives=[str(a) for a in raw.get("alternatives", structural.alternatives)],  # type: ignore[union-attr]
                evidence=[str(e) for e in raw.get("evidence", structural.evidence)],  # type: ignore[union-attr]
                implications=[str(i) for i in raw.get("implications", structural.implications)],  # type: ignore[union-attr]
                source="llm_inferred",
            )
            decisions.append(enriched)
        except (ProviderError, Exception):
            # Fall back to structural decision if LLM fails
            decisions.append(structural)

    return decisions
