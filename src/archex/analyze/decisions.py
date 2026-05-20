"""Architecture decision inference: derive ArchDecision records from patterns and structure."""

from __future__ import annotations

from archex.models import ArchDecision, DetectedPattern, Interface, Module

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


def infer_decisions(
    patterns: list[DetectedPattern],
    modules: list[Module],
    interfaces: list[Interface],
) -> list[ArchDecision]:
    """Infer architectural decisions from detected patterns.

    Decisions are generated structurally for patterns with confidence >= 0.5.
    """
    del modules, interfaces  # reserved for future structural signals
    return [
        _build_decision_from_pattern(pattern) for pattern in patterns if pattern.confidence >= 0.5
    ]
