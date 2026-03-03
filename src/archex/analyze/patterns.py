"""Architectural pattern detection: identify structural, behavioral, and creational patterns."""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from archex.exceptions import ConfigError
from archex.models import (
    DetectedPattern,
    ParsedFile,
    PatternCategory,
    PatternEvidence,
    Symbol,
    SymbolKind,
    Visibility,
)

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph

logger = logging.getLogger(__name__)

# Type alias for detector functions
PatternDetector = Callable[["list[ParsedFile]", "DependencyGraph"], "DetectedPattern | None"]


class PatternRegistry:
    """Registry of pattern detector functions, supporting decorator and entry-point registration."""

    def __init__(self) -> None:
        self._detectors: list[PatternDetector] = []
        self._entry_points_loaded: bool = False
        self._entry_points_strict: bool = False

    def register(self, fn: PatternDetector) -> PatternDetector:
        """Decorator to register a pattern detector function."""
        self._detectors.append(fn)
        return fn

    def add(self, fn: PatternDetector) -> None:
        """Programmatic registration (non-decorator)."""
        self._detectors.append(fn)

    @property
    def detectors(self) -> list[PatternDetector]:
        return list(self._detectors)

    def load_entry_points(
        self,
        group: str = "archex.pattern_detectors",
        strict: bool = False,
    ) -> None:
        """Load detector functions from installed entry points."""
        if self._entry_points_loaded and (not strict or self._entry_points_strict):
            return  # already loaded at equal or higher strictness
        eps = sorted(importlib.metadata.entry_points(group=group), key=lambda ep: ep.name)
        for ep in eps:
            try:
                fn = ep.load()
                self._detectors.append(fn)
                logger.info("Loaded pattern detector %s from entry point", ep.name)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise ConfigError(
                        f"Failed to load pattern detector entry point {ep.name!r}: {exc}"
                    ) from exc
                logger.warning("Failed to load pattern detector entry point %s: %s", ep.name, exc)
        self._entry_points_loaded = True
        self._entry_points_strict = strict


# Module-level default registry
default_registry = PatternRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_methods_of(class_name: str, symbols: list[Symbol]) -> list[Symbol]:
    """Return public method symbols belonging to a class."""
    return [
        s
        for s in symbols
        if s.kind == SymbolKind.METHOD
        and s.parent == class_name
        and s.visibility == Visibility.PUBLIC
    ]


def _method_names(class_name: str, symbols: list[Symbol]) -> set[str]:
    return {s.name for s in _public_methods_of(class_name, symbols)}


def _classes(symbols: list[Symbol]) -> list[Symbol]:
    return [s for s in symbols if s.kind == SymbolKind.CLASS]


def _confidence(evidence_count: int) -> float:
    if evidence_count >= 3:
        return 0.85
    if evidence_count == 2:
        return 0.60
    if evidence_count == 1:
        return 0.30
    return 0.0


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _detect_middleware(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,  # noqa: ARG001
) -> DetectedPattern | None:
    """Detect Middleware Chain / Chain-of-Responsibility pattern."""
    evidence: list[PatternEvidence] = []

    for pf in parsed_files:
        classes = _classes(pf.symbols)
        for cls in classes:
            methods = _method_names(cls.name, pf.symbols)
            has_next = "set_next" in methods or "next" in methods
            has_process = "process" in methods or "handle" in methods

            if has_next and has_process:
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Class '{cls.name}' has set_next/next + process/handle methods"
                        ),
                    )
                )
            elif has_process:
                # Check if it's a subclass that only overrides process/handle (chain participant)
                # A class with process/handle but no set_next may be a concrete handler
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Class '{cls.name}' overrides process/handle (chain participant)"
                        ),
                    )
                )

    if not evidence:
        return None

    return DetectedPattern(
        name="middleware_chain",
        display_name="Middleware Chain",
        confidence=_confidence(len(evidence)),
        evidence=evidence,
        description=(
            "A chain-of-responsibility / middleware pattern where handlers are linked "
            "sequentially and each can pass requests to the next handler."
        ),
        category=PatternCategory.BEHAVIORAL,
    )


def _detect_plugin_system(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,  # noqa: ARG001
) -> DetectedPattern | None:
    """Detect Plugin / Extension System pattern."""
    evidence: list[PatternEvidence] = []
    seen: set[str] = set()

    def _add(pf: ParsedFile, sym: Symbol, explanation: str) -> None:
        key = f"{pf.path}:{sym.name}"
        if key not in seen:
            seen.add(key)
            evidence.append(
                PatternEvidence(
                    file_path=pf.path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    symbol=sym.name,
                    explanation=explanation,
                )
            )

    for pf in parsed_files:
        classes = _classes(pf.symbols)
        for cls in classes:
            methods = _method_names(cls.name, pf.symbols)
            has_registry_methods = "register" in methods or "unregister" in methods
            has_query = "get" in methods or "all" in methods

            has_registry_name = any(
                keyword in cls.name.lower() for keyword in ("registry", "manager", "store")
            )

            if has_registry_methods and (has_query or has_registry_name):
                _add(
                    pf,
                    cls,
                    (
                        f"Class '{cls.name}' has register/unregister + get/all methods"
                        " (plugin registry)"
                    ),
                )

        # Look for Protocol/ABC used as a plugin interface
        for sym in pf.symbols:
            if sym.kind == SymbolKind.CLASS and sym.parent is None:
                name_lower = sym.name.lower()
                if "plugin" in name_lower or "extension" in name_lower or "handler" in name_lower:
                    _add(pf, sym, f"Protocol/interface class '{sym.name}' defines plugin contract")

    if not evidence:
        return None

    return DetectedPattern(
        name="plugin_system",
        display_name="Plugin / Extension System",
        confidence=_confidence(len(evidence)),
        evidence=evidence,
        description=(
            "A plugin/extension system with a registry that manages registered implementations "
            "of a common protocol or interface."
        ),
        category=PatternCategory.STRUCTURAL,
    )


def _detect_event_bus(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,  # noqa: ARG001
) -> DetectedPattern | None:
    """Detect Event Bus / Pub-Sub pattern."""
    evidence: list[PatternEvidence] = []

    subscribe_names = {"subscribe", "on", "listen", "add_listener", "add_handler"}
    emit_names = {"emit", "publish", "dispatch", "trigger", "fire", "send"}

    for pf in parsed_files:
        classes = _classes(pf.symbols)
        for cls in classes:
            methods = _method_names(cls.name, pf.symbols)
            has_subscribe = bool(methods & subscribe_names)
            has_emit = bool(methods & emit_names)

            if has_subscribe and has_emit:
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Class '{cls.name}' has subscribe/listen +"
                            " emit/publish methods (event bus)"
                        ),
                    )
                )
            elif has_subscribe or has_emit:
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Class '{cls.name}' has "
                            f"{'subscriber' if has_subscribe else 'emitter'} methods"
                        ),
                    )
                )

    if not evidence:
        return None

    return DetectedPattern(
        name="event_bus",
        display_name="Event Bus / Pub-Sub",
        confidence=_confidence(len(evidence)),
        evidence=evidence,
        description=(
            "An event bus / publish-subscribe pattern allowing components to communicate "
            "through events without direct coupling."
        ),
        category=PatternCategory.BEHAVIORAL,
    )


def _detect_repository(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,  # noqa: ARG001
) -> DetectedPattern | None:
    """Detect Repository / DAO pattern."""
    evidence: list[PatternEvidence] = []

    crud_names = {"get", "find", "save", "create", "update", "delete", "all", "list", "fetch"}

    for pf in parsed_files:
        classes = _classes(pf.symbols)
        for cls in classes:
            methods = _method_names(cls.name, pf.symbols)
            crud_hits = methods & crud_names

            if len(crud_hits) >= 2:
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Class '{cls.name}' has CRUD-like methods: "
                            f"{', '.join(sorted(crud_hits))}"
                        ),
                    )
                )

    if not evidence:
        return None

    return DetectedPattern(
        name="repository",
        display_name="Repository / DAO",
        confidence=_confidence(len(evidence)),
        evidence=evidence,
        description=(
            "A repository/data-access-object pattern encapsulating persistence logic "
            "behind a collection-oriented interface."
        ),
        category=PatternCategory.STRUCTURAL,
    )


def _detect_strategy(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,  # noqa: ARG001
) -> DetectedPattern | None:
    """Detect Strategy pattern.

    Looks for the triad: protocol/interface + multiple concrete implementations sharing
    the same method name + a context class that holds and delegates to the strategy.
    """
    all_evidence: list[PatternEvidence] = []
    seen: set[str] = set()

    def _add(pf: ParsedFile, sym: Symbol, explanation: str) -> None:
        key = f"{pf.path}:{sym.name}"
        if key not in seen:
            seen.add(key)
            all_evidence.append(
                PatternEvidence(
                    file_path=pf.path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    symbol=sym.name,
                    explanation=explanation,
                )
            )

    for pf in parsed_files:
        classes = _classes(pf.symbols)
        if len(classes) < 2:
            continue

        # Collect method name -> classes that implement it (excluding __init__, __repr__, etc.)
        method_to_classes: dict[str, list[Symbol]] = {}
        for cls in classes:
            for m in _public_methods_of(cls.name, pf.symbols):
                if not (m.name.startswith("__") and m.name.endswith("__")):
                    method_to_classes.setdefault(m.name, []).append(cls)

        # Find methods implemented by 2+ classes (shared interface method — strategy indicator)
        shared_methods = {m: clss for m, clss in method_to_classes.items() if len(clss) >= 2}
        if not shared_methods:
            continue

        # Determine protocol-like classes (name contains strategy/policy keywords or has very few
        # public non-dunder methods — Protocol classes typically have 1-2)
        strategy_keywords = {"strategy", "policy", "algorithm", "interface", "protocol"}
        context_keywords = {"context", "executor", "runner", "engine", "sorter", "dispatcher"}

        protocol_candidates: list[Symbol] = []
        context_candidates: list[Symbol] = []
        concrete_candidates: list[Symbol] = []

        for cls in classes:
            name_lower = cls.name.lower()
            public_non_dunder = [
                s
                for s in pf.symbols
                if s.kind == SymbolKind.METHOD
                and s.parent == cls.name
                and not (s.name.startswith("__") and s.name.endswith("__"))
            ]
            non_dunder_count = len(public_non_dunder)

            is_protocol_name = any(kw in name_lower for kw in strategy_keywords)
            is_context_name = any(kw in name_lower for kw in context_keywords)
            in_shared = shared_methods and any(cls in clss for clss in shared_methods.values())

            # Context takes priority; check it first
            if is_context_name and non_dunder_count >= 1:
                context_candidates.append(cls)
            elif is_protocol_name and non_dunder_count <= 3:
                protocol_candidates.append(cls)
            elif in_shared:
                concrete_candidates.append(cls)

        # If we found a protocol but no explicit concretes from name-based heuristic,
        # treat all non-protocol classes that share the same method as concretes.
        if protocol_candidates and not concrete_candidates:
            proto_methods: set[str] = set()
            for proto in protocol_candidates:
                proto_methods |= {
                    s.name
                    for s in pf.symbols
                    if s.kind == SymbolKind.METHOD
                    and s.parent == proto.name
                    and not (s.name.startswith("__") and s.name.endswith("__"))
                }
            for cls in classes:
                if cls in protocol_candidates or cls in context_candidates:
                    continue
                cls_methods = _method_names(cls.name, pf.symbols) - {
                    m for m in _method_names(cls.name, pf.symbols) if m.startswith("__")
                }
                if cls_methods & proto_methods:
                    concrete_candidates.append(cls)

        # Only emit evidence if we have at least: protocol + concretes OR concretes + context
        if (protocol_candidates and concrete_candidates) or (
            concrete_candidates and context_candidates
        ):
            for proto in protocol_candidates:
                _add(pf, proto, f"Protocol/ABC '{proto.name}' defines strategy interface")
            for concrete in concrete_candidates:
                _add(pf, concrete, f"Concrete strategy implementation: '{concrete.name}'")
            for ctx in context_candidates:
                _add(pf, ctx, f"Context class '{ctx.name}' delegates to a strategy")

    if not all_evidence:
        return None

    return DetectedPattern(
        name="strategy",
        display_name="Strategy Pattern",
        confidence=_confidence(len(all_evidence)),
        evidence=all_evidence,
        description=(
            "A strategy pattern where a family of algorithms is encapsulated behind "
            "a common interface, and the algorithm can be swapped at runtime."
        ),
        category=PatternCategory.BEHAVIORAL,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Register built-in detectors
for _fn in [
    _detect_middleware,
    _detect_plugin_system,
    _detect_event_bus,
    _detect_repository,
    _detect_strategy,
]:
    default_registry.register(_fn)


def detect_patterns(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,
    registry: PatternRegistry | None = None,
) -> list[DetectedPattern]:
    """Run all pattern detectors and return non-None results."""
    reg = registry or default_registry
    results: list[DetectedPattern] = []
    for detector in reg.detectors:
        pattern = detector(parsed_files, graph)
        if pattern is not None:
            results.append(pattern)
    return results
