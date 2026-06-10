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


_PATTERN_CONTAINER_KINDS = {SymbolKind.CLASS, SymbolKind.TYPE, SymbolKind.INTERFACE}


def _public_methods_of(container_name: str, symbols: list[Symbol]) -> list[Symbol]:
    """Return public method symbols belonging to a class, type, or interface."""
    return [
        s
        for s in symbols
        if s.kind == SymbolKind.METHOD
        and s.parent == container_name
        and s.visibility == Visibility.PUBLIC
    ]


def _method_names(container_name: str, symbols: list[Symbol]) -> set[str]:
    return {s.name.lower() for s in _public_methods_of(container_name, symbols)}


def _classes(symbols: list[Symbol]) -> list[Symbol]:
    return [s for s in symbols if s.kind in _PATTERN_CONTAINER_KINDS]


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
        full_chain_found = False

        # First pass: find full chain containers
        for cls in classes:
            methods = _method_names(cls.name, pf.symbols)
            has_next = "set_next" in methods or "next" in methods
            has_process = "process" in methods or "handle" in methods

            if has_next and has_process:
                full_chain_found = True
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Container '{cls.name}' has set_next/next + process/handle methods"
                        ),
                    )
                )

        # Second pass: find chain participants
        for cls in classes:
            methods = _method_names(cls.name, pf.symbols)
            has_next = "set_next" in methods or "next" in methods
            has_process = "process" in methods or "handle" in methods

            if has_next and has_process:
                continue  # Already added in first pass

            if has_process:
                has_chain_name = any(
                    kw in cls.name.lower()
                    for kw in ("middleware", "handler", "chain", "filter", "interceptor")
                )
                if has_chain_name or full_chain_found:
                    evidence.append(
                        PatternEvidence(
                            file_path=pf.path,
                            start_line=cls.start_line,
                            end_line=cls.end_line,
                            symbol=cls.name,
                            explanation=(
                                f"Container '{cls.name}' overrides process/handle "
                                "(chain participant)"
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

            # "on" alone is too generic -- require corroboration
            if methods & subscribe_names == {"on"}:
                name_lower = cls.name.lower()
                if not any(
                    kw in name_lower for kw in ("event", "bus", "emitter", "dispatcher", "listener")
                ):
                    continue

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

            name_indicates_repo = any(
                kw in cls.name.lower() for kw in ("repo", "repository", "dao", "store", "data")
            )
            if len(crud_hits) >= 3 or (len(crud_hits) >= 2 and name_indicates_repo):
                evidence.append(
                    PatternEvidence(
                        file_path=pf.path,
                        start_line=cls.start_line,
                        end_line=cls.end_line,
                        symbol=cls.name,
                        explanation=(
                            f"Container '{cls.name}' has CRUD-like methods: "
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
    """Detect Strategy pattern across files in a repository slice.

    Looks for the triad: protocol/interface + multiple concrete implementations sharing
    method names + a context class that holds and delegates to the strategy.
    """
    all_evidence: list[PatternEvidence] = []
    seen: set[str] = set()
    strategy_keywords = {"strategy", "policy", "algorithm", "interface", "protocol"}
    context_keywords = {"context", "executor", "runner", "engine", "sorter", "dispatcher"}

    def _add(pf: ParsedFile, sym: Symbol, explanation: str) -> None:
        key = f"{pf.path}:{sym.name}"
        if key in seen:
            return
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

    class_records: list[tuple[ParsedFile, Symbol, set[str]]] = []
    for pf in parsed_files:
        for cls in _classes(pf.symbols):
            methods = {
                method.name
                for method in _public_methods_of(cls.name, pf.symbols)
                if not (method.name.startswith("__") and method.name.endswith("__"))
            }
            if not methods:
                continue
            record = (pf, cls, methods)
            class_records.append(record)

    protocol_candidates: list[tuple[ParsedFile, Symbol, set[str]]] = []
    context_candidates: list[tuple[ParsedFile, Symbol, set[str]]] = []
    concrete_candidates: list[tuple[ParsedFile, Symbol, set[str]]] = []

    for record in class_records:
        _pf, cls, methods = record
        name_lower = cls.name.lower()
        is_protocol_name = any(keyword in name_lower for keyword in strategy_keywords)
        is_context_name = any(keyword in name_lower for keyword in context_keywords)
        if is_context_name:
            context_candidates.append(record)
        elif is_protocol_name and len(methods) <= 3:
            protocol_candidates.append(record)

    if protocol_candidates:
        protocol_methods: set[str] = set()
        for pf, proto, _methods in protocol_candidates:
            protocol_methods |= {
                symbol.name
                for symbol in pf.symbols
                if symbol.kind == SymbolKind.METHOD
                and symbol.parent == proto.name
                and not (symbol.name.startswith("__") and symbol.name.endswith("__"))
            }
        for record in class_records:
            if record in protocol_candidates or record in context_candidates:
                continue
            _pf, _cls, methods = record
            if methods & protocol_methods:
                concrete_candidates.append(record)

    if not (
        protocol_candidates
        and concrete_candidates
        and len(protocol_candidates) + len(concrete_candidates) >= 3
    ):
        return None

    for pf, proto, _methods in protocol_candidates:
        _add(pf, proto, f"Protocol/ABC '{proto.name}' defines strategy interface")
    for pf, concrete, _methods in concrete_candidates:
        _add(pf, concrete, f"Concrete strategy implementation: '{concrete.name}'")
    for pf, ctx, _methods in context_candidates:
        _add(pf, ctx, f"Context class '{ctx.name}' delegates to a strategy")

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


PatternVerifier = Callable[["DetectedPattern", "list[ParsedFile]"], "float | None"]


def detect_patterns(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,
    registry: PatternRegistry | None = None,
    verifier: PatternVerifier | None = None,
) -> list[DetectedPattern]:
    """Run all pattern detectors and return non-None results.

    When *verifier* is provided, each detected pattern is passed through it.
    The verifier returns an adjusted confidence float, or ``None`` to keep the
    original confidence unchanged.
    """
    reg = registry or default_registry
    results: list[DetectedPattern] = []
    for detector in reg.detectors:
        pattern = detector(parsed_files, graph)
        if pattern is not None:
            if verifier is not None:
                adjusted = verifier(pattern, parsed_files)
                if adjusted is not None:
                    pattern = pattern.model_copy(update={"confidence": adjusted})
            results.append(pattern)
    return results
