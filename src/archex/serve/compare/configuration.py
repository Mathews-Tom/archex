"""Configuration dimension: config modules, env-var usage, config-library dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import ArchProfile, DimensionComparison
from archex.serve.compare._base import (
    classify_delta,
    interface_excerpts,
    interfaces_matching,
    library_diff_trade_offs,
    matched_deps,
    module_excerpts,
    modules_matching,
    pattern_excerpts,
    patterns_matching,
    repo_label,
)

_PATTERN_KEYWORDS = {"config", "settings", "env", "environment", "options", "configuration"}
_MODULE_KEYWORDS = {"config", "settings", "configuration", "options"}
_INTERFACE_KEYWORDS = {"env", "getenv", "config", "settings", "load_config"}
_CONFIG_DEPS = {
    "pydantic_settings",
    "pydantic",
    "dynaconf",
    "environs",
    "python_dotenv",
    "dotenv",
    "configargparse",
    "hydra_core",
    "omegaconf",
}


@dataclass
class ConfigurationEvidence:
    """Quantitative signals for configuration posture."""

    repo: str
    config_modules: int
    env_interfaces: int
    pattern_count: int
    config_deps: list[str]
    pattern_samples: list[str]
    module_samples: list[str]
    interface_samples: list[str]


def extract(profile: ArchProfile) -> ConfigurationEvidence:
    modules = modules_matching(profile.module_map, _MODULE_KEYWORDS)
    interfaces = interfaces_matching(profile.interface_surface, _INTERFACE_KEYWORDS)
    patterns = patterns_matching(profile.pattern_catalog, _PATTERN_KEYWORDS)
    return ConfigurationEvidence(
        repo=repo_label(profile),
        config_modules=len(modules),
        env_interfaces=len(interfaces),
        pattern_count=len(patterns),
        config_deps=matched_deps(profile.module_map, _CONFIG_DEPS),
        pattern_samples=pattern_excerpts(patterns),
        module_samples=module_excerpts(modules),
        interface_samples=interface_excerpts(interfaces),
    )


def _approach(ev: ConfigurationEvidence) -> str:
    if ev.config_modules + ev.env_interfaces + len(ev.config_deps) == 0:
        return "No configuration layer detected"
    parts: list[str] = []
    if ev.config_modules:
        parts.append(f"{ev.config_modules} config module(s)")
    if ev.env_interfaces:
        parts.append(f"{ev.env_interfaces} env-aware interface(s)")
    if ev.config_deps:
        parts.append(f"libraries: {', '.join(ev.config_deps)}")
    return "; ".join(parts)


def _evidence_lines(ev: ConfigurationEvidence) -> list[str]:
    lines = [
        f"Config modules: {ev.config_modules}",
        f"Env-aware interfaces: {ev.env_interfaces}",
        f"Config patterns: {ev.pattern_count}",
        f"Config libraries: {len(ev.config_deps)}",
    ]
    if ev.config_deps:
        lines.append(f"Libraries: {', '.join(ev.config_deps)}")
    lines.extend(ev.module_samples)
    lines.extend(ev.interface_samples)
    lines.extend(ev.pattern_samples)
    return lines


def compare(a: ConfigurationEvidence, b: ConfigurationEvidence) -> DimensionComparison:
    trade_offs = [classify_delta("config-module", a.config_modules, b.config_modules)]
    trade_offs.extend(library_diff_trade_offs(a.repo, b.repo, a.config_deps, b.config_deps))
    return DimensionComparison(
        dimension="configuration",
        repo_a_approach=_approach(a),
        repo_b_approach=_approach(b),
        evidence_a=_evidence_lines(a),
        evidence_b=_evidence_lines(b),
        trade_offs=trade_offs,
    )
