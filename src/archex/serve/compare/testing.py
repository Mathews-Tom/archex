"""Testing dimension: test-module count, test-language presence, test-related patterns."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import ArchProfile, DimensionComparison
from archex.serve.compare._base import (
    classify_delta,
    module_excerpts,
    modules_matching,
    pattern_excerpts,
    patterns_matching,
)

_PATTERN_KEYWORDS = {"test", "fixture", "mock", "spec", "assert", "parametrize"}
_MODULE_KEYWORDS = {"test", "tests", "spec", "specs", "__tests__"}
_TEST_DEPS = {
    "pytest",
    "unittest",
    "nose",
    "hypothesis",
    "mock",
    "responses",
    "respx",
    "freezegun",
    "factory_boy",
    "tox",
}


@dataclass
class TestingEvidence:
    """Quantitative signals for test posture."""

    repo: str
    test_modules: int
    test_files_total: int
    total_files: int
    pattern_count: int
    test_deps: list[str]
    pattern_samples: list[str]
    module_samples: list[str]


def extract(profile: ArchProfile) -> TestingEvidence:
    matched_modules = modules_matching(profile.module_map, _MODULE_KEYWORDS)
    test_files = sum(m.file_count for m in matched_modules)
    patterns = patterns_matching(profile.pattern_catalog, _PATTERN_KEYWORDS)
    deps: set[str] = set()
    for mod in profile.module_map:
        for dep in mod.external_deps:
            normalized = dep.lower().split(".")[0].replace("-", "_")
            if normalized in _TEST_DEPS:
                deps.add(normalized)
    return TestingEvidence(
        repo=profile.repo.url or profile.repo.local_path or "repo",
        test_modules=len(matched_modules),
        test_files_total=test_files,
        total_files=profile.stats.total_files,
        pattern_count=len(patterns),
        test_deps=sorted(deps),
        pattern_samples=pattern_excerpts(patterns),
        module_samples=module_excerpts(matched_modules),
    )


def _ratio(ev: TestingEvidence) -> float:
    if ev.total_files == 0:
        return 0.0
    return ev.test_files_total / ev.total_files


def _approach(ev: TestingEvidence) -> str:
    if ev.test_modules + ev.pattern_count + len(ev.test_deps) == 0:
        return "No test modules or test-framework dependencies detected"
    parts: list[str] = []
    if ev.test_modules:
        parts.append(
            f"{ev.test_modules} test module(s) covering {ev.test_files_total} file(s) "
            f"({_ratio(ev):.0%} of repo)"
        )
    if ev.test_deps:
        parts.append(f"frameworks: {', '.join(ev.test_deps)}")
    return "; ".join(parts) or "Test signals present but unstructured"


def _evidence_lines(ev: TestingEvidence) -> list[str]:
    lines = [
        f"Test modules: {ev.test_modules}",
        f"Test files: {ev.test_files_total} of {ev.total_files} ({_ratio(ev):.1%})",
        f"Test/fixture patterns: {ev.pattern_count}",
        f"Test libraries: {len(ev.test_deps)}",
    ]
    if ev.test_deps:
        lines.append(f"Frameworks: {', '.join(ev.test_deps)}")
    lines.extend(ev.module_samples)
    lines.extend(ev.pattern_samples)
    return lines


def compare(a: TestingEvidence, b: TestingEvidence) -> DimensionComparison:
    trade_offs = [classify_delta("test-module", a.test_modules, b.test_modules)]
    ratio_a = _ratio(a)
    ratio_b = _ratio(b)
    if ratio_a >= ratio_b + 0.1:
        trade_offs.append(
            f"`{a.repo}` invests more in test surface ({ratio_a:.0%} vs {ratio_b:.0%})."
        )
    elif ratio_b >= ratio_a + 0.1:
        trade_offs.append(
            f"`{b.repo}` invests more in test surface ({ratio_b:.0%} vs {ratio_a:.0%})."
        )
    a_only = set(a.test_deps) - set(b.test_deps)
    b_only = set(b.test_deps) - set(a.test_deps)
    if a_only:
        trade_offs.append(f"`{a.repo}` uses {', '.join(sorted(a_only))} not seen in `{b.repo}`.")
    if b_only:
        trade_offs.append(f"`{b.repo}` uses {', '.join(sorted(b_only))} not seen in `{a.repo}`.")
    return DimensionComparison(
        dimension="testing",
        repo_a_approach=_approach(a),
        repo_b_approach=_approach(b),
        evidence_a=_evidence_lines(a),
        evidence_b=_evidence_lines(b),
        trade_offs=trade_offs,
    )
