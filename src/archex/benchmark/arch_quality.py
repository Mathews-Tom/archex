"""Architecture-quality benchmark scoring for analyze/explain outputs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from archex.api import analyze
from archex.benchmark.loader import load_arch_tasks
from archex.benchmark.models import (
    ArchitectureBenchmarkResult,
    ArchitectureBenchmarkTask,
    ArchitectureDimensionScores,
    ArchitectureExpectedModule,
)
from archex.exceptions import BenchmarkCloneError
from archex.models import ArchProfile, Config, RepoSource


def run_architecture_benchmark(task: ArchitectureBenchmarkTask) -> ArchitectureBenchmarkResult:
    """Run architecture analysis for a single labeled task and score it against its oracle."""
    repo_path = _prepare_architecture_repo(task)
    try:
        config = Config(languages=task.languages, cache=False, parallel=False)
        profile = analyze(RepoSource(local_path=str(repo_path)), config)
        scores = score_architecture_profile(task, profile)
        return ArchitectureBenchmarkResult(
            task_id=task.task_id,
            repo=task.repo,
            commit=task.commit,
            scores=scores,
            detected_modules=[module.name for module in profile.module_map],
            detected_patterns=[pattern.name for pattern in profile.pattern_catalog],
            detected_interfaces=[interface.symbol.name for interface in profile.interface_surface],
            detected_decisions=[decision.decision for decision in profile.decision_log],
        )
    finally:
        if repo_path != Path.cwd():
            shutil.rmtree(repo_path, ignore_errors=True)


def run_architecture_all(
    tasks_dir: Path,
    output_dir: Path,
    *,
    task_filter: str | None = None,
) -> list[ArchitectureBenchmarkResult]:
    """Run every architecture-quality task and write one JSON result per task."""
    tasks = load_arch_tasks(tasks_dir)
    if task_filter is not None:
        tasks = [task for task in tasks if task.task_id == task_filter]
        if not tasks:
            raise ValueError(f"No architecture task found with id '{task_filter}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ArchitectureBenchmarkResult] = []
    for task in tasks:
        result = run_architecture_benchmark(task)
        results.append(result)
        result_path = output_dir / f"{task.task_id}.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return results


def score_architecture_profile(
    task: ArchitectureBenchmarkTask,
    profile: ArchProfile,
) -> ArchitectureDimensionScores:
    """Score module, pattern, interface, and decision outputs against an architecture oracle."""
    oracle = task.arch_oracle
    boundary_precision, boundary_recall, boundary_f1 = _boundary_scores(
        oracle.modules,
        [set(module.files) for module in profile.module_map],
    )
    responsibility_recall = _responsibility_recall(oracle.modules, profile)
    pattern_precision, pattern_recall = _pattern_scores(
        [pattern.name for pattern in oracle.patterns],
        [pattern.name for pattern in profile.pattern_catalog],
    )
    interface_completeness = _interface_completeness(
        {(interface.file_path, interface.name) for interface in oracle.interfaces},
        {
            (interface.symbol.file_path, interface.symbol.name)
            for interface in profile.interface_surface
        },
    )
    decision_recall = _decision_recall(
        [decision.decision_terms for decision in oracle.decisions],
        [decision.decision for decision in profile.decision_log],
    )
    measured = [
        boundary_f1,
        responsibility_recall,
        pattern_precision,
        pattern_recall,
        interface_completeness,
        decision_recall,
    ]
    overall = round(sum(measured) / len(measured), 4)
    return ArchitectureDimensionScores(
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
        responsibility_recall=responsibility_recall,
        pattern_precision=pattern_precision,
        pattern_recall=pattern_recall,
        interface_completeness=interface_completeness,
        decision_recall=decision_recall,
        overall=overall,
    )


def format_architecture_summary(results: list[ArchitectureBenchmarkResult]) -> str:
    """Render per-dimension architecture-quality scores as Markdown."""
    lines = [
        "# Architecture Quality Benchmark",
        "",
        "| Task | Boundary F1 | Pattern P | Pattern R | Interface C | Decision R | Overall |",
        "|------|-------------|-----------|-----------|-------------|------------|---------|",
    ]
    for result in results:
        scores = result.scores
        lines.append(
            f"| {result.task_id} | {scores.boundary_f1:.3f} | "
            f"{scores.pattern_precision:.3f} | {scores.pattern_recall:.3f} | "
            f"{scores.interface_completeness:.3f} | {scores.decision_recall:.3f} | "
            f"{scores.overall:.3f} |"
        )
    lines.append("")
    lines.append("Architecture-quality gate mode: ADVISORY")
    return "\n".join(lines)


def architecture_gate_warnings(
    results: list[ArchitectureBenchmarkResult],
    *,
    baseline_results: list[ArchitectureBenchmarkResult] | None = None,
    min_boundary_f1: float = 0.80,
    min_pattern_precision: float = 0.80,
    min_pattern_recall: float = 0.80,
    min_interface_completeness: float = 0.80,
) -> list[str]:
    """Return advisory architecture-quality gate warnings without blocking releases."""
    warnings: list[str] = []
    checks = [
        ("boundary_f1", min_boundary_f1),
        ("pattern_precision", min_pattern_precision),
        ("pattern_recall", min_pattern_recall),
        ("interface_completeness", min_interface_completeness),
    ]
    for result in results:
        for metric, threshold in checks:
            actual = getattr(result.scores, metric)
            if actual < threshold:
                warnings.append(f"{result.task_id} {metric}: {actual:.3f} < {threshold:.3f}")

    if baseline_results is None:
        return warnings

    baseline_by_task = {result.task_id: result for result in baseline_results}
    regression_metrics = [
        "boundary_f1",
        "pattern_precision",
        "pattern_recall",
        "interface_completeness",
        "decision_recall",
        "overall",
    ]
    for result in results:
        baseline = baseline_by_task.get(result.task_id)
        if baseline is None:
            warnings.append(f"{result.task_id}: missing architecture baseline result")
            continue
        for metric in regression_metrics:
            actual = getattr(result.scores, metric)
            expected = getattr(baseline.scores, metric)
            if actual < expected:
                warnings.append(
                    f"{result.task_id} {metric} regressed: {actual:.3f} < baseline {expected:.3f}"
                )
    return warnings


def load_architecture_results(input_dir: Path) -> list[ArchitectureBenchmarkResult]:
    """Load architecture-quality result JSON files from a directory."""
    results: list[ArchitectureBenchmarkResult] = []
    for json_file in sorted(input_dir.glob("*.json")):
        results.append(
            ArchitectureBenchmarkResult.model_validate_json(json_file.read_text(encoding="utf-8"))
        )
    return results


def _prepare_architecture_repo(task: ArchitectureBenchmarkTask) -> Path:
    if task.repo != ".":
        msg = "architecture benchmark tasks must use local repo '.' in this benchmark harness"
        raise BenchmarkCloneError(msg)

    source_root = Path.cwd()
    target = Path(tempfile.mkdtemp(prefix="archex-arch-bench-"))
    try:
        for rel_path in task.include_paths:
            source = source_root / rel_path
            if not source.exists():
                msg = f"include path {rel_path!r} does not exist"
                raise BenchmarkCloneError(msg)
            destination = target / rel_path
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        git_init = subprocess.run(
            ["git", "init", "--quiet"],
            cwd=target,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_init.returncode != 0:
            detail = git_init.stderr.strip() or "unknown git error"
            raise BenchmarkCloneError(f"git init failed for architecture benchmark repo: {detail}")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def _boundary_scores(
    expected_modules: list[ArchitectureExpectedModule],
    detected_modules: list[set[str]],
) -> tuple[float, float, float]:
    if not expected_modules:
        return 1.0, 1.0, 1.0

    expected_files = {path for module in expected_modules for path in module.files}
    expected_pairs = _cluster_pairs(
        [set(module.files) for module in expected_modules], expected_files
    )
    detected_pairs = _cluster_pairs(detected_modules, expected_files)
    if not expected_pairs and not detected_pairs:
        return 1.0, 1.0, 1.0

    true_positive = len(expected_pairs & detected_pairs)
    precision = _ratio(true_positive, len(detected_pairs), empty_value=0.0)
    recall = _ratio(true_positive, len(expected_pairs), empty_value=0.0)
    return precision, recall, _f1(precision, recall)


def _cluster_pairs(clusters: list[set[str]], universe: set[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        scoped = sorted(cluster & universe)
        for index, left in enumerate(scoped):
            for right in scoped[index + 1 :]:
                pairs.add((left, right))
    return pairs


def _responsibility_recall(
    expected_modules: list[ArchitectureExpectedModule],
    profile: ArchProfile,
) -> float:
    expected_terms = [
        (module, term.lower())
        for module in expected_modules
        for term in module.responsibility_terms
    ]
    if not expected_terms:
        return 1.0

    matches = 0
    detected = profile.module_map
    for expected, term in expected_terms:
        expected_files = set(expected.files)
        matched_module = max(
            detected,
            key=lambda module: len(set(module.files) & expected_files),
            default=None,
        )
        if (
            matched_module is None
            or not set(matched_module.files) & expected_files
            or matched_module.responsibility is None
        ):
            continue
        if term in matched_module.responsibility.lower():
            matches += 1
    return _ratio(matches, len(expected_terms), empty_value=1.0)


def _pattern_scores(expected_names: list[str], detected_names: list[str]) -> tuple[float, float]:
    expected = set(expected_names)
    detected = set(detected_names)
    true_positive = len(expected & detected)
    precision = _ratio(true_positive, len(detected), empty_value=1.0 if not expected else 0.0)
    recall = _ratio(true_positive, len(expected), empty_value=1.0)
    return precision, recall


def _interface_completeness(
    expected: set[tuple[str, str]],
    detected: set[tuple[str, str]],
) -> float:
    if not expected:
        return 1.0
    return _ratio(len(expected & detected), len(expected), empty_value=1.0)


def _decision_recall(expected_terms: list[list[str]], detected_decisions: list[str]) -> float:
    if not expected_terms:
        return 1.0

    detected = [decision.lower() for decision in detected_decisions]
    matches = 0
    for terms in expected_terms:
        lowered_terms = [term.lower() for term in terms]
        if any(all(term in decision for term in lowered_terms) for decision in detected):
            matches += 1
    return _ratio(matches, len(expected_terms), empty_value=1.0)


def _ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return round(numerator / denominator, 4)


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)
