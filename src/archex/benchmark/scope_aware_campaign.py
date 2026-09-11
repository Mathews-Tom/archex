"""Fail-closed validation for the frozen R27 scope-aware campaign."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from archex.models import IndexConfig

CAMPAIGN_ID = "r27-scope-aware-monorepo-ranking"
PROTOCOL_ID = "R26-CANDIDATE-A"
CONTROL_REVISION = "1eda0c85de26b4950490062802b41da9f2e00e68"
POPULATION_FILENAME = "population.json"
MANIFEST_FILENAME = "manifest.json"
CELLS_FILENAME = "cells.json"
EXPECTED_MANIFEST_SHA256 = "20caed725c4091a4ad675ead0d09f8bbd8ee6fcd1844b631480658a862175ca2"
EXPECTED_MANIFEST_FILE_SHA256 = "e4a35c04c468671b971dc920c250f0003ced0fbc036330aa80cd354998717e4b"
EXPECTED_CELLS_SHA256 = "cf0fee569c373441875035ae325f47a244362854ec513eaf20c2fb50c313a7eb"
CONTROL_RECEIPTS_FILENAME = "control_receipts.json"
POWER_FILENAME = "power.json"

EXPECTED_FAMILY_COUNTS = {
    "cross_scope_dependency": 688,
    "lexical_collision": 688,
    "single_scope_control": 16,
    "weak_participation_control": 672,
}
EXPECTED_KIND_COUNTS = {"single_scope_control": 16, "treatment": 2048}
EXPECTED_PAYLOAD_PROJECTION = (
    "query",
    "chunks",
    "structural_context",
    "type_definitions",
    "dependency_summary",
    "token_count",
    "token_budget",
    "truncated",
)
_HEX_64 = r"^[0-9a-f]{64}$"
_HEX_40 = r"^[0-9a-f]{40}$"


class ScopeAwareCampaignError(ValueError):
    """Raised when a frozen scope-aware campaign artifact is invalid."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LicenseRecord(_FrozenModel):
    spdx: Literal["Apache-2.0", "BSD-2-Clause", "MIT"]
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX_64)
    url: str = Field(min_length=1)


class ControlRecord(_FrozenModel):
    scope: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_file: str = Field(min_length=1)
    scope_marker_count: Literal[1]
    index_chunk_count: int = Field(gt=0)
    payload_sha256: str = Field(pattern=_HEX_64)
    payload_bytes: int = Field(gt=0)


class RepositoryRecord(_FrozenModel):
    repo_id: str = Field(min_length=1)
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    url: str = Field(min_length=1)
    commit: str = Field(pattern=_HEX_40)
    license: LicenseRecord
    languages: tuple[str, ...] = Field(min_length=1)
    corpus_paths: tuple[str, ...] = Field(min_length=2)
    eligible_source_digest_sha256: str = Field(pattern=_HEX_64)
    eligible_file_count: int = Field(gt=0)
    indexed_chunk_count: int = Field(gt=0)
    scope_marker_count: int = Field(gt=0)
    scope_chunk_counts: dict[str, int]
    dominant_scope: str = Field(min_length=1)
    dominant_scope_chunks: int = Field(gt=0)
    eligible_required_scopes: tuple[str, ...] = Field(min_length=1)
    control: ControlRecord
    scope_set_id: str = Field(min_length=1)


class PopulationCounts(_FrozenModel):
    repositories: Literal[16]
    treatment_tasks: Literal[2048]
    single_scope_controls: Literal[16]
    tasks_total: Literal[2064]
    family_counts: dict[str, int]
    kind_counts: dict[str, int]
    tasks_per_repository: Literal[129]


class PopulationTask(_FrozenModel):
    task_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    kind: Literal["single_scope_control", "treatment"]
    family: Literal[
        "cross_scope_dependency",
        "lexical_collision",
        "single_scope_control",
        "weak_participation_control",
    ]
    question: str = Field(min_length=1)
    expected_files: tuple[str, ...] = Field(min_length=1)
    required_scopes: tuple[str, ...] = Field(min_length=1)
    non_dominant_required_scopes: tuple[str, ...]
    budget_tokens: Literal[4000]
    scope_set_id: str = Field(min_length=1)
    required_file_scopes: dict[str, str]
    dominant_scope: str | None = None
    dominant_to_required_chunk_ratio: float | None = None
    collision_term: str | None = None
    dependency: str | None = None
    source_scope: str | None = None
    target_scope: str | None = None
    weak_scope: str | None = None
    weak_term: str | None = None
    control_scope: str | None = None
    control_scope_marker_count: int | None = None
    pre_candidate_payload_sha256: str | None = Field(default=None, pattern=_HEX_64)
    payload_bytes: int | None = None


class ScopeAwarePopulation(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"]
    protocol_id: Literal["R26-CANDIDATE-A"]
    control_archex_revision: Literal["1eda0c85de26b4950490062802b41da9f2e00e68"]
    created_at: str = Field(min_length=1)
    treatment_blind: Literal[True]
    r19_population_substitution: Literal[False]
    scope_definition: dict[str, Any]
    file_eligibility: dict[str, Any]
    index_config: dict[str, Any]
    query_config: dict[str, Any]
    population: PopulationCounts
    repositories: tuple[RepositoryRecord, ...]
    tasks: tuple[PopulationTask, ...]
    population_sha256: str = Field(pattern=_HEX_64)


class WarmRun(_FrozenModel):
    run: Literal[1, 2]
    payload_sha256: str = Field(pattern=_HEX_64)
    payload_bytes: int = Field(gt=0)


class RepositoryControlReceipt(_FrozenModel):
    repository_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=_HEX_40)
    scope: str = Field(min_length=1)
    scope_marker_count: Literal[1]
    question: str = Field(min_length=1)
    expected_file: str = Field(min_length=1)
    expected_file_returned: Literal[True]
    warm_runs: tuple[WarmRun, WarmRun]


class ControlReceipts(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"]
    protocol_id: Literal["R26-CANDIDATE-A"]
    control_archex_revision: Literal["1eda0c85de26b4950490062802b41da9f2e00e68"]
    measured_at: str = Field(min_length=1)
    environment: dict[str, Any]
    payload_projection: tuple[str, ...]
    payload_encoding: str = Field(min_length=1)
    reproduction_command: str = Field(min_length=1)
    repositories: tuple[RepositoryControlReceipt, ...]


class PowerParameters(_FrozenModel):
    base_rate: float
    cluster_sd: float
    effect_points: float
    effect_sd: float
    simulations: Literal[10000]
    resamples: Literal[1000]
    seed: Literal[20260912]


class PowerResult(_FrozenModel):
    power: float = Field(ge=0.0, le=1.0)
    mean_ci_width_percentage_points: float = Field(gt=0.0)
    threshold: float
    passes: Literal[True]


class PrimarySelector(_FrozenModel):
    kind: Literal["treatment"]
    task_count: Literal[2048]
    single_scope_controls_excluded: Literal[16]


class PowerArtifact(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"]
    method: Literal["archex.benchmark.corpus_audit.simulate_power"]
    command: str = Field(min_length=1)
    independent_unit: Literal["repository"]
    cluster_sizes: tuple[int, ...]
    primary_selector: PrimarySelector
    parameters: PowerParameters
    calibration: dict[str, Any]
    result: PowerResult


class ScopeAwarePopulationCoverage(_FrozenModel):
    repositories: int
    tasks: int


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_model(path: Path, model: type[_ModelT]) -> _ModelT:
    try:
        return model.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc:
        raise ScopeAwareCampaignError(f"Invalid {path.name}: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScopeAwareCampaignError(message)


def _validate_repository(repository: RepositoryRecord) -> None:
    _require(
        repository.url == f"https://github.com/{repository.repository}.git",
        f"{repository.repo_id}: repository URL is not pinned to its GitHub repository",
    )
    expected_license_url = f"https://github.com/{repository.repository}/blob/{repository.commit}/{repository.license.path}"
    _require(
        repository.license.url == expected_license_url,
        f"{repository.repo_id}: license URL is not commit-pinned",
    )
    _require(
        repository.indexed_chunk_count == sum(repository.scope_chunk_counts.values()),
        f"{repository.repo_id}: scope chunk counts do not reconcile",
    )
    _require(
        repository.dominant_scope in repository.scope_chunk_counts,
        f"{repository.repo_id}: dominant scope is absent",
    )
    _require(
        repository.scope_chunk_counts[repository.dominant_scope]
        == repository.dominant_scope_chunks,
        f"{repository.repo_id}: dominant chunk count does not reconcile",
    )
    _require(
        repository.control.scope in repository.scope_chunk_counts,
        f"{repository.repo_id}: control scope is absent",
    )
    _require(
        repository.control.index_chunk_count
        == repository.scope_chunk_counts[repository.control.scope],
        f"{repository.repo_id}: control chunk count does not reconcile",
    )
    _require(
        len(set(repository.corpus_paths)) == len(repository.corpus_paths),
        f"{repository.repo_id}: duplicate corpus path",
    )
    _require(
        len(set(repository.eligible_required_scopes)) == len(repository.eligible_required_scopes),
        f"{repository.repo_id}: duplicate eligible required scope",
    )
    for scope in repository.eligible_required_scopes:
        _require(
            scope in repository.scope_chunk_counts,
            f"{repository.repo_id}: eligible required scope {scope!r} is absent",
        )
        _require(
            repository.dominant_scope_chunks / repository.scope_chunk_counts[scope] >= 4.0,
            f"{repository.repo_id}: scope {scope!r} misses the 4:1 imbalance gate",
        )


def _validate_task(task: PopulationTask, repository: RepositoryRecord) -> None:
    _require(
        task.scope_set_id == repository.scope_set_id, f"{task.task_id}: wrong scope-set identity"
    )
    _require(
        len(set(task.expected_files)) == len(task.expected_files),
        f"{task.task_id}: duplicate expected file",
    )
    _require(
        len(set(task.required_scopes)) == len(task.required_scopes),
        f"{task.task_id}: duplicate required scope",
    )
    _require(
        set(task.required_file_scopes) == set(task.expected_files),
        f"{task.task_id}: required-file labels do not reconcile",
    )
    _require(
        set(task.required_file_scopes.values()) == set(task.required_scopes),
        f"{task.task_id}: required-scope labels do not reconcile",
    )
    _require(
        set(task.required_scopes) <= set(repository.scope_chunk_counts),
        f"{task.task_id}: unknown required scope",
    )

    if task.kind == "single_scope_control":
        _require(
            task.family == "single_scope_control", f"{task.task_id}: control has a treatment family"
        )
        _require(
            task.required_scopes == (task.control_scope,),
            f"{task.task_id}: control must name exactly one scope",
        )
        _require(
            task.non_dominant_required_scopes == (),
            f"{task.task_id}: control has non-dominant scopes",
        )
        _require(
            task.control_scope_marker_count == 1,
            f"{task.task_id}: control scope must contain one marker",
        )
        _require(
            task.pre_candidate_payload_sha256 == repository.control.payload_sha256,
            f"{task.task_id}: control payload digest drift",
        )
        _require(
            task.payload_bytes == repository.control.payload_bytes,
            f"{task.task_id}: control payload size drift",
        )
        _require(task.question == repository.control.query, f"{task.task_id}: control query drift")
        _require(
            task.expected_files == (repository.control.expected_file,),
            f"{task.task_id}: control expected file drift",
        )
        _require(
            all(
                value is None
                for value in (
                    task.dominant_scope,
                    task.dominant_to_required_chunk_ratio,
                    task.collision_term,
                    task.dependency,
                    task.source_scope,
                    task.target_scope,
                    task.weak_scope,
                    task.weak_term,
                )
            ),
            f"{task.task_id}: control carries treatment labels",
        )
        return

    _require(
        task.family != "single_scope_control", f"{task.task_id}: treatment uses control family"
    )
    _require(
        task.control_scope is None
        and task.control_scope_marker_count is None
        and task.pre_candidate_payload_sha256 is None
        and task.payload_bytes is None,
        f"{task.task_id}: treatment carries control labels",
    )
    _require(
        task.dominant_scope == repository.dominant_scope, f"{task.task_id}: dominant scope drift"
    )
    _require(
        bool(task.non_dominant_required_scopes), f"{task.task_id}: no non-dominant required scope"
    )
    _require(
        set(task.non_dominant_required_scopes) <= set(task.required_scopes),
        f"{task.task_id}: non-dominant scopes are not required",
    )
    _require(
        task.dominant_scope not in task.non_dominant_required_scopes,
        f"{task.task_id}: dominant scope is labeled non-dominant",
    )
    ratios = [
        repository.dominant_scope_chunks / repository.scope_chunk_counts[scope]
        for scope in task.non_dominant_required_scopes
    ]
    expected_ratio = round(min(ratios), 6)
    _require(expected_ratio >= 4.0, f"{task.task_id}: misses the 4:1 imbalance gate")
    _require(
        task.dominant_to_required_chunk_ratio == expected_ratio,
        f"{task.task_id}: recorded imbalance ratio drift",
    )

    family_fields = {
        "lexical_collision": (
            (task.collision_term,),
            (
                task.dependency,
                task.source_scope,
                task.target_scope,
                task.weak_scope,
                task.weak_term,
            ),
        ),
        "cross_scope_dependency": (
            (task.dependency, task.source_scope, task.target_scope),
            (task.collision_term, task.weak_scope, task.weak_term),
        ),
        "weak_participation_control": (
            (task.weak_scope, task.weak_term),
            (task.collision_term, task.dependency, task.source_scope, task.target_scope),
        ),
    }
    required_fields, forbidden_fields = family_fields[task.family]
    _require(all(required_fields), f"{task.task_id}: missing {task.family} label")
    _require(
        all(value is None for value in forbidden_fields),
        f"{task.task_id}: carries labels from another task family",
    )
    if task.family == "lexical_collision":
        _require(
            len(task.expected_files) >= 2,
            f"{task.task_id}: lexical collision needs at least two labeled files",
        )
    elif task.family == "cross_scope_dependency":
        _require(
            task.source_scope in task.required_scopes and task.target_scope in task.required_scopes,
            f"{task.task_id}: dependency scopes are not both required",
        )
    else:
        _require(
            task.weak_scope in repository.scope_chunk_counts
            and task.weak_scope not in task.required_scopes,
            f"{task.task_id}: weak scope must be an existing non-required scope",
        )


def _validate_receipts(
    receipts: ControlReceipts, repositories: dict[str, RepositoryRecord]
) -> None:
    _require(
        receipts.payload_projection == EXPECTED_PAYLOAD_PROJECTION,
        "control receipt payload projection drift",
    )
    _require(
        receipts.payload_encoding
        == "UTF-8 canonical JSON: sort_keys=true, separators=(comma,colon), ensure_ascii=false",
        "control receipt payload encoding drift",
    )
    _require(
        receipts.environment
        == {
            "archex": "0.30.0",
            "architecture": "arm64",
            "cpu": "Apple M1 Pro",
            "index_config": IndexConfig().model_dump(mode="json"),
            "kernel": "Darwin 25.6.0",
            "os": "darwin",
            "python": "3.11.11",
            "query": {
                "explicit_token_budget": True,
                "lifecycle": (
                    "one unmeasured index/query warm-up followed by two measured "
                    "warm queries in one process"
                ),
                "parallel": False,
                "python_hash_seed": 0,
                "refresh": False,
                "token_budget": 4000,
            },
            "uv": "0.6.14",
        },
        "control receipt environment drift",
    )
    _require(
        receipts.reproduction_command
        == (
            'cd "$CONTROL_WORKTREE" && PYTHONHASHSEED=0 uv run python '
            '"$CAMPAIGN_CHECKOUT/benchmarks/campaigns/r27_scope_aware/'
            'reproduce_controls.py" --population "$CAMPAIGN_CHECKOUT/benchmarks/'
            'campaigns/r27_scope_aware/population.json" --receipts '
            '"$CAMPAIGN_CHECKOUT/benchmarks/campaigns/r27_scope_aware/'
            'control_receipts.json" --repo-cache "$REPO_CACHE"'
        ),
        "control reproduction command drift",
    )
    _require(len(receipts.repositories) == 16, "control receipts must cover 16 repositories")
    _require(
        len({receipt.repository_id for receipt in receipts.repositories}) == 16,
        "duplicate control receipt repository",
    )
    for receipt in receipts.repositories:
        repository = repositories.get(receipt.repository_id)
        _require(
            repository is not None, f"unknown control receipt repository {receipt.repository_id!r}"
        )
        assert repository is not None
        _require(
            receipt.repository == repository.repository and receipt.commit == repository.commit,
            f"{receipt.repository_id}: control receipt identity drift",
        )
        _require(
            receipt.scope == repository.control.scope,
            f"{receipt.repository_id}: control receipt scope drift",
        )
        _require(
            receipt.question == repository.control.query,
            f"{receipt.repository_id}: control receipt query drift",
        )
        _require(
            receipt.expected_file == repository.control.expected_file,
            f"{receipt.repository_id}: control receipt file drift",
        )
        _require(
            tuple(run.run for run in receipt.warm_runs) == (1, 2),
            f"{receipt.repository_id}: warm run sequence drift",
        )
        for run in receipt.warm_runs:
            _require(
                run.payload_sha256 == repository.control.payload_sha256,
                f"{receipt.repository_id}: warm payload digest drift",
            )
            _require(
                run.payload_bytes == repository.control.payload_bytes,
                f"{receipt.repository_id}: warm payload size drift",
            )


def validate_scope_aware_population(directory: Path) -> ScopeAwarePopulationCoverage:
    """Validate the R27 population, control receipts, and power artifact."""
    population_path = directory / POPULATION_FILENAME
    try:
        raw_population = cast("dict[str, Any]", json.loads(population_path.read_text()))
        population = ScopeAwarePopulation.model_validate(raw_population)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ScopeAwareCampaignError(f"Invalid {population_path.name}: {exc}") from exc
    receipts = _load_model(directory / CONTROL_RECEIPTS_FILENAME, ControlReceipts)
    power = _load_model(directory / POWER_FILENAME, PowerArtifact)

    population_without_digest = {
        key: value for key, value in raw_population.items() if key != "population_sha256"
    }
    _require(
        _canonical_sha256(population_without_digest) == population.population_sha256,
        "population digest drift",
    )
    _require(
        population.scope_definition
        == {
            "markers": ["package.json", "Cargo.toml"],
            "ownership": "deepest marker-root prefix wins",
            "path_normalization": "repository-relative POSIX paths sorted bytewise",
            "empty_scopes": "discarded after file eligibility",
            "shared_statistics": (
                "one IndexStore and one repository-wide BM25 FTS corpus for every "
                "task corpus; no per-scope rebuild"
            ),
        },
        "scope definition drift",
    )
    _require(
        population.file_eligibility
        == {
            "languages_by_repository": True,
            "ignores": "src/archex/acquire/discovery.py DEFAULT_IGNORES at control revision",
            "max_file_size": 10_000_000,
            "binary_check": "existing discover_files UTF-8/NUL rules",
            "tracked_or_unignored": "existing git ls-files --cached --others --exclude-standard",
        },
        "file-eligibility contract drift",
    )
    _require(
        population.index_config == IndexConfig().model_dump(mode="json"),
        "index configuration drift",
    )
    _require(
        population.query_config
        == {
            "token_budget": 4000,
            "explicit_token_budget": True,
            "refresh": False,
            "cache_lifecycle": "build the pinned scoped index once, then measure warm query calls",
            "payload_projection": list(EXPECTED_PAYLOAD_PROJECTION),
            "payload_encoding": (
                "UTF-8 canonical JSON: sort_keys=true, separators=(comma,colon), ensure_ascii=false"
            ),
        },
        "query configuration drift",
    )

    _require(len(population.repositories) == 16, "population must contain 16 repositories")
    repositories = {repository.repo_id: repository for repository in population.repositories}
    _require(len(repositories) == 16, "duplicate repository identity")
    _require(
        len({repository.repository for repository in population.repositories}) == 16,
        "duplicate repository pin",
    )
    _require(
        len({repository.scope_set_id for repository in population.repositories}) == 16,
        "duplicate scope-set identity",
    )
    for repository in population.repositories:
        _validate_repository(repository)

    _require(len(population.tasks) == 2064, "population must contain 2,064 tasks")
    _require(len({task.task_id for task in population.tasks}) == 2064, "duplicate task identity")
    repository_counts = Counter(task.repository_id for task in population.tasks)
    _require(
        repository_counts == Counter({repo_id: 129 for repo_id in repositories}),
        "each repository must contribute 129 tasks",
    )
    family_counts = Counter(task.family for task in population.tasks)
    kind_counts = Counter(task.kind for task in population.tasks)
    _require(
        dict(sorted(family_counts.items())) == EXPECTED_FAMILY_COUNTS, "task-family counts drift"
    )
    _require(dict(sorted(kind_counts.items())) == EXPECTED_KIND_COUNTS, "task-kind counts drift")
    _require(
        population.population.family_counts == EXPECTED_FAMILY_COUNTS,
        "recorded task-family counts drift",
    )
    _require(
        population.population.kind_counts == EXPECTED_KIND_COUNTS, "recorded task-kind counts drift"
    )
    for repo_id in repositories:
        repo_families = {task.family for task in population.tasks if task.repository_id == repo_id}
        _require(
            repo_families == set(EXPECTED_FAMILY_COUNTS),
            f"{repo_id}: incomplete task-family coverage",
        )
    for task in population.tasks:
        repository = repositories.get(task.repository_id)
        _require(repository is not None, f"{task.task_id}: unknown repository")
        assert repository is not None
        _validate_task(task, repository)

    _validate_receipts(receipts, repositories)
    _require(
        power.command
        == (
            'uv run python -c "from archex.benchmark.corpus_audit import '
            "simulate_power; print(simulate_power([128] * 16, base_rate=0.5, "
            "cluster_sd=0.08, effect_points=5.0, effect_sd=0.0, "
            'simulations=10000, resamples=1000, seed=20260912))"'
        ),
        "power reproduction command drift",
    )
    _require(
        power.calibration
        == {
            "artifact": "benchmarks/evidence/s2-corpus-validity.json",
            "balanced_clusters_note": (
                "with 128 tasks per repository, pooled simulated mean equals "
                "equal-weight repository mean"
            ),
            "effect_sd_basis": ("estimated between-repository spread was below sampling noise"),
        },
        "power calibration drift",
    )
    _require(power.cluster_sizes == (128,) * 16, "power cluster sizes drift")
    _require(
        power.parameters.model_dump()
        == {
            "base_rate": 0.5,
            "cluster_sd": 0.08,
            "effect_points": 5.0,
            "effect_sd": 0.0,
            "simulations": 10000,
            "resamples": 1000,
            "seed": 20260912,
        },
        "power parameters drift",
    )
    _require(power.result.threshold == 0.8, "power threshold drift")
    _require(power.result.power == 0.9048, "power result drift")
    _require(
        power.result.mean_ci_width_percentage_points == 5.7060986328125,
        "power interval-width result drift",
    )
    _require(power.result.power >= power.result.threshold, "population misses the 0.80 power gate")

    return ScopeAwarePopulationCoverage(repositories=16, tasks=2064)


class CampaignArtifactDigests(_FrozenModel):
    population_file_sha256: str = Field(pattern=_HEX_64)
    population_sha256: str = Field(pattern=_HEX_64)
    control_receipts_file_sha256: str = Field(pattern=_HEX_64)
    power_file_sha256: str = Field(pattern=_HEX_64)
    cells_file_sha256: str = Field(pattern=_HEX_64)
    cells_sha256: str = Field(pattern=_HEX_64)
    repository_identity_sha256: str = Field(pattern=_HEX_64)
    task_label_sha256: str = Field(pattern=_HEX_64)


class ScopeAwareCell(_FrozenModel):
    cell_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    kind: Literal["single_scope_control", "treatment"]
    family: Literal[
        "cross_scope_dependency",
        "lexical_collision",
        "single_scope_control",
        "weak_participation_control",
    ]
    arm: Literal["archex_query_control", "scope_aware_candidate"]
    eligibility: Literal["planned"]


class ScopeAwareCells(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"]
    protocol_id: Literal["R26-CANDIDATE-A"]
    population_sha256: str = Field(pattern=_HEX_64)
    arms: tuple[Literal["archex_query_control", "scope_aware_candidate"], ...]
    planned_cells: Literal[4128]
    cells: tuple[ScopeAwareCell, ...]
    cells_sha256: str = Field(pattern=_HEX_64)


class ScopeAwareManifest(_FrozenModel):
    schema_version: Literal[1]
    campaign_id: Literal["r27-scope-aware-monorepo-ranking"]
    protocol_id: Literal["R26-CANDIDATE-A"]
    status: Literal["frozen-before-candidate-source"]
    frozen_at: str = Field(min_length=1)
    control_archex_revision: Literal["1eda0c85de26b4950490062802b41da9f2e00e68"]
    r26_preregistration: Literal[
        "benchmarks/preregistrations/R26-graft-informed-retrieval-candidate-tests.md"
    ]
    graft_inspiration: dict[str, Any]
    artifact_digests: CampaignArtifactDigests
    population_contract: dict[str, Any]
    repository_identity: tuple[dict[str, Any], ...]
    control_payload_sha256: dict[str, str]
    candidate_source_binding: dict[str, Any]
    arms: tuple[dict[str, Any], ...]
    commands: dict[str, Any]
    candidate_interface: dict[str, Any]
    indexing_and_packing: dict[str, Any]
    determinism: dict[str, Any]
    inference: dict[str, Any]
    receipts: dict[str, Any]
    binding_gates: dict[str, Any]
    failure_policy: dict[str, Any]
    decision_rules: dict[str, Any]
    prohibitions: tuple[str, ...]
    manifest_sha256: str = Field(pattern=_HEX_64)


class ScopeAwareCampaignCoverage(_FrozenModel):
    repositories: int
    tasks: int
    cells: int


def _load_raw_model(path: Path, model: type[_ModelT]) -> tuple[dict[str, Any], _ModelT]:
    try:
        raw = cast("dict[str, Any]", json.loads(path.read_text()))
        return raw, model.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ScopeAwareCampaignError(f"Invalid {path.name}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ScopeAwareCampaignError(f"Cannot read {path.name}: {exc}") from exc


def _expected_repository_identity(
    repositories: tuple[RepositoryRecord, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "repository_id": repository.repo_id,
            "repository": repository.repository,
            "url": repository.url,
            "commit": repository.commit,
            "license": repository.license.model_dump(mode="json"),
            "eligible_source_digest_sha256": repository.eligible_source_digest_sha256,
            "scope_set_id": repository.scope_set_id,
        }
        for repository in repositories
    ]


def _expected_task_labels(tasks: tuple[PopulationTask, ...]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "repository_id": task.repository_id,
            "kind": task.kind,
            "family": task.family,
            "question": task.question,
            "expected_files": list(task.expected_files),
            "required_scopes": list(task.required_scopes),
            "non_dominant_required_scopes": list(task.non_dominant_required_scopes),
            "required_file_scopes": task.required_file_scopes,
        }
        for task in tasks
    ]


def validate_scope_aware_campaign(directory: Path) -> ScopeAwareCampaignCoverage:
    """Validate the immutable R27 manifest and complete eligible cell matrix."""
    population_coverage = validate_scope_aware_population(directory)
    population = _load_model(directory / POPULATION_FILENAME, ScopeAwarePopulation)
    receipts = _load_model(directory / CONTROL_RECEIPTS_FILENAME, ControlReceipts)
    cells_raw, cells = _load_raw_model(directory / CELLS_FILENAME, ScopeAwareCells)
    manifest_raw, manifest = _load_raw_model(directory / MANIFEST_FILENAME, ScopeAwareManifest)

    cells_without_digest = {key: value for key, value in cells_raw.items() if key != "cells_sha256"}
    _require(
        _canonical_sha256(cells_without_digest) == cells.cells_sha256,
        "cell-matrix digest drift",
    )
    manifest_without_digest = {
        key: value for key, value in manifest_raw.items() if key != "manifest_sha256"
    }
    _require(
        _canonical_sha256(manifest_without_digest) == manifest.manifest_sha256,
        "manifest digest drift",
    )

    _require(
        cells.population_sha256 == population.population_sha256,
        "cell matrix population identity drift",
    )
    _require(
        cells.arms == ("archex_query_control", "scope_aware_candidate"),
        "cell arm order drift",
    )
    _require(len(cells.cells) == 4128, "cell matrix must contain 4,128 cells")
    _require(
        len({cell.cell_id for cell in cells.cells}) == 4128,
        "duplicate cell identity",
    )

    tasks = {task.task_id: task for task in population.tasks}
    expected_pairs = {
        (task_id, arm)
        for task_id in tasks
        for arm in ("archex_query_control", "scope_aware_candidate")
    }
    observed_pairs = {(cell.task_id, cell.arm) for cell in cells.cells}
    _require(observed_pairs == expected_pairs, "missing or undeclared campaign cell")
    for cell in cells.cells:
        task = tasks[cell.task_id]
        _require(
            cell.cell_id == f"{cell.task_id}::{cell.arm}",
            f"{cell.cell_id}: non-canonical cell identity",
        )
        _require(
            (
                cell.repository_id,
                cell.kind,
                cell.family,
            )
            == (
                task.repository_id,
                task.kind,
                task.family,
            ),
            f"{cell.cell_id}: task metadata drift",
        )

    expected_file_digests = {
        "population_file_sha256": _file_sha256(directory / POPULATION_FILENAME),
        "population_sha256": population.population_sha256,
        "control_receipts_file_sha256": _file_sha256(directory / CONTROL_RECEIPTS_FILENAME),
        "power_file_sha256": _file_sha256(directory / POWER_FILENAME),
        "cells_file_sha256": _file_sha256(directory / CELLS_FILENAME),
        "cells_sha256": cells.cells_sha256,
        "repository_identity_sha256": _canonical_sha256(
            _expected_repository_identity(population.repositories)
        ),
        "task_label_sha256": _canonical_sha256(_expected_task_labels(population.tasks)),
    }
    _require(
        manifest.artifact_digests.model_dump() == expected_file_digests,
        "manifest artifact identity drift",
    )
    _require(
        list(manifest.repository_identity)
        == _expected_repository_identity(population.repositories),
        "manifest repository identity drift",
    )
    _require(
        manifest.control_payload_sha256
        == {
            repository.repo_id: repository.control.payload_sha256
            for repository in population.repositories
        },
        "manifest control payload identity drift",
    )
    _require(
        manifest.determinism.get("operator_environment") == receipts.environment,
        "manifest operator environment drift",
    )
    _require(
        manifest.population_contract
        == {
            "repositories": 16,
            "tasks_total": 2064,
            "treatment_tasks": 2048,
            "single_scope_controls": 16,
            "tasks_per_repository": 129,
            "family_counts": EXPECTED_FAMILY_COUNTS,
            "r19_population_substitution": False,
            "minimum_dominant_to_required_chunk_ratio": 4.0,
            "primary_selector": {
                "field": "kind",
                "equals": "treatment",
                "task_count": 2048,
            },
            "invariant_selector": {
                "field": "kind",
                "equals": "single_scope_control",
                "task_count": 16,
            },
        },
        "manifest population contract drift",
    )
    _require(
        manifest.candidate_source_binding.get("status") == "unbound-until-r29"
        and manifest.candidate_source_binding.get("revision") is None
        and manifest.candidate_source_binding.get("identity_artifact") is None,
        "candidate source identity must remain unbound until R29",
    )
    _require(
        manifest.candidate_interface.get("candidate_limit_per_scope") == 150
        and manifest.candidate_interface.get("normalization", {}).get("formula")
        == "normalized_score = raw_score / repository_global_max_raw_score"
        and manifest.candidate_interface.get("participation", {}).get("operator") == ">="
        and manifest.candidate_interface.get("participation", {}).get("threshold") == 0.25
        and manifest.candidate_interface.get("single_scope")
        == (
            "bypass candidate ranking and delegate retrieval, assembly, packing, "
            "and payload serialization to unchanged archex_query"
        )
        and manifest.candidate_interface.get("serialization_delta")
        == "none in ContextBundle; scope receipt is a separate benchmark sidecar",
        "candidate interface drift",
    )
    _require(
        manifest.inference.get("independent_unit") == "repository"
        and manifest.inference.get("primary_selector")
        == {
            "kind": "treatment",
            "paired_tasks": 2048,
            "single_scope_controls_excluded": 16,
        }
        and manifest.inference.get("margins") == {"MWG": 0.05, "NIM": -0.02, "EQM": 0.02}
        and manifest.inference.get("bootstrap")
        == {
            "method": "whole-repository percentile bootstrap",
            "resamples": 10000,
            "seed": 20260913,
            "confidence_interval": 0.95,
        },
        "campaign inference contract drift",
    )
    _require(
        manifest.binding_gates.get("warm_p95_milliseconds") == 3000
        and manifest.binding_gates.get("complete_cells")
        == "every declared cell appears exactly once as success or recorded failure",
        "campaign binding gates drift",
    )
    _require(
        manifest.failure_policy.get("planned_cells") == 4128
        and manifest.failure_policy.get("success_or_recorded_failure") is True
        and manifest.failure_policy.get("missing_cell") == "invalid campaign result"
        and manifest.failure_policy.get("duplicate_cell") == "invalid campaign result"
        and manifest.failure_policy.get("undeclared_cell") == "invalid campaign result"
        and manifest.failure_policy.get("failed_cell_primary_score") == 0.0,
        "campaign failure policy drift",
    )
    _require(
        manifest.decision_rules.get("continuation_requires")
        == [
            "primary point estimate >= +0.05",
            "95% interval lower bound > 0",
            "every binding gate passes",
        ]
        and manifest.decision_rules.get("compatible_only_with_NIM_or_EQM") == "EVIDENCE NO-GO"
        and manifest.decision_rules.get("any_binding_gate_failure") == "EVIDENCE NO-GO",
        "terminal decision rules drift",
    )
    _require(cells.cells_sha256 == EXPECTED_CELLS_SHA256, "unexpected cell-matrix identity")
    _require(
        manifest.manifest_sha256 == EXPECTED_MANIFEST_SHA256,
        "unexpected immutable manifest identity",
    )
    _require(
        _file_sha256(directory / MANIFEST_FILENAME) == EXPECTED_MANIFEST_FILE_SHA256,
        "unexpected manifest file identity",
    )

    return ScopeAwareCampaignCoverage(
        repositories=population_coverage.repositories,
        tasks=population_coverage.tasks,
        cells=4128,
    )
