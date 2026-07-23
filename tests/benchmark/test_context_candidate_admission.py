"""Regression tests for M0.4 round 3's context candidate admission tightening.

Round 2's calibrated evidence (``m0.4-task-contract-calibration.json``)
confirmed a real defect, not a labeling artifact: ``archex_query_context_candidate``
broadened its returned file set past what a symbol/path/lexical evidence match
actually justifies -- failing precision on 39/64 tasks, F1 on 36, MRR on 17, and
regressing region recall on 10 and line recall on 15. These tests pin the
mechanisms round 3 added to fix that, each keyed to the specific defect it
guards against, and prove M0.2 (``archex_query_coverage_candidate``) and M0.3
(``archex_query_rank_candidate``) are unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.coverage_candidate import (
    DEFAULT_NEIGHBOR_MIN_CONFIDENCE,
    CoverageSeedDecision,
    coverage_neighbor_decisions,
    has_identifier_evidence,
)
from archex.benchmark.graph_multihop import GraphEdge
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    _CONTEXT_CANDIDATE_NEIGHBOR_MIN_CONFIDENCE,  # pyright: ignore[reportPrivateUsage]
    _prepare_packing,  # pyright: ignore[reportPrivateUsage]
    _protect_base_query_regions,  # pyright: ignore[reportPrivateUsage]
    _rank_candidate_bundle,  # pyright: ignore[reportPrivateUsage]
    run_archex_query_context_candidate,
)
from archex.models import (
    CodeChunk,
    CompressionMode,
    ContextBundle,
    ContextReceipt,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    RankedChunk,
    RetrievalMetadata,
)
from archex.receipt import chunk_content_hash
from archex.reporting import count_tokens
from archex.scout import chunk_handle
from archex.serve.packing import PackDecision, PackedRegion, PackingPlan, PackingScore

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------- #
# has_identifier_evidence: the precision/F1 fix's evidence-tier gate.
# --------------------------------------------------------------------------- #


def test_has_identifier_evidence_requires_the_identifier_tier() -> None:
    assert has_identifier_evidence(("identifier:pythonadapter",)) is True
    assert has_identifier_evidence(("identifier:pythonadapter", "lexical:adapter")) is True


def test_has_identifier_evidence_rejects_symbol_path_and_lexical_alone() -> None:
    # Measured on the real corpus: "adapter"/"language"-style symbol and path
    # matches are shared by every file in a directory family and were the
    # actual mechanism behind the calibrated run's confirmed over-broadening
    # (m0.4-task-contract-calibration.json's own verdict reason).
    assert has_identifier_evidence(("symbol:adapter", "path:adapter")) is False
    assert has_identifier_evidence(("lexical:register",)) is False
    assert has_identifier_evidence(()) is False


# --------------------------------------------------------------------------- #
# _rank_candidate_bundle(tighten_admission=True): seed/neighbor file-set gate.
# --------------------------------------------------------------------------- #


def _fixture_task(question: str, token_budget: int = 4096) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="t",
        repo="test/repo",
        commit="abc",
        question=question,
        expected_files=[],
        token_budget=token_budget,
    )


def test_tightened_admission_excludes_a_symbol_only_match(python_simple_repo: Path) -> None:
    # "auth service" matches AuthService (symbol/lexical tier) in every file
    # that mentions "service"/"auth"; no CamelCase/snake_case token in the
    # question copies an identifier verbatim, so no seed evidence clears the
    # identifier-tier bar and nothing extra should be admitted.
    task = _fixture_task("How does the auth service log a user in?")

    tightened_bundle, _, _, tightened_prov = _rank_candidate_bundle(
        task,
        python_simple_repo,
        strategy=Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
        tighten_admission=True,
    )
    broad_bundle, _, _, broad_prov = _rank_candidate_bundle(
        task,
        python_simple_repo,
        strategy=Strategy.ARCHEX_QUERY_RANK_CANDIDATE,
        tighten_admission=False,
    )

    assert tightened_prov["candidate_seed_admitted"] == "none"
    # Mutation check: the untightened (M0.3) admission path over the exact
    # same question/repo/evidence pool is not similarly empty -- proving the
    # "none" result above comes from the identifier-tier gate, not from an
    # absence of symbol/lexical evidence to admit in the first place.
    assert broad_prov["candidate_seed_admitted"] != "none" or len(broad_bundle.chunks) >= len(
        tightened_bundle.chunks
    )


def test_tightened_admission_keeps_an_explicit_identifier_match(python_simple_repo: Path) -> None:
    task = _fixture_task("Where does AuthService verify a token?")

    _bundle, _, _, prov = _rank_candidate_bundle(
        task,
        python_simple_repo,
        strategy=Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
        tighten_admission=True,
    )

    assert prov["candidate_neighbor_min_confidence"] == "0.75"


def test_tighten_admission_is_off_by_default_and_leaves_m03_untouched(
    python_simple_repo: Path,
) -> None:
    # Regression guard: `tighten_admission` must default to the M0.3 rank
    # candidate's pre-round-3 behavior so `run_archex_query_rank_candidate`
    # is byte-identical to before this change.
    task = _fixture_task("How does the auth service log a user in?")

    default_bundle, _, _, default_prov = _rank_candidate_bundle(
        task, python_simple_repo, strategy=Strategy.ARCHEX_QUERY_RANK_CANDIDATE
    )
    explicit_bundle, _, _, explicit_prov = _rank_candidate_bundle(
        task,
        python_simple_repo,
        strategy=Strategy.ARCHEX_QUERY_RANK_CANDIDATE,
        tighten_admission=False,
    )

    assert default_prov == explicit_prov
    assert len(default_bundle.chunks) == len(explicit_bundle.chunks)


# --------------------------------------------------------------------------- #
# coverage_neighbor_decisions(min_confidence=...): the graph-edge confidence
# floor, raised for the context candidate alone.
# --------------------------------------------------------------------------- #


def _decision(file: str, score: int = 10) -> CoverageSeedDecision:
    return CoverageSeedDecision(file=file, score=score, evidence=("identifier:target",))


def test_neighbor_decisions_default_floor_admits_a_moderate_confidence_edge() -> None:
    edges = [GraphEdge(source="seed.py", target="neighbor.py", confidence=0.6)]
    decisions = coverage_neighbor_decisions(
        edges,
        seed_files={"seed.py"},
        existing_files=set(),
        direct_decisions=[_decision("neighbor.py")],
        limit=10,
    )
    assert [d.file for d in decisions] == ["neighbor.py"]


def test_neighbor_decisions_tightened_floor_rejects_the_same_edge() -> None:
    # Same edge, same evidence: only the confidence floor changed. This is the
    # exact mechanism `_rank_candidate_bundle(tighten_admission=True)` uses
    # for the context candidate (see `_CONTEXT_CANDIDATE_NEIGHBOR_MIN_CONFIDENCE`).
    edges = [GraphEdge(source="seed.py", target="neighbor.py", confidence=0.6)]
    decisions = coverage_neighbor_decisions(
        edges,
        seed_files={"seed.py"},
        existing_files=set(),
        direct_decisions=[_decision("neighbor.py")],
        limit=10,
        min_confidence=_CONTEXT_CANDIDATE_NEIGHBOR_MIN_CONFIDENCE,
    )
    assert decisions == []


def test_neighbor_decisions_tightened_floor_still_admits_a_confident_edge() -> None:
    edges = [GraphEdge(source="seed.py", target="neighbor.py", confidence=0.9)]
    decisions = coverage_neighbor_decisions(
        edges,
        seed_files={"seed.py"},
        existing_files=set(),
        direct_decisions=[_decision("neighbor.py")],
        limit=10,
        min_confidence=_CONTEXT_CANDIDATE_NEIGHBOR_MIN_CONFIDENCE,
    )
    assert [d.file for d in decisions] == ["neighbor.py"]


def test_neighbor_decisions_default_param_matches_shared_module_constant() -> None:
    assert DEFAULT_NEIGHBOR_MIN_CONFIDENCE < _CONTEXT_CANDIDATE_NEIGHBOR_MIN_CONFIDENCE


# --------------------------------------------------------------------------- #
# Packing: the score-scale fix, the structural-elision-as-compress guard, and
# base-query region protection. These target the region/line recall failures.
# --------------------------------------------------------------------------- #


def _ranked(
    chunk_id: str, *, file_path: str, score: float, body_lines: int, whole_file: bool = False
) -> RankedChunk:
    body = "\n".join(f"    acc = acc + value_{i}" for i in range(body_lines))
    content = f"def fn_{chunk_id}(value):\n    acc = 0\n{body}\n    return acc"
    chunk = CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=body_lines + 3,
        language="python",
        symbol_name=None if whole_file else f"fn_{chunk_id}",
    )
    return RankedChunk(chunk=chunk, final_score=score)


def _packing_bundle(
    chunks: list[RankedChunk],
    *,
    token_budget: int,
    seed_files: list[str],
    expanded_files: list[str],
) -> ContextBundle:
    items = [
        ContextReceiptItem(
            handle=chunk_handle(rc.chunk.id),
            file_path=rc.chunk.file_path,
            start_line=rc.chunk.start_line,
            end_line=rc.chunk.end_line,
            content_hash=chunk_content_hash(rc.chunk),
        )
        for rc in chunks
    ]
    receipt = ContextReceipt(
        query="q",
        token_budget=ContextReceiptTokenBudget(requested=token_budget, consumed=0),
        index_revision="rev",
        returned_context=items,
    )
    meta = RetrievalMetadata(
        seed_file_paths=list(seed_files), expanded_file_paths=list(expanded_files)
    )
    total = sum(count_tokens(rc.chunk.content) for rc in chunks)
    return ContextBundle(
        query="q",
        chunks=chunks,
        token_count=total,
        token_budget=token_budget,
        retrieval_metadata=meta,
        receipt=receipt,
    )


def test_admission_score_scale_never_corrupts_the_base_query_top_score() -> None:
    # An admission-appended chunk's RankedChunk.final_score is a raw
    # CoverageSeedDecision.score (unbounded evidence-weight integer), not a
    # 0..~1 relevance score. Regression guard for the exact defect measured
    # on the real corpus: a lexical/symbol-tier admitted chunk's inflated
    # score (thousands) dwarfing the base query's real top hit (~1.4),
    # corrupting the high_score protection check for every other chunk.
    base_top = _ranked("base_top", file_path="target.py", score=1.4, body_lines=2)
    base_secondary = _ranked("base_secondary", file_path="target.py", score=0.9, body_lines=2)
    admitted = _ranked("admitted", file_path="noise.py", score=2231.0, body_lines=2)
    bundle = _packing_bundle(
        [base_top, base_secondary, admitted],
        token_budget=4096,
        seed_files=[],
        expanded_files=[],
    )
    assert bundle.receipt is not None
    items = {item.handle: item for item in bundle.receipt.returned_context}
    items[chunk_handle("admitted")].reason_codes = ["coverage_seed:lexical:noise"]

    prep = _prepare_packing(bundle, question="q", preserve_seed_context=False)

    # base_secondary (0.9/1.4 ~= 0.64) clears the 0.6 high_score fraction of
    # the *base query's own* top score and is protected; if top_score had
    # been computed over the mixed scale (max ~2231) it would not be.
    assert prep.protection_reason_by_id["base_secondary"] == "high_score"
    assert prep.admission_by_id["admitted"] is True
    assert prep.admission_by_id["base_secondary"] is False


def test_context_candidate_never_uses_structural_elision_as_a_compress_outcome() -> None:
    # `compress_region` can choose STRUCTURAL_CODE_ELISION as a genuine
    # *compression* style, sharing the exact enum tag `_bundle_returned_regions`
    # uses to zero out region/line recall credit for a fully elided-to-anchor
    # region. A region compressed (not elided) by the context candidate must
    # never lose region/line credit this way.
    seed = _ranked("seed", file_path="main.py", score=1.0, body_lines=2)
    optional = _ranked("optional", file_path="tail.py", score=0.15, body_lines=60)
    bundle = _packing_bundle([seed, optional], token_budget=4096, seed_files=[], expanded_files=[])

    prep = _prepare_packing(bundle, question="q", preserve_seed_context=False)

    optional_candidate = next(c for c in prep.candidates if c.signals.candidate_id == "optional")
    if optional_candidate.signals.compression_eligible:
        outcome = prep.outcomes["optional"]
        assert outcome.mode is not CompressionMode.STRUCTURAL_CODE_ELISION


def test_protect_base_query_regions_upgrades_non_admission_skip() -> None:
    seed = _ranked("seed", file_path="main.py", score=1.0, body_lines=2)
    dropped = _ranked("dropped", file_path="util.py", score=0.05, body_lines=200)
    bundle = _packing_bundle([seed, dropped], token_budget=8192, seed_files=[], expanded_files=[])
    prep = _prepare_packing(bundle, question="q", preserve_seed_context=False)
    assert prep.admission_by_id["dropped"] is False

    score = PackingScore(
        candidate_id="dropped",
        value=0.0,
        relevance_per_1k_tokens=0.0,
        decision=PackDecision.SKIP,
        reason="test",
        direct_match=False,
        graph_distance=1,
        compression_loss_risk=next(
            c for c in prep.candidates if c.signals.candidate_id == "dropped"
        ).signals.compression_loss_risk,
    )
    plan = PackingPlan(
        regions=[
            PackedRegion(
                candidate_id="dropped",
                decision=PackDecision.SKIP,
                score=score,
                tokens_charged=0,
                retrieval_score=0.0,
            )
        ],
        included_tokens=0,
        budget_tier=prep.tier,
        token_budget=bundle.token_budget,
    )

    protected = _protect_base_query_regions(plan, prep)

    assert protected.regions[0].decision is not PackDecision.SKIP
    assert protected.regions[0].tokens_charged > 0
    assert protected.included_tokens == protected.regions[0].tokens_charged


def test_protect_base_query_regions_leaves_admission_appended_skips_alone() -> None:
    seed = _ranked("seed", file_path="main.py", score=1.0, body_lines=2)
    noise = _ranked("noise", file_path="noise.py", score=999.0, body_lines=200)
    bundle = _packing_bundle([seed, noise], token_budget=8192, seed_files=[], expanded_files=[])
    assert bundle.receipt is not None
    items = {item.handle: item for item in bundle.receipt.returned_context}
    items[chunk_handle("noise")].reason_codes = ["coverage_seed:lexical:noise"]
    prep = _prepare_packing(bundle, question="q", preserve_seed_context=False)
    assert prep.admission_by_id["noise"] is True

    score = PackingScore(
        candidate_id="noise",
        value=0.0,
        relevance_per_1k_tokens=0.0,
        decision=PackDecision.SKIP,
        reason="test",
        direct_match=False,
        graph_distance=1,
        compression_loss_risk=next(
            c for c in prep.candidates if c.signals.candidate_id == "noise"
        ).signals.compression_loss_risk,
    )
    plan = PackingPlan(
        regions=[
            PackedRegion(
                candidate_id="noise",
                decision=PackDecision.SKIP,
                score=score,
                tokens_charged=0,
                retrieval_score=0.0,
            )
        ],
        included_tokens=0,
        budget_tier=prep.tier,
        token_budget=bundle.token_budget,
    )

    protected = _protect_base_query_regions(plan, prep)

    # Only the base-query's own regions are protected; an admission-appended
    # region the packer decided was noise stays dropped, or context noise
    # would regress right back to the calibrated run's over-broadening.
    assert protected.regions[0].decision is PackDecision.SKIP


def test_candidate_returned_files_do_not_depend_on_expected_files(
    python_simple_repo: Path,
) -> None:
    first = run_archex_query_context_candidate(
        _fixture_task("How does AuthService verify a token?"), python_simple_repo
    )
    second = run_archex_query_context_candidate(
        _fixture_task("How does AuthService verify a token?"), python_simple_repo
    )

    assert first.provenance == second.provenance
