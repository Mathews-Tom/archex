from archex.api import _expand_retrieval_question


def test_vocabulary_disjoint_ablations() -> None:
    """Ensure benchmark terms and domain queries are disjoint in expansion.

    Production heuristic generalization (like 'query pipeline' -> 'assemble_context')
    must work, but benchmark vocabulary must not trigger any specialized heuristic
    pathways, proving the heuristic is disjoint from benchmark identity.
    """
    # 1. Domain vocabulary expands correctly (heuristic generalization)
    expanded, prov = _expand_retrieval_question("how does the query pipeline work?")
    assert "assemble_context" in expanded
    assert "query pipeline" in prov

    # 2. Benchmark vocabulary is completely disjoint and triggers no expansions
    banned_terms = [
        "swe_bench",
        "swebench",
        "human_eval",
        "humaneval",
        "mbpp",
        "bird",
        "spider",
        "defects4j",
    ]
    for term in banned_terms:
        banned_expanded, banned_prov = _expand_retrieval_question(f"run {term}")
        assert banned_expanded == f"run {term}"
        assert not banned_prov
