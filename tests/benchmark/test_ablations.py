from archex.api import _expand_retrieval_question  # pyright: ignore[reportPrivateUsage]


def test_vocabulary_disjoint_ablations() -> None:
    """Benchmark vocabulary must not trigger any query-expansion pathway."""
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
