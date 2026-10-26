import pytest
from advanced_rag.fusion import reciprocal_rank_fusion


def test_rrf_combines_rankings_and_deduplicates_ids():
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "D", "A"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert len(ids) == len(set(ids)) == 4
    assert ids[0] == "A" or ids[0] == "B"
    scores = dict(fused)
    assert scores["A"] == pytest.approx(1/61 + 1/63)
    assert scores["B"] == pytest.approx(1/62 + 1/61)


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_rrf_empty_lists_are_supported():
    assert reciprocal_rank_fusion([[], []]) == []
