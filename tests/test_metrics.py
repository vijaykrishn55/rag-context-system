import pytest
from advanced_rag.metrics import hit_rate_at_k, reciprocal_rank, ndcg_at_k, aggregate_metrics


def test_hit_rate_at_k():
    assert hit_rate_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}, 5) == pytest.approx(0.5)
    assert reciprocal_rank(["x"], {"y"}, 5) == 0.0


def test_ndcg_perfect_and_miss():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"b"}, 2) == 0.0


def test_aggregate_metrics():
    result = aggregate_metrics([
        {"hit_rate_at_k": 1.0, "mrr_at_k": 1.0, "ndcg_at_k": 1.0},
        {"hit_rate_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0},
    ])
    assert result["query_count"] == 2
    assert result["hit_rate_at_k"] == 0.5
