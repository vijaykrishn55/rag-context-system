import pytest
from advanced_rag.reranker import CrossEncoderReranker
from advanced_rag.schemas import RetrievedParent


def doc(i, text):
    return RetrievedParent(id=str(i), text=text, metadata={"source": f"s{i}"})


def test_reranker_sorts_descending_and_truncates_top_n():
    ranks = {"low relevance": 0.1, "medium relevance": 0.5, "high relevance": 0.9}
    reranker = CrossEncoderReranker(scorer=lambda query, texts: [ranks[t] for t in texts])
    result = reranker.rerank("query", [doc(1, "low relevance"), doc(2, "high relevance"), doc(3, "medium relevance")], top_n=2)
    assert [d.text for d in result] == ["high relevance", "medium relevance"]
    assert result[0].rerank_score >= result[1].rerank_score


def test_reranker_handles_empty_candidates():
    reranker = CrossEncoderReranker(scorer=lambda query, texts: [])
    assert reranker.rerank("query", []) == []


def test_reranker_rejects_invalid_top_n():
    reranker = CrossEncoderReranker(scorer=lambda query, texts: [])
    with pytest.raises(ValueError):
        reranker.rerank("query", [], 0)
