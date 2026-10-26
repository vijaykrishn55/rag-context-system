"""Standard information-retrieval metrics for source relevance judgments."""
from __future__ import annotations
import math
from typing import Iterable, Sequence


def hit_rate_at_k(ranked_sources: Sequence[str], relevant_sources: set[str], k: int = 5) -> float:
    return float(any(source in relevant_sources for source in ranked_sources[:k]))


def reciprocal_rank(ranked_sources: Sequence[str], relevant_sources: set[str], k: int = 10) -> float:
    for rank, source in enumerate(ranked_sources[:k], start=1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_sources: Sequence[str], relevant_sources: set[str], k: int = 5) -> float:
    def dcg(values: Sequence[str]) -> float:
        return sum((1.0 / math.log2(rank + 1)) for rank, source in enumerate(values[:k], start=1)
                   if source in relevant_sources)
    ideal_count = min(k, len(relevant_sources))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return 0.0 if ideal == 0 else dcg(ranked_sources) / ideal


def aggregate_metrics(rows: list[dict], k: int = 5) -> dict:
    if not rows:
        return {"query_count": 0, "hit_rate_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}
    return {
        "query_count": len(rows),
        "hit_rate_at_k": round(sum(r["hit_rate_at_k"] for r in rows) / len(rows), 4),
        "mrr_at_k": round(sum(r["mrr_at_k"] for r in rows) / len(rows), 4),
        "ndcg_at_k": round(sum(r["ndcg_at_k"] for r in rows) / len(rows), 4),
        "k": k,
    }
