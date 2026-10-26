"""Reciprocal Rank Fusion (RRF) for combining independent ranked lists."""
from __future__ import annotations
from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Return unique IDs sorted by RRF score, preserving stable tie ordering."""
    if k < 1:
        raise ValueError("k must be >= 1")
    scores = defaultdict(float)
    first_seen = {}
    rank_accumulator = 0
    for ranking in rankings:
        seen_in_list = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = rank_accumulator
                rank_accumulator += 1
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
