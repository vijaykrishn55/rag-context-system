"""BM25 lexical index over child chunks."""
from __future__ import annotations
import re
from typing import Iterable
from .schemas import ChildChunk


def tokenize(text: str) -> list[str]:
    # Keep identifiers such as PART-NX-4407 intact as tokens as well as components.
    raw = re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower())
    tokens = []
    for token in raw:
        tokens.append(token)
        tokens.extend(part for part in re.split(r"[-_]", token) if part and part != token)
    return tokens


class BM25SparseIndex:
    def __init__(self, children: list[ChildChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required; install requirements.txt") from exc
        if not children:
            raise ValueError("Cannot create BM25 index from an empty list")
        self.children = children
        self.ids = [child.id for child in children]
        self._tokens = [tokenize(child.text) for child in children]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 50, metadata_filter: dict | None = None) -> list[dict]:
        if not self.children:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        eligible = [i for i, child in enumerate(self.children)
                    if not metadata_filter or all(child.metadata.get(key) == value
                                                  for key, value in metadata_filter.items())]
        ranked = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)
        hits = []
        for index in ranked[:min(top_k, len(ranked))]:
            child = self.children[index]
            hits.append({"id": child.id, "text": child.text,
                         "metadata": child.metadata, "score": float(scores[index]),
                         "rank": len(hits) + 1})
        return hits
