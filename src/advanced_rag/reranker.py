"""Local cross-encoder reranking with an injectable scoring function for tests."""
from __future__ import annotations
from typing import Callable
from .schemas import RetrievedParent


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str | None = None, scorer: Callable | None = None) -> None:
        self.model_name = model_name
        self._scorer = scorer
        self.model = None
        if scorer is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required; install requirements.txt") from exc
            kwargs = {}
            if device and device != "auto":
                kwargs["device"] = device
            self.model = CrossEncoder(model_name, **kwargs)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self._scorer is not None:
            return [float(x) for x in self._scorer(query, texts)]
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]

    def rerank(self, query: str, documents: list[RetrievedParent], top_n: int = 5) -> list[RetrievedParent]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not documents:
            return []
        scores = self.score(query, [doc.text for doc in documents])
        ranked = []
        for doc, score in zip(documents, scores):
            ranked.append(doc.model_copy(update={"rerank_score": score}))
        ranked.sort(key=lambda doc: float(doc.rerank_score or 0.0), reverse=True)
        return ranked[:top_n]
