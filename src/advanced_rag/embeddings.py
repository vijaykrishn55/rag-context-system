"""Local SentenceTransformer embeddings; no hosted embedding API."""
from __future__ import annotations
from typing import Sequence


class LocalEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required; install requirements.txt") from exc
        kwargs = {}
        if device and device != "auto":
            kwargs["device"] = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Sequence[str], **kwargs):
        kwargs.setdefault("convert_to_numpy", True)
        kwargs.setdefault("normalize_embeddings", True)
        kwargs.setdefault("show_progress_bar", False)
        return self.model.encode(list(texts), **kwargs)

    def encode_documents(self, texts: Sequence[str]):
        return self.encode(texts)

    def encode_query(self, query: str):
        return self.encode([query])[0]
