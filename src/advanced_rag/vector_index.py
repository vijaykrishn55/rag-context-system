"""ChromaDB dense index with explicit local embeddings."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .schemas import ChildChunk


class ChromaDenseIndex:
    def __init__(self, embedding_model, persist_directory: str | Path,
                 collection_name: str = "advanced_rag_children", reset: bool = True) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required; install requirements.txt") from exc
        self.embedding_model = embedding_model
        self.persist_directory = str(persist_directory)
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name = collection_name
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "Child chunk embeddings"},
        )

    def index(self, children: list[ChildChunk], batch_size: int = 128) -> None:
        if not children:
            raise ValueError("Cannot index an empty child chunk list")
        for start in range(0, len(children), batch_size):
            batch = children[start:start + batch_size]
            vectors = self.embedding_model.encode_documents([c.text for c in batch])
            metadatas = []
            for child in batch:
                # Chroma metadata values must be primitive scalars, not nested dicts/lists.
                meta = {}
                for key, value in child.metadata.items():
                    if value is None:
                        continue
                    if isinstance(value, (str, int, float, bool)):
                        meta[key] = value
                    else:
                        meta[key] = str(value)
                meta["parent_id"] = child.parent_id
                metadatas.append(meta)
            self.collection.add(
                ids=[c.id for c in batch], embeddings=vectors.tolist(),
                documents=[c.text for c in batch], metadatas=metadatas,
            )

    def search(self, query: str, top_k: int = 50, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        count = int(self.collection.count())
        if count == 0:
            return []
        vector = self.embedding_model.encode_query(query)
        query_kwargs = {
            "query_embeddings": [vector.tolist()],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter
        result = self.collection.query(**query_kwargs)
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for idx, child_id in enumerate(ids):
            hits.append({"id": child_id, "text": docs[idx], "metadata": metas[idx] or {},
                         "distance": float(distances[idx]) if idx < len(distances) else None,
                         "rank": idx + 1})
        return hits

    def count(self) -> int:
        return int(self.collection.count())
