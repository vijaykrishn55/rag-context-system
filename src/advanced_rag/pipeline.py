"""End-to-end local Advanced RAG pipeline: chunk, index, retrieve, fuse, rerank, pack."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
from .chunking import ParentChildChunker
from .context_packer import ContextPacker, PackedContext
from .embeddings import LocalEmbeddingModel
from .fusion import reciprocal_rank_fusion
from .reranker import CrossEncoderReranker
from .schemas import ChildChunk, ParentChunk, PipelineConfig, RetrievedParent, SourceDocument
from .sparse_index import BM25SparseIndex
from .tokenization import TokenCounter
from .vector_index import ChromaDenseIndex


class AdvancedRAGPipeline:
    def __init__(self, config: PipelineConfig | None = None,
                 embedding_model: Any | None = None, reranker: Any | None = None,
                 persist_directory: str | Path = ".cache/chroma",
                 collection_name: str = "advanced_rag_children",
                 device: str | None = None,
                 semantic_chunking: bool | None = None) -> None:
        self.config = config or PipelineConfig()
        self.counter = TokenCounter()
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.device = device
        self.semantic_chunking = self.config.semantic_chunking if semantic_chunking is None else semantic_chunking
        self.parents: dict[str, ParentChunk] = {}
        self.children: dict[str, ChildChunk] = {}
        self.dense_index = None
        self.sparse_index = None
        self.chunker = None
        self.last_retrieval_stats: dict[str, Any] = {}

    def _ensure_models(self) -> None:
        if self.embedding_model is None:
            self.embedding_model = LocalEmbeddingModel(device=self.device)
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(device=self.device)

    def ingest(self, documents: list[SourceDocument]) -> dict[str, int | str]:
        if not documents:
            raise ValueError("Ingestion requires at least one source document")
        self._ensure_models()
        cfg = self.config.model_copy(update={"semantic_chunking": self.semantic_chunking})
        self.chunker = ParentChildChunker(
            config=cfg, token_counter=self.counter,
            semantic_embedder=self.embedding_model if self.semantic_chunking else None,
        )
        parents, children = self.chunker.split_corpus(documents)
        if not parents or not children:
            raise ValueError("Chunking produced no parents or children")
        self.parents = {p.id: p for p in parents}
        self.children = {c.id: c for c in children}
        self.dense_index = ChromaDenseIndex(
            self.embedding_model, self.persist_directory, self.collection_name, reset=True
        )
        self.dense_index.index(children)
        self.sparse_index = BM25SparseIndex(children)
        return {
            "source_documents": len(documents),
            "parent_chunks": len(parents),
            "child_chunks": len(children),
            "tokenizer": self.counter.mode,
            "embedding_model": getattr(self.embedding_model, "model_name", type(self.embedding_model).__name__),
            "reranker_model": getattr(self.reranker, "model_name", type(self.reranker).__name__),
        }

    def retrieve(self, query: str, dense_k: int | None = None, sparse_k: int | None = None,
                 top_n: int | None = None, metadata_filter: dict[str, Any] | None = None) -> list[RetrievedParent]:
        if self.dense_index is None or self.sparse_index is None:
            raise RuntimeError("Call ingest(documents) before retrieve(query)")
        dense_k = dense_k or self.config.dense_top_k
        sparse_k = sparse_k or self.config.sparse_top_k
        top_n = top_n or self.config.rerank_top_n
        start = time.perf_counter()
        dense_hits = self.dense_index.search(query, dense_k, metadata_filter=metadata_filter)
        sparse_hits = self.sparse_index.search(query, sparse_k, metadata_filter=metadata_filter)
        fused_all = reciprocal_rank_fusion(
            [[hit["id"] for hit in dense_hits], [hit["id"] for hit in sparse_hits]],
            k=self.config.rrf_k,
        )
        # Bound neural reranking cost: at most top-k fused child candidates are expanded to parents.
        fused = fused_all[:max(dense_k, sparse_k)]

        # Map precise child matches back to their larger parent context, deduplicating parents.
        parent_hits: dict[str, dict[str, Any]] = {}
        child_rank = {child_id: (score, rank) for rank, (child_id, score) in enumerate(fused, start=1)}
        for child_id, rrf_score in fused:
            child = self.children.get(child_id)
            if child is None:
                continue
            record = parent_hits.setdefault(child.parent_id, {
                "rrf_score": 0.0, "matched_child_ids": [],
                "best_child_rank": child_rank[child_id][1],
            })
            record["rrf_score"] += float(rrf_score)
            record["matched_child_ids"].append(child_id)
            record["best_child_rank"] = min(record["best_child_rank"], child_rank[child_id][1])

        candidates = []
        for parent_id, info in parent_hits.items():
            parent = self.parents.get(parent_id)
            if parent is None:
                continue
            candidates.append(RetrievedParent(
                id=parent.id, text=parent.text, metadata=parent.metadata,
                rrf_score=info["rrf_score"], matched_child_ids=info["matched_child_ids"],
            ))
        candidates.sort(key=lambda d: (-d.rrf_score, d.id))
        reranked = self.reranker.rerank(query, candidates, top_n=top_n)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_retrieval_stats = {
            "latency_ms": round(elapsed_ms, 3), "dense_candidates": len(dense_hits),
            "sparse_candidates": len(sparse_hits), "fused_child_candidates": len(fused),
            "fused_unique_children_before_cap": len(fused_all),
            "metadata_filter": metadata_filter,
            "unique_parent_candidates": len(candidates), "returned_parents": len(reranked),
        }
        return reranked

    def build_context(self, documents: list[RetrievedParent], token_budget: int | None = None) -> PackedContext:
        packer = ContextPacker(self.counter)
        return packer.pack(documents, token_budget or self.config.context_token_budget)

    def export_parent_store(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: item.model_dump(mode="json") for key, item in self.parents.items()}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
