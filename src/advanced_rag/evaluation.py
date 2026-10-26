"""Retrieval-quality evaluation with per-query source judgments and timing."""
from __future__ import annotations
import statistics
import time
from .metrics import hit_rate_at_k, reciprocal_rank, ndcg_at_k, aggregate_metrics
from .schemas import EvaluationQuery
from .pipeline import AdvancedRAGPipeline


def evaluate_retrieval(pipeline: AdvancedRAGPipeline, queries: list[EvaluationQuery],
                       k: int = 5, token_budget: int = 1500) -> dict:
    rows = []
    for q in queries:
        start = time.perf_counter()
        docs = pipeline.retrieve(q.query, top_n=max(k, pipeline.config.rerank_top_n))
        latency = (time.perf_counter() - start) * 1000
        # Judgments are source-level, so count each source only once in ranking metrics.
        unique_source_docs = []
        seen_sources = set()
        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            if source not in seen_sources:
                seen_sources.add(source)
                unique_source_docs.append(doc)
        sources = [str(doc.metadata.get("source", "unknown")) for doc in unique_source_docs]
        relevant = set(q.relevant_sources)
        packed = pipeline.build_context(docs[:k], token_budget)
        row = {
            "query_id": q.query_id,
            "query": q.query,
            "tags": q.tags,
            "relevant_sources": sorted(relevant),
            "ranked_sources": sources[:k],
            "ranked_parent_ids": [doc.id for doc in unique_source_docs[:k]],
            "hit_rate_at_k": hit_rate_at_k(sources, relevant, k),
            "mrr_at_k": reciprocal_rank(sources, relevant, k),
            "ndcg_at_k": ndcg_at_k(sources, relevant, k),
            "retrieval_latency_ms": round(latency, 3),
            "context_token_count": packed.token_count,
            "context_token_budget": packed.budget,
            "context_sources": packed.sources,
            "omitted_parent_ids": packed.omitted_parent_ids,
            "retrieval_stats": dict(pipeline.last_retrieval_stats),
            "reranked": [
                {"parent_id": d.id, "source": d.metadata.get("source"),
                 "score": d.rerank_score, "rrf_score": d.rrf_score,
                 "matched_child_ids": d.matched_child_ids}
                for d in docs[:k]
            ],
        }
        rows.append(row)
    metrics = aggregate_metrics(rows, k=k)
    latencies = sorted(r["retrieval_latency_ms"] for r in rows)
    metrics["latency_ms"] = {
        "mean": round(statistics.mean(latencies), 3) if latencies else 0,
        "p50": round(latencies[(len(latencies)-1)//2], 3) if latencies else 0,
        "p95": round(latencies[min(len(latencies)-1, int(0.95 * len(latencies)))], 3) if latencies else 0,
        "target_ms": 800,
        "target_met_in_this_run": bool(latencies and max(latencies) <= 800),
        "note": "Includes retrieval and reranking, excludes initial model loading and document ingestion.",
    }
    return {"metrics": metrics, "queries": rows}
