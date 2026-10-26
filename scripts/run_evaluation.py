#!/usr/bin/env python
"""Run structural dry checks or the full local hybrid-RAG evaluation."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advanced_rag.chunking import ParentChildChunker, parse_structural_blocks
from advanced_rag.evaluation import evaluate_retrieval
from advanced_rag.evaluation_queries import EVALUATION_QUERIES
from advanced_rag.pipeline import AdvancedRAGPipeline
from advanced_rag.schemas import PipelineConfig, RetrievedParent
from advanced_rag.loaders import load_documents
from advanced_rag.tokenization import TokenCounter


def structural_dry_run(input_dir: Path, output_path: Path, config: PipelineConfig) -> dict:
    documents = load_documents(input_dir)
    counter = TokenCounter()
    chunker = ParentChildChunker(config=config, token_counter=counter, semantic_embedder=None)
    parents, children = chunker.split_corpus(documents)
    parent_map = {p.id: p for p in parents}
    child_mapping_ok = all(c.parent_id in parent_map for c in children)
    parent_child_size_ok = all(counter.count(parent_map[c.parent_id].text) >= counter.count(c.text)
                               for c in children)
    all_parent_within_hard_cap = all(counter.count(p.text) <= config.parent_max_tokens for p in parents)
    all_child_within_hard_cap = all(counter.count(c.text) <= config.parent_max_tokens for c in children)
    fixture = next((d for d in documents if d.source == "structural_code_sample.md"), None)
    code_blocks = [b.text for b in parse_structural_blocks(fixture.text) if b.kind == "code"] if fixture else []
    code_blocks_preserved = all(any(code in p.text for p in parents if p.metadata.get("source") == fixture.source)
                                for code in code_blocks) if fixture else False
    checks = {
        "source_documents_loaded": len(documents),
        "parent_chunks_created": len(parents),
        "child_chunks_created": len(children),
        "child_parent_mapping_valid": child_mapping_ok,
        "parent_token_count_ge_child": parent_child_size_ok,
        "all_parent_chunks_within_hard_cap": all_parent_within_hard_cap,
        "all_child_chunks_within_hard_cap": all_child_within_hard_cap,
        "fenced_code_blocks_preserved_intact": code_blocks_preserved,
        "tokenizer_mode": counter.mode,
    }
    checks["failed_checks"] = [key for key, value in checks.items()
                               if key not in {"source_documents_loaded", "parent_chunks_created", "child_chunks_created", "tokenizer_mode"}
                               and value is not True]
    checks["passed"] = len(checks["failed_checks"]) == 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"Structural dry run: {'PASS' if checks['passed'] else 'FAIL'}")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print(f"JSON report: {output_path}")
    return checks


def run_hybrid_probe(pipeline: AdvancedRAGPipeline) -> dict:
    query = "For part PART-NX-4407, what cache and indexing approach gives high read throughput?"
    dense = pipeline.dense_index.search(query, 50)
    sparse = pipeline.sparse_index.search(query, 50)
    from advanced_rag.fusion import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([[h["id"] for h in dense], [h["id"] for h in sparse]], k=pipeline.config.rrf_k)
    def source_rank(hits):
        for rank, item in enumerate(hits, 1):
            if item.get("metadata", {}).get("source") == "platform_parts_catalog.md":
                return rank
        return None
    fused_source_rank = None
    for rank, (child_id, score) in enumerate(fused, 1):
        child = pipeline.children.get(child_id)
        if child and child.metadata.get("source") == "platform_parts_catalog.md":
            fused_source_rank = rank
            break
    return {
        "query": query,
        "expected_source": "platform_parts_catalog.md",
        "dense_rank": source_rank(dense),
        "bm25_rank": source_rank(sparse),
        "rrf_child_rank": fused_source_rank,
        "bm25_found_exact_part_source": source_rank(sparse) is not None,
        "note": "Dense embeddings may also retrieve exact identifiers; this probe verifies BM25 and RRF preserve the lexical match.",
    }


def run_reranker_probe(pipeline: AdvancedRAGPipeline) -> dict:
    query = "What encryption and authentication standards protect Project Titan data?"
    positive = RetrievedParent(
        id="probe-security", text=("Project Titan encrypts data in transit and at rest with AES-256. "
                                    "Authentication uses multi-factor cryptographic tokens managed by Hardware Security Modules (HSMs)."),
        metadata={"source": "expected_security_context"},
    )
    distractors = []
    themes = [
        "Vector storage uses HNSW indexing to speed up nearest neighbor lookup.",
        "The client retries temporary timeouts with exponential backoff.",
        "Audit logs are retained for 365 days and require approval before purge.",
        "BM25 scores term frequency and inverse document frequency.",
        "The cache adapter uses a segmented LRU policy for fast reads.",
    ]
    for i in range(19):
        distractors.append(RetrievedParent(
            id=f"probe-distractor-{i:02d}",
            text=themes[i % len(themes)] + f" Additional unrelated system detail {i}.",
            metadata={"source": f"distractor-{i:02d}"},
        ))
    ranked = pipeline.reranker.rerank(query, [positive] + distractors, top_n=20)
    rank = next((i for i, doc in enumerate(ranked, 1) if doc.id == positive.id), None)
    return {
        "query": query, "candidate_count": 20, "expected_top_parent": positive.id,
        "relevant_candidate_rank": rank,
        "passed": rank == 1,
        "top_five": [{"id": d.id, "score": d.rerank_score} for d in ranked[:5]],
        "note": "Cross-encoder scores are relevance signals, not probabilities of truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Run structural checks without downloading ML models")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "sample_corpus"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "evaluation.json"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--context-token-budget", type=int, default=1500)
    parser.add_argument("--disable-semantic-chunking", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    config = PipelineConfig(
        dense_top_k=args.top_k, sparse_top_k=args.top_k,
        rerank_top_n=args.top_n, context_token_budget=args.context_token_budget,
        semantic_chunking=not args.disable_semantic_chunking,
    )
    if args.dry_run:
        result = structural_dry_run(input_dir, output_path, config)
        if not result["passed"]:
            raise SystemExit(1)
        return

    from advanced_rag.embeddings import LocalEmbeddingModel
    from advanced_rag.reranker import CrossEncoderReranker
    print(f"Loading local embeddings: {args.embedding_model}")
    embedding_model = LocalEmbeddingModel(args.embedding_model, device=args.device)
    print(f"Loading local cross-encoder: {args.reranker_model}")
    reranker = CrossEncoderReranker(args.reranker_model, device=args.device)
    pipeline = AdvancedRAGPipeline(
        config=config, embedding_model=embedding_model, reranker=reranker,
        persist_directory=ROOT / ".cache" / "chroma",
        device=args.device,
        semantic_chunking=not args.disable_semantic_chunking,
    )
    documents = load_documents(input_dir)
    print(f"Ingesting {len(documents)} source documents...")
    ingestion = pipeline.ingest(documents)
    print(f"Indexed {ingestion['parent_chunks']} parents and {ingestion['child_chunks']} children.")

    report = evaluate_retrieval(pipeline, EVALUATION_QUERIES, k=args.top_n,
                                token_budget=args.context_token_budget)
    report["project"] = "Advanced RAG Context"
    report["config"] = config.model_dump(mode="json")
    report["corpus_directory"] = str(input_dir)
    report["models"] = {
        "embedding": args.embedding_model,
        "reranker": args.reranker_model,
        "answer_generation": "optional; not loaded by retrieval evaluation",
    }
    report["ingestion"] = ingestion
    if not args.skip_probes:
        print("Running hybrid retrieval probe...")
        report["hybrid_synergy_probe"] = run_hybrid_probe(pipeline)
        print("Running 20-candidate reranking probe...")
        report["reranking_probe"] = run_reranker_probe(pipeline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RETRIEVAL METRICS ===")
    print(json.dumps(report["metrics"], indent=2))
    print(f"JSON report: {output_path}")


if __name__ == "__main__":
    main()
