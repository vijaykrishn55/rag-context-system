#!/usr/bin/env python
"""Query the local Advanced RAG pipeline and optionally generate a local answer."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from advanced_rag.schemas import PipelineConfig
from advanced_rag.loaders import load_documents
from advanced_rag.embeddings import LocalEmbeddingModel
from advanced_rag.reranker import CrossEncoderReranker
from advanced_rag.pipeline import AdvancedRAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to retrieve context for")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "sample_corpus"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--context-token-budget", type=int, default=1500)
    parser.add_argument("--generate", action="store_true", help="Also load Qwen locally to generate an answer")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--source", help="Optional exact source-file metadata filter")
    args = parser.parse_args()

    cfg = PipelineConfig(rerank_top_n=args.top_n, context_token_budget=args.context_token_budget)
    embedder = LocalEmbeddingModel(device=args.device)
    reranker = CrossEncoderReranker(device=args.device)
    pipeline = AdvancedRAGPipeline(cfg, embedder, reranker, persist_directory=ROOT / ".cache" / "chroma", device=args.device)
    pipeline.ingest(load_documents(args.input_dir))
    docs = pipeline.retrieve(args.query, top_n=args.top_n,
                             metadata_filter={"source": args.source} if args.source else None)
    packed = pipeline.build_context(docs, args.context_token_budget)
    print("\n=== RERANKED PARENTS ===")
    for rank, doc in enumerate(docs, 1):
        print(f"{rank}. source={doc.metadata.get('source')} score={doc.rerank_score:.4f} parent={doc.id}")
        print(doc.text[:500].strip().replace("\n", " ") + ("..." if len(doc.text) > 500 else ""))
    print(f"\n=== PACKED CONTEXT ===\nTokens: {packed.token_count}/{packed.budget}\nSources: {packed.sources}")
    print(packed.text)
    if args.generate:
        from advanced_rag.local_generator import LocalAnswerGenerator
        generator = LocalAnswerGenerator(args.generator_model, device=args.device)
        print("\n=== LOCAL MODEL ANSWER ===")
        print(generator.generate(args.query, packed.text))


if __name__ == "__main__":
    main()
