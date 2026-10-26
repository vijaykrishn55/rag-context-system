# Advanced RAG Context

A local-first Retrieval-Augmented Generation pipeline featuring structure-aware chunking, parent-child retrieval, hybrid search, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and token-budgeted context packing. The retrieval and evaluation workflow uses local models and does not require an OpenAI API key.

## Features

- Markdown-aware structural and optional semantic chunking
- Parent-child retrieval for focused search and richer context
- Dense vector search with ChromaDB and local SentenceTransformers embeddings
- BM25 lexical search with RRF rank fusion
- Local cross-encoder reranking
- Token-budgeted context packing with source attribution
- Retrieval evaluation using Hit Rate@K, MRR@K, and NDCG@K
- Optional local Qwen answer generation

## Architecture

| Module | Responsibility |
|---|---|
| `chunking.py` | Structural and semantic chunking, parent-child IDs |
| `embeddings.py`, `vector_index.py` | Local embeddings and ChromaDB indexing |
| `sparse_index.py`, `fusion.py` | BM25 retrieval and RRF fusion |
| `pipeline.py`, `reranker.py` | Retrieval orchestration and reranking |
| `context_packer.py` | Context packing within a token budget |
| `evaluation.py`, `metrics.py` | Retrieval metrics and evaluation evidence |
| `local_generator.py` | Optional local answer generation |
| `langchain_adapter.py` | Optional LangChain parent-document retriever adapter |

The main pipeline uses a custom parent-child retriever to fuse dense and BM25 rankings. The LangChain adapter is optional.

## Setup

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Tests

```bash
python -m pytest -q
```

## Structural validation

```bash
python scripts/run_evaluation.py --dry-run --output reports/dry-run.json
```

## Retrieval evaluation

```bash
python scripts/run_evaluation.py \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --top-k 50 --top-n 5 \
  --output reports/evaluation.json
```

The first run downloads the local embedding and reranker models. Evaluation reports include retrieval metrics, per-query rankings, latency, and probe results.

## Query the system

```bash
python scripts/demo_query.py "What encryption and authentication standards protect Project Titan data?"
```

To enable optional local answer generation:

```bash
python scripts/demo_query.py "What encryption and authentication standards protect Project Titan data?" --generate
```

The generation model requires an additional download and more memory/VRAM.

## Configuration notes

- Parent hard cap: 512 tokens
- Child target: 120 tokens
- Child overlap: 20 tokens
- Context token budget: 1,500 tokens by default

Cross-encoder scores indicate relevance, not truth. Evaluation results depend on the corpus, relevance labels, model versions, and hardware. Expand the evaluation dataset before making production-quality claims.
