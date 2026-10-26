# Project Titan Retrieval and Reranking Specification

## Two-Stage Retrieval
Retrieval within Project Titan uses a two-stage pipeline. A fast bi-encoder retrieves an initial candidate set using dense vector similarity and lexical BM25 search. Reciprocal Rank Fusion (RRF) combines the independent rankings using a score of `1 / (k + rank)` so results can be merged without assuming that the original score scales are comparable.

The candidate set is then passed through a neural cross-encoder. The model jointly reads each query and candidate text, allowing token-level interactions to influence the relevance score. Candidates are sorted by their cross-encoder score, and only the requested top-N parent contexts are forwarded to the generation stage.

## Reliability Notes
Reranking improves ordering but does not prove a document is factually correct. The application must retain source metadata, pack only relevant parents within a token budget, and require the answer generator to cite or stay grounded in retrieved context. A retrieval score is a relevance signal, not a calibrated probability of truth.
