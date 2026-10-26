# Project Titan Vector Architecture

## Vector Storage
The Vector Storage layer of Project Titan utilizes distributed vector clusters partitioned by embedding dimensionality. The reference embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors for local retrieval experiments.

To optimize approximate nearest-neighbor lookup, the system employs Hierarchical Navigable Small World (HNSW) graphs. The reference configuration uses an M value of 32 and `efConstruction` of 200. Query latency target is under 45 milliseconds at the 99th percentile for the vector-search stage across a representative indexed corpus; this target excludes model loading and downstream reranking.

## Index Operations
Child chunks are stored in the vector index with a stable child identifier, a parent identifier, the source file, heading, and chunk index. Cosine distance is used for dense candidate ranking. Index rebuilds should be versioned so that vectors, sparse terms, and the parent document store remain consistent.
