# Platform Parts Catalog

## Component PART-NX-4407
Part number `PART-NX-4407` identifies the high-throughput read cache adapter used by the platform's retrieval gateway. Its lexical catalog key is retained exactly because rare part identifiers are often missed by semantic-only search. The adapter uses a segmented least-recently-used (LRU) cache and a bloom filter to reduce unnecessary storage lookups. When a cache entry is missing, the gateway looks up the versioned record by partition key, validates the record version, and then updates the cache.

The adapter is designed to accelerate exact record retrieval rather than compute embeddings. For this reason, hybrid search must include a sparse keyword path so the exact identifier `PART-NX-4407` can match even when a vector embedding favors documents that merely describe high-throughput caching in general.

## Operational Limits
The cache must not be used as an authorization source. Expired entries are evicted, and stale entries are invalidated when the authoritative record version changes. A cache hit must still pass the caller's authorization checks.
