"""Small, auditable retrieval benchmark for the included demo corpus."""
from .schemas import EvaluationQuery

EVALUATION_QUERIES = [
    EvaluationQuery(
        query_id="latency-sla",
        query="What is the guaranteed query latency SLA for Project Titan?",
        relevant_sources=["titan_vector_architecture.md"],
        tags=["semantic", "retrieval"],
    ),
    EvaluationQuery(
        query_id="reranking-method",
        query="How does the two-stage reranking pipeline improve relevance and reduce hallucinations?",
        relevant_sources=["titan_reranking_spec.md"],
        tags=["semantic", "reranking"],
    ),
    EvaluationQuery(
        query_id="security-protocols",
        query="Describe the encryption and HSM authentication protocols used by Project Titan.",
        relevant_sources=["titan_security_whitepaper.md"],
        tags=["security", "semantic"],
    ),
    EvaluationQuery(
        query_id="rare-part-number",
        query="For part PART-NX-4407, what cache and indexing approach gives high read throughput?",
        relevant_sources=["platform_parts_catalog.md"],
        tags=["hybrid", "exact-keyword"],
    ),
    EvaluationQuery(
        query_id="retention-policy",
        query="When can archived audit logs be purged, and what approval is required?",
        relevant_sources=["enterprise_retention_policy.md"],
        tags=["structural", "metadata"],
    ),
    EvaluationQuery(
        query_id="client-retry-code",
        query="Which client function retries failed retrieval requests with exponential backoff?",
        relevant_sources=["titan_client_implementation.md"],
        tags=["code", "structural"],
    ),
]
