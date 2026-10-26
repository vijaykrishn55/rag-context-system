from advanced_rag.pipeline import AdvancedRAGPipeline
from advanced_rag.schemas import ChildChunk, ParentChunk, PipelineConfig, RetrievedParent


class FakeIndex:
    def __init__(self, rows):
        self.rows = rows
        self.last_filter = None

    def search(self, query, top_k=50, metadata_filter=None):
        self.last_filter = metadata_filter
        rows = self.rows
        if metadata_filter:
            rows = [r for r in rows if all(r.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())]
        return rows[:top_k]


class FakeReranker:
    model_name = "fake-cross-encoder"

    def rerank(self, query, documents, top_n=5):
        ranked = [doc.model_copy(update={"rerank_score": float(doc.metadata.get("priority", 0))}) for doc in documents]
        ranked.sort(key=lambda doc: doc.rerank_score, reverse=True)
        return ranked[:top_n]


def test_hybrid_fusion_maps_children_to_parents_and_reranks_top_n(tmp_path):
    config = PipelineConfig(dense_top_k=3, sparse_top_k=3, rerank_top_n=2, semantic_chunking=False)
    pipe = AdvancedRAGPipeline(config=config, embedding_model=object(), reranker=FakeReranker(),
                               persist_directory=tmp_path, semantic_chunking=False)
    parents = [
        ParentChunk(id="p1", text="parent one details", metadata={"source": "a.md", "priority": 1}),
        ParentChunk(id="p2", text="parent two details", metadata={"source": "b.md", "priority": 3}),
        ParentChunk(id="p3", text="parent three details", metadata={"source": "c.md", "priority": 2}),
    ]
    children = [
        ChildChunk(id="c1", parent_id="p1", text="child one", metadata={"source": "a.md", "parent_id": "p1"}),
        ChildChunk(id="c2", parent_id="p2", text="child two", metadata={"source": "b.md", "parent_id": "p2"}),
        ChildChunk(id="c3", parent_id="p3", text="child three", metadata={"source": "c.md", "parent_id": "p3"}),
    ]
    pipe.parents = {p.id: p for p in parents}
    pipe.children = {c.id: c for c in children}
    pipe.dense_index = FakeIndex([{"id": "c1", "metadata": {"source": "a.md"}},
                                  {"id": "c2", "metadata": {"source": "b.md"}},
                                  {"id": "c3", "metadata": {"source": "c.md"}}])
    pipe.sparse_index = FakeIndex([{"id": "c2", "metadata": {"source": "b.md"}},
                                   {"id": "c1", "metadata": {"source": "a.md"}},
                                   {"id": "c3", "metadata": {"source": "c.md"}}])

    result = pipe.retrieve("query", top_n=2, metadata_filter={"source": "b.md"})

    assert len(result) == 1
    assert result[0].id == "p2"
    assert result[0].matched_child_ids == ["c2"]
    assert pipe.dense_index.last_filter == {"source": "b.md"}
    assert pipe.sparse_index.last_filter == {"source": "b.md"}
    assert pipe.last_retrieval_stats["returned_parents"] == 1


def test_retrieval_requires_ingestion_or_indexes():
    pipe = AdvancedRAGPipeline(embedding_model=object(), reranker=FakeReranker())
    try:
        pipe.retrieve("query")
    except RuntimeError as exc:
        assert "ingest" in str(exc)
    else:
        raise AssertionError("retrieve should require ingestion")
