"""Optional LangChain ParentDocumentRetriever adapter using local embeddings only.

The main AdvancedRAGPipeline implements hybrid child search + RRF directly so BM25
and Chroma rankings can be fused. This adapter is provided for integrations that
specifically want LangChain's ParentDocumentRetriever abstraction.
"""
from __future__ import annotations
from langchain_core.embeddings import Embeddings


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Minimal LangChain Embeddings implementation backed by a local ST model."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install langchain-core and sentence-transformers first") from exc
        self._model = SentenceTransformer(model_name, **({"device": device} if device and device != "auto" else {}))
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


def build_parent_document_retriever(documents, persist_directory: str = ".cache/langchain_chroma",
                                    device: str | None = None):
    """Build LangChain's retriever with local embeddings and in-memory parent store.

    `documents` should be a list of LangChain Document objects. The returned retriever
    indexes child chunks in ChromaDB and stores parent chunks in an InMemoryStore.
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain.storage import InMemoryStore
        from langchain.retrievers import ParentDocumentRetriever
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError("Install langchain, langchain-community, and langchain-text-splitters") from exc
    embeddings = LocalSentenceTransformerEmbeddings(device=device)
    vectorstore = Chroma(
        collection_name="advanced_rag_parent_child",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    store = InMemoryStore()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800, chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=480, chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=store,
        parent_splitter=parent_splitter, child_splitter=child_splitter,
        id_key="parent_id",
    )
