#!/usr/bin/env python
"""Optional demonstration of LangChain ParentDocumentRetriever with local embeddings."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from advanced_rag.loaders import load_documents
from advanced_rag.langchain_adapter import build_parent_document_retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to retrieve parent context for")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "sample_corpus"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to use the LangChain adapter") from exc

    sources = load_documents(args.input_dir)
    langchain_docs = [Document(page_content=d.text, metadata={**d.metadata, "source": d.source}) for d in sources]
    retriever = build_parent_document_retriever(langchain_docs, persist_directory=str(ROOT / ".cache" / "langchain_chroma"), device=args.device)
    retriever.add_documents(langchain_docs)
    parents = retriever.invoke(args.query)
    print(f"Retrieved {len(parents)} parent documents")
    for index, doc in enumerate(parents, 1):
        print(f"\n{index}. source={doc.metadata.get('source')} parent_id={doc.metadata.get('parent_id')}")
        print(doc.page_content[:1200])


if __name__ == "__main__":
    main()
