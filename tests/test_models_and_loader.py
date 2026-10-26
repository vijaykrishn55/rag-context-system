from pathlib import Path
import pytest
from pydantic import ValidationError
from advanced_rag.schemas import PipelineConfig, SourceDocument
from advanced_rag.loaders import load_documents

ROOT = Path(__file__).resolve().parents[1]


def test_config_rejects_child_larger_than_parent():
    with pytest.raises(ValidationError):
        PipelineConfig(parent_max_tokens=100, child_max_tokens=120)


def test_config_rejects_overlap_ge_child_size():
    with pytest.raises(ValidationError):
        PipelineConfig(child_max_tokens=30, child_overlap_tokens=30)


def test_sample_corpus_loads_expected_documents():
    docs = load_documents(ROOT / "data" / "sample_corpus")
    names = {doc.source for doc in docs}
    assert "titan_security_whitepaper.md" in names
    assert "platform_parts_catalog.md" in names
    assert len(docs) >= 6


def test_source_document_rejects_empty_text():
    with pytest.raises(ValidationError):
        SourceDocument(source="empty.md", text="")
