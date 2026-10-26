from pathlib import Path
from advanced_rag.chunking import ParentChildChunker, parse_structural_blocks, semantic_split
from advanced_rag.schemas import PipelineConfig, SourceDocument
from advanced_rag.tokenization import TokenCounter

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_parser_preserves_fenced_code_as_one_block():
    text = "# Title\n\nA paragraph.\n\n```python\ndef hello():\n    return 'hi'\n```\n"
    blocks = parse_structural_blocks(text)
    code = [b for b in blocks if b.kind == "code"]
    assert len(code) == 1
    assert "def hello" in code[0].text
    assert code[0].text.startswith("```python") and code[0].text.endswith("```")


def test_oversized_paragraph_chunks_stay_under_hard_cap():
    counter = TokenCounter()
    text = " ".join(["retrieval context preserves useful semantic boundaries"] * 150)
    doc = SourceDocument(source="long.md", text=text)
    chunker = ParentChildChunker(PipelineConfig(parent_max_tokens=100, child_max_tokens=40, child_overlap_tokens=5, semantic_chunking=False), counter)
    parents, children = chunker.split_document(doc)
    assert len(parents) > 1
    assert all(counter.count(p.text) <= 100 for p in parents)
    assert all(counter.count(c.text) <= 100 for c in children)
    assert all(c.parent_id in {p.id for p in parents} for c in children)


def test_300_word_fenced_code_block_is_preserved_as_atomic_parent_and_child():
    counter = TokenCounter()
    words = ("readable comments preserve code structure and explain each reliable processing step "
             "without losing the surrounding function boundary ").split()
    comment = " ".join((words * 38)[:300])
    code = f"```python\ndef preserve_context():\n    # {comment}\n    return 'complete'\n```"
    doc = SourceDocument(source="code.md", text="# Code fixture\n\n" + code)
    config = PipelineConfig(parent_max_tokens=512, child_max_tokens=120, child_overlap_tokens=20, semantic_chunking=False)
    parents, children = ParentChildChunker(config, counter).split_document(doc)
    matching_parents = [p for p in parents if code in p.text]
    assert matching_parents, "Complete code fence should exist in one parent chunk"
    matching_parent = matching_parents[0]
    assert counter.count(matching_parent.text) <= 512
    matching_children = [c for c in children if c.parent_id == matching_parent.id]
    assert len(matching_children) == 1
    assert matching_children[0].text == matching_parent.text


def test_parent_child_max_token_relationship():
    doc = SourceDocument(source="doc.md", text="# Heading\n\n" + ("Some useful retrieval text. " * 35))
    counter = TokenCounter()
    config = PipelineConfig(parent_max_tokens=100, child_max_tokens=30, child_overlap_tokens=5, semantic_chunking=False)
    parents, children = ParentChildChunker(config, counter).split_document(doc)
    pmap = {p.id: p for p in parents}
    assert all(counter.count(pmap[c.parent_id].text) >= counter.count(c.text) for c in children)
    assert all(counter.count(p.text) <= config.parent_max_tokens for p in parents)


class _TopicEmbedder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            if "database" in text.lower() or "index" in text.lower():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def test_semantic_split_splits_large_topic_shift():
    text = (
        "Cats sleep in sunny places. Cats purr near people. "
        "The database index stores vectors. Search retrieves matching documents."
    )
    parts = semantic_split(text, _TopicEmbedder(), TokenCounter(), max_tokens=100)
    assert len(parts) >= 2


def test_sliding_child_chunks_use_configured_overlap():
    counter = TokenCounter()
    text = " ".join(f"term{i}" for i in range(100))
    chunks = counter.split(text, max_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    first = counter.encode(chunks[0])
    second = counter.encode(chunks[1])
    assert first[-5:] == second[:5]
