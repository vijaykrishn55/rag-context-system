"""Token-budgeted context packing with parent/source metadata."""
from __future__ import annotations
from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr
from .schemas import RetrievedParent
from .tokenization import TokenCounter


@dataclass
class PackedContext:
    text: str
    parent_ids: list[str]
    sources: list[str]
    token_count: int
    budget: int
    omitted_parent_ids: list[str]


class ContextPacker:
    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.counter = token_counter or TokenCounter()

    def pack(self, documents: list[RetrievedParent], token_budget: int = 1500) -> PackedContext:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        blocks: list[str] = []
        parent_ids: list[str] = []
        sources: list[str] = []
        omitted: list[str] = []
        used = 0
        for doc in documents:
            source = str(doc.metadata.get("source", "unknown"))
            # Retrieved document text is external content, not executable markup.
            block = (f"<retrieved_context source={quoteattr(source)} parent_id={quoteattr(doc.id)}>\n"
                     f"{escape(doc.text.strip())}\n</retrieved_context>")
            count = self.counter.count(block)
            if used + count <= token_budget:
                blocks.append(block)
                parent_ids.append(doc.id)
                sources.append(source)
                used += count
            else:
                omitted.append(doc.id)
        return PackedContext("\n\n".join(blocks), parent_ids, sources, used, token_budget, omitted)
