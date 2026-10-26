"""Structure-aware, semantic, and parent-child chunking utilities."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Protocol
from .schemas import SourceDocument, ParentChunk, ChildChunk, PipelineConfig
from .tokenization import TokenCounter


@dataclass
class StructuralBlock:
    text: str
    kind: str  # heading, paragraph, code, table
    heading: str = ""
    semantic_boundary: bool = False


class TextEmbedder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any): ...


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


def parse_structural_blocks(text: str) -> list[StructuralBlock]:
    """Keep fenced code blocks intact and split prose around Markdown structure."""
    lines = text.splitlines()
    blocks: list[StructuralBlock] = []
    paragraph: list[str] = []
    code: list[str] = []
    fence = None
    current_heading = ""

    def flush_paragraph() -> None:
        nonlocal paragraph
        content = "\n".join(paragraph).strip()
        if content:
            blocks.append(StructuralBlock(content, "paragraph", current_heading))
        paragraph = []

    i = 0
    while i < len(lines):
        line = lines[i]
        fence_match = _FENCE_RE.match(line)
        if fence is None and fence_match:
            flush_paragraph()
            fence = fence_match.group(1)[:3]
            code = [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                if lines[i].lstrip().startswith(fence):
                    break
                i += 1
            # An unclosed fence is still retained as a single code block.
            blocks.append(StructuralBlock("\n".join(code).strip(), "code", current_heading))
            code = []
            fence = None
            i += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            current_heading = heading_match.group(2).strip()
            blocks.append(StructuralBlock(line.strip(), "heading", current_heading))
            i += 1
            continue

        # Preserve markdown table rows as a single structural block.
        if line.strip().startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            blocks.append(StructuralBlock("\n".join(table_lines), "table", current_heading))
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.rstrip())
        i += 1

    flush_paragraph()
    return blocks


def _cosine_distance(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denom == 0 else 1.0 - float(np.dot(a, b) / denom)


def semantic_split(text: str, embedder: TextEmbedder, token_counter: TokenCounter,
                   max_tokens: int, threshold_std: float = 0.75,
                   min_sentences: int = 3) -> list[str]:
    """Split prose where adjacent-sentence embedding distance rises unusually high.

    Embedding distance is used as a boundary signal. Oversized resulting segments are
    still split to respect the configured token cap.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`#])", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < min_sentences:
        return token_counter.split(text, max_tokens)
    vectors = embedder.encode(sentences, normalize_embeddings=True)
    distances = [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    try:
        import numpy as np
        mean = float(np.mean(distances))
        spread = float(np.std(distances))
    except Exception:
        mean = sum(distances) / len(distances)
        variance = sum((d - mean) ** 2 for d in distances) / len(distances)
        spread = variance ** 0.5
    threshold = mean + threshold_std * spread
    # With very small samples, mean + std can exceed the largest observed distance.
    # Cap below the observed maximum when there is a meaningful spread, so a clear
    # topic shift cannot be hidden by the threshold formula.
    if max(distances) - min(distances) > 1e-8:
        threshold = min(threshold, max(distances) - 1e-6)
    segments: list[str] = []
    current = [sentences[0]]
    for i, distance in enumerate(distances):
        if distance > threshold:
            segments.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        segments.append(" ".join(current))
    output: list[str] = []
    for segment in segments:
        output.extend(token_counter.split(segment, max_tokens))
    return output


def _split_oversized_code(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Split oversized fenced code at function/class boundaries before line boundaries."""
    if counter.count(text) <= max_tokens:
        return [text]
    lines = text.splitlines()
    has_fence = bool(lines and _FENCE_RE.match(lines[0]))
    fence_open = lines[0] if has_fence else "```"
    body = lines[1:] if has_fence else lines[:]
    if has_fence and body and _FENCE_RE.match(body[-1]):
        body = body[:-1]

    boundaries = [0]
    for i, line in enumerate(body):
        if i > 0 and re.match(r"^(\s{0,4})(async\s+)?(def|class)\s+", line):
            boundaries.append(i)
    boundaries.append(len(body))
    function_units = ["\n".join(body[boundaries[i]:boundaries[i + 1]]).strip()
                      for i in range(len(boundaries) - 1)]
    function_units = [unit for unit in function_units if unit]

    safe_units: list[str] = []
    for unit in function_units:
        wrapped_unit = f"{fence_open}\n{unit}\n```" if has_fence else unit
        if counter.count(wrapped_unit) <= max_tokens:
            safe_units.append(unit)
            continue
        # A single oversized function has no complete function boundary available;
        # fall back to line boundaries and keep every result within the hard cap.
        current: list[str] = []
        for line in unit.splitlines():
            proposed = "\n".join(current + [line])
            wrapped = f"{fence_open}\n{proposed}\n```" if has_fence else proposed
            if current and counter.count(wrapped) > max_tokens:
                safe_units.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            safe_units.append("\n".join(current).strip())

    chunks = []
    for unit in safe_units:
        wrapped = f"{fence_open}\n{unit}\n```" if has_fence else unit
        if counter.count(wrapped) > max_tokens:
            # A single source line may itself exceed the configured hard cap.
            chunks.extend(counter.split(wrapped, max_tokens))
        else:
            chunks.append(wrapped.strip())
    return chunks


class ParentChildChunker:
    def __init__(self, config: PipelineConfig | None = None,
                 token_counter: TokenCounter | None = None,
                 semantic_embedder: TextEmbedder | None = None) -> None:
        self.config = config or PipelineConfig()
        self.counter = token_counter or TokenCounter()
        self.semantic_embedder = semantic_embedder

    def split_document(self, document: SourceDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
        blocks = parse_structural_blocks(document.text)
        if not blocks:
            return [], []
        # Turn prose blocks into semantic segments when an embedding model is available.
        units: list[StructuralBlock] = []
        for block in blocks:
            if block.kind == "code":
                pieces = _split_oversized_code(block.text, self.config.parent_max_tokens, self.counter)
                units.extend(StructuralBlock(piece, "code", block.heading) for piece in pieces)
            elif block.kind in {"paragraph", "table"} and self.semantic_embedder and self.config.semantic_chunking:
                if self.counter.count(block.text) > self.config.parent_max_tokens or len(re.findall(r"(?<=[.!?])\s+", block.text)) >= 3:
                    semantic_parts = semantic_split(block.text, self.semantic_embedder, self.counter,
                                                    self.config.parent_max_tokens)
                    if len(semantic_parts) > 1:
                        units.extend(StructuralBlock(part, block.kind, block.heading, True) for part in semantic_parts)
                    else:
                        units.append(block)
                else:
                    units.append(block)
            elif self.counter.count(block.text) > self.config.parent_max_tokens:
                units.extend(StructuralBlock(part, block.kind, block.heading)
                             for part in self.counter.split(block.text, self.config.parent_max_tokens))
            else:
                units.append(block)

        parent_texts: list[tuple[str, str]] = []
        current_parts: list[str] = []
        current_kinds: list[str] = []
        current_heading = ""
        for unit in units:
            text = unit.text.strip()
            if not text:
                continue
            proposed = "\n\n".join(current_parts + [text])
            size_overflow = bool(current_parts) and self.counter.count(proposed) > self.config.parent_max_tokens
            # Keep a heading with its first semantic segment. Fenced code is atomic when
            # it fits the cap; semantic segments become parent boundaries after being added.
            semantic_before = (unit.semantic_boundary and current_parts
                               and any(kind != "heading" for kind in current_kinds))
            code_before = unit.kind == "code" and bool(current_parts)
            if current_parts and (size_overflow or semantic_before or code_before):
                parent_texts.append(("\n\n".join(current_parts), current_heading))
                current_parts = []
                current_kinds = []
                current_heading = ""
            if not current_parts:
                current_heading = unit.heading
            current_parts.append(text)
            current_kinds.append(unit.kind)
            if unit.kind == "code" or unit.semantic_boundary:
                parent_texts.append(("\n\n".join(current_parts), current_heading))
                current_parts = []
                current_kinds = []
                current_heading = ""
        if current_parts:
            parent_texts.append(("\n\n".join(current_parts), current_heading))

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for p_idx, (parent_text, heading) in enumerate(parent_texts):
            parent_id = f"{document.source}::parent-{p_idx:04d}"
            metadata = {**document.metadata, "source": document.source, "heading": heading,
                        "parent_index": p_idx, "token_count": self.counter.count(parent_text)}
            parent = ParentChunk(id=parent_id, text=parent_text, metadata=metadata)
            parents.append(parent)
            parent_blocks = parse_structural_blocks(parent_text)
            # Keep a complete fenced block atomic in the child index when it fits under
            # the hard parent cap. This may exceed the smaller child target, but never
            # the configured hard maximum; code is not bisected just to hit a target size.
            atomic_code = (len(parent_blocks) == 1 and parent_blocks[0].kind == "code"
                           and self.counter.count(parent_text) <= self.config.parent_max_tokens)
            child_texts = ([parent_text] if atomic_code else
                           self.counter.split(parent_text, self.config.child_max_tokens,
                                              self.config.child_overlap_tokens))
            for c_idx, child_text in enumerate(child_texts):
                child_meta = {**metadata, "parent_id": parent_id, "child_index": c_idx,
                              "token_count": self.counter.count(child_text)}
                children.append(ChildChunk(
                    id=f"{parent_id}::child-{c_idx:04d}", parent_id=parent_id,
                    text=child_text, metadata=child_meta,
                ))
        for parent, child in ((p, c) for p in parents for c in children if c.parent_id == p.id):
            if self.counter.count(parent.text) < self.counter.count(child.text):
                raise ValueError(f"Parent token count is smaller than child for {child.id}")
        return parents, children

    def split_corpus(self, documents: list[SourceDocument]) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for document in documents:
            p, c = self.split_document(document)
            parents.extend(p)
            children.extend(c)
        return parents, children
