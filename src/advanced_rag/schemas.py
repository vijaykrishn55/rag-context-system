"""Validated data contracts shared by the retrieval pipeline."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class PipelineConfig(BaseModel):
    """Runtime configuration. Token limits are safety caps, not model context sizes."""
    model_config = ConfigDict(extra="forbid")

    parent_max_tokens: int = Field(default=512, ge=32, le=4096)
    child_max_tokens: int = Field(default=120, ge=16, le=1024)
    child_overlap_tokens: int = Field(default=20, ge=0, le=256)
    dense_top_k: int = Field(default=50, ge=1, le=500)
    sparse_top_k: int = Field(default=50, ge=1, le=500)
    rerank_top_n: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    context_token_budget: int = Field(default=1500, ge=64, le=32000)
    semantic_chunking: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.child_max_tokens > self.parent_max_tokens:
            raise ValueError("child_max_tokens cannot exceed parent_max_tokens")
        if self.child_overlap_tokens >= self.child_max_tokens:
            raise ValueError("child_overlap_tokens must be smaller than child_max_tokens")


class SourceDocument(BaseModel):
    """Normalized source document before chunking."""
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentChunk(BaseModel):
    id: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildChunk(BaseModel):
    id: str
    parent_id: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedParent(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float | None = None
    matched_child_ids: list[str] = Field(default_factory=list)


class EvaluationQuery(BaseModel):
    query_id: str
    query: str
    relevant_sources: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
