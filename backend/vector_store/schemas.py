"""
Pydantic schemas for the vector-store API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Body for ``POST /api/vector-store/index``."""

    filename: str = Field(
        ..., description="Document filename (e.g. 'report.pdf')."
    )
    chunks_filename: str = Field(
        ...,
        description=(
            "Saved-chunk identifier — either a 'db:<id>' string (DB backend) "
            "or the bare JSON filename (local backend)."
        ),
    )
    collection_name: str | None = Field(
        default=None,
        description=(
            "Qdrant collection name.  When null, a deterministic name is "
            "derived from the document filename and chunking configuration."
        ),
    )


class IndexResponse(BaseModel):
    """Response from ``POST /api/vector-store/index``."""

    success: bool
    collection: str
    indexed_count: int
    vector_dim: int
    message: str


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CollectionInfo(BaseModel):
    """Summary of a single Qdrant collection managed by Chunky."""

    name: str
    points_count: int
    vector_size: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


class DeleteCollectionResponse(BaseModel):
    success: bool
    collection: str
    message: str


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Body for ``POST /api/vector-store/search``."""

    query: str = Field(..., min_length=1, description="Natural-language query.")
    collection: str = Field(..., description="Qdrant collection name to search.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return.")
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0–1). Results below this are dropped.",
    )


class SearchHit(BaseModel):
    """A single search result returned by the semantic search endpoint."""

    score: float
    chunk_index: int
    content: str
    cleaned_chunk: str
    title: str
    context: str
    summary: str
    keywords: list[str]
    questions: list[str]
    parent_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Source document info stored in the point payload
    filename: str = ""
    collection: str = ""


class SearchResponse(BaseModel):
    query: str
    collection: str
    hits: list[SearchHit]
