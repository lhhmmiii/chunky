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



