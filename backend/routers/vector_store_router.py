"""
Router for vector-store endpoints.

Prefix: /api/vector-store

POST /api/vector-store/index
    Load a saved chunk set, embed all chunks, and upsert into Qdrant.

GET  /api/vector-store/collections
    List Qdrant collections managed by Chunky (filtered by collection prefix).

DELETE /api/vector-store/collections/{collection}
    Delete a named Qdrant collection.

POST /api/vector-store/search
    Semantic search within a collection given a natural-language query.

All CPU/IO-bound work (embedding, Qdrant calls) is dispatched via
``asyncio.to_thread`` so the event loop stays responsive.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.vector_store.schemas import (
    DeleteCollectionResponse,
    IndexRequest,
    IndexResponse,
)
from backend.services.vector_store_service import VectorStoreService

router = APIRouter(prefix="/api/vector-store", tags=["vector-store"])

# Module-level service instance.
_svc = VectorStoreService()


@router.post("/index", response_model=IndexResponse)
async def index_chunks(request: IndexRequest) -> IndexResponse:
    """Embed a saved chunk set and upsert into Qdrant.

    The handler:
    1. Loads the chunk set via the active storage backend.
    2. Embeds each chunk (``cleaned_chunk`` → ``content`` fallback).
    3. Derives a deterministic collection name (or uses the user-supplied one).
    4. Creates /recreates the Qdrant collection and upserts all points.
    """
    return await _svc.index_chunks(request)


@router.delete("/collections/{collection}", response_model=DeleteCollectionResponse)
async def delete_collection(collection: str) -> DeleteCollectionResponse:
    """Delete a named Qdrant collection."""
    return await _svc.delete_collection(collection)

