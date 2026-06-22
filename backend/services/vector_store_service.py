"""
Vector Store Service — orchestrates indexing chunks, searching, and collection management.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException

from backend.config import get_settings
from backend.vector_store.embedder import Embedder
from backend.vector_store.qdrant_store import QdrantVectorStore, collection_name_for
from backend.vector_store.schemas import (
    CollectionInfo,
    CollectionsResponse,
    DeleteCollectionResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
)
from backend.services.chunk_storage_service import get_chunk_storage

logger = logging.getLogger(__name__)


def _pick_embed_text(chunk: dict) -> str:
    """Return the best text to embed for a chunk.

    Prefers ``cleaned_chunk`` (post-enrichment) when non-empty, otherwise
    falls back to ``content`` (raw chunked text).
    """
    cleaned = (chunk.get("cleaned_chunk") or "").strip()
    return cleaned if cleaned else (chunk.get("content") or "").strip()


class VectorStoreService:
    """Handles all vector-store level operations: indexing chunks, listing collections,
    deleting collections, and semantic search.
    """

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._store = QdrantVectorStore()

    async def index_chunks(self, request: IndexRequest) -> IndexResponse:
        """Embed a saved chunk set and upsert into Qdrant.

        1. Loads the chunk set via the active storage backend.
        2. Embeds each chunk (``cleaned_chunk`` → ``content`` fallback).
        3. Derives a deterministic collection name (or uses the user-supplied one).
        4. Creates/recreates the Qdrant collection and upserts all points.
        """
        # 1. Load chunk set
        storage = get_chunk_storage()
        import inspect

        load_fn = storage.load_chunks_by_filename
        if inspect.iscoroutinefunction(load_fn):
            loaded = await load_fn(request.filename, request.chunks_filename)
        else:
            loaded = await asyncio.to_thread(load_fn, request.filename, request.chunks_filename)

        chunks = loaded.chunks
        if not chunks:
            raise HTTPException(status_code=422, detail="The selected chunk set is empty.")

        # 2. Pick texts to embed
        texts = [_pick_embed_text(c) for c in chunks]
        empty_indices = [i for i, t in enumerate(texts) if not t]
        if empty_indices:
            logger.warning(
                "index_chunks: %d chunk(s) have no embeddable text (indices: %s) — "
                "using a single space as placeholder.",
                len(empty_indices), empty_indices[:10],
            )
            texts = [t if t else " " for t in texts]

        # 3. Build collection name
        settings = get_settings()
        fn = request.chunks_filename

        md_source: str | None = None
        library: str | None = None
        algorithm: str | None = None
        chunk_size: int | None = None
        chunk_overlap: int | None = None

        if fn.startswith("db:"):
            # DB-backed storage: fetch chunk set metadata from PostgreSQL
            try:
                chunk_set_id = int(fn[3:])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid db chunk filename: {fn}",
                )
            from backend.db import get_session_factory  # noqa: PLC0415
            from backend.models.chunk_models import ChunkSetRecord  # noqa: PLC0415
            async with get_session_factory()() as session:
                chunk_set = await session.get(ChunkSetRecord, chunk_set_id)
            if chunk_set is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Chunk set id={chunk_set_id} not found.",
                )
            md_source = chunk_set.md_source
            library = chunk_set.chunker_library
            algorithm = chunk_set.chunker_type
            chunk_size = chunk_set.chunk_size
            chunk_overlap = chunk_set.chunk_overlap
        else:
            # Local file: parse via existing utility
            from backend.services.chunk_storage_service import _parse_chunk_filename  # noqa: PLC0415
            md_source, library, algorithm, chunk_size, chunk_overlap = _parse_chunk_filename(fn)

        col_name = collection_name_for(
            filename=request.filename,
            md_source=md_source,
            library=library,
            algorithm=algorithm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            prefix=settings.VECTOR_STORE_COLLECTION_PREFIX,
        )

        # 4. Embed
        logger.info(
            "Embedding %d chunks for collection '%s' …", len(texts), col_name
        )
        vectors: list[list[float]] = await asyncio.to_thread(self._embedder.embed, texts)

        # 5. Index into Qdrant (IO-bound — run in thread pool)
        indexed = await asyncio.to_thread(
            self._store.index_chunks, chunks, vectors, col_name, request.filename
        )

        return IndexResponse(
            success=True,
            collection=col_name,
            indexed_count=indexed,
            vector_dim=len(vectors[0]),
            message=(
                f"Indexed {indexed} chunk(s) from '{request.filename}' "
                f"into collection '{col_name}'."
            ),
        )

    async def list_collections(self) -> CollectionsResponse:
        """Return all Qdrant collections whose names start with the configured prefix."""
        settings = get_settings()
        prefix = settings.VECTOR_STORE_COLLECTION_PREFIX

        raw = await asyncio.to_thread(self._store.list_collections, prefix)
        return CollectionsResponse(
            collections=[
                CollectionInfo(
                    name=c["name"],
                    points_count=c["points_count"],
                    vector_size=c["vector_size"],
                )
                for c in raw
            ]
        )

    async def delete_collection(self, collection: str) -> DeleteCollectionResponse:
        """Delete a named Qdrant collection."""
        ok = await asyncio.to_thread(self._store.delete_collection, collection)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection}' not found or could not be deleted.",
            )
        return DeleteCollectionResponse(
            success=True,
            collection=collection,
            message=f"Collection '{collection}' deleted.",
        )

    async def search_chunks(self, request: SearchRequest) -> SearchResponse:
        """Semantic search within a Qdrant collection.

        Embeds the query with the same model used for indexing, then runs ANN
        search and returns the top-*k* results ordered by cosine similarity.
        """
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="Query must not be blank.")

        # Embed query
        query_vector: list[float] = await asyncio.to_thread(
            self._embedder.embed_query, request.query
        )

        # Search
        raw_hits = await asyncio.to_thread(
            self._store.search,
            request.collection,
            query_vector,
            request.top_k,
            request.score_threshold,
        )

        hits = [
            SearchHit(
                score=h["score"],
                chunk_index=h["chunk_index"],
                content=h["content"],
                cleaned_chunk=h["cleaned_chunk"],
                title=h["title"],
                context=h["context"],
                summary=h["summary"],
                keywords=h["keywords"],
                questions=h["questions"],
                parent_content=h["parent_content"],
                metadata=h["metadata"],
                filename=h["filename"],
                collection=h["collection"],
            )
            for h in raw_hits
        ]

        return SearchResponse(
            query=request.query,
            collection=request.collection,
            hits=hits,
        )
