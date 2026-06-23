"""
Vector Store Service — orchestrates indexing chunks, searching, and collection management.
"""

from __future__ import annotations

import asyncio
import logging

import re
from fastapi import HTTPException

from backend.config import get_settings
from backend.vector_store.qdrant_store import VectorDbManager
from backend.vector_store.schemas import (
    CollectionInfo,
    CollectionsResponse,
    DeleteCollectionResponse,
    IndexRequest,
    IndexResponse,
)
from backend.services.chunk_storage_service import get_chunk_storage

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Handles all vector-store level operations: indexing chunks, listing collections,
    deleting collections, and semantic search.
    """

    _vector_db_manager = VectorDbManager()

    def __init__(self) -> None:
        pass

    async def index_chunks(self, request: IndexRequest) -> IndexResponse:
        """Embed a saved chunk set and upsert into Qdrant.

        1. Loads the chunk set via the active storage backend.
        2. Derives a deterministic collection name (or uses the user-supplied one).
        3. Creates/validates the Qdrant collection.
        4. Upserts all points via QdrantVectorStore.add_texts (handles embedding internally).
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

        # 4. Create (or validate) the collection
        await asyncio.to_thread(self._vector_db_manager.create_collection, col_name)

        # 5. Get QdrantVectorStore for this collection and upsert texts
        store = await asyncio.to_thread(self._vector_db_manager.get_collection, col_name)

        metadatas = [
            {
                "source": request.filename,
                "chunk_index": i,
                **(chunk if isinstance(chunk, dict) else {}),
            }
            for i, chunk in enumerate(chunks)
        ]

        logger.info("Indexing %d chunks into collection '%s' …", len(texts), col_name)
        ids = await asyncio.to_thread(store.add_texts, texts, metadatas)

        indexed = len(ids)
        vector_dim = await asyncio.to_thread(self._vector_db_manager._dense_vector_size)

        return IndexResponse(
            success=True,
            collection=col_name,
            indexed_count=indexed,
            vector_dim=vector_dim,
            message=(
                f"Indexed {indexed} chunk(s) from '{request.filename}' "
                f"into collection '{col_name}'."
            ),
        )

    async def delete_collection(self, collection: str) -> DeleteCollectionResponse:
        """Delete a named Qdrant collection."""
        try:
            await asyncio.to_thread(self._vector_db_manager.delete_collection, collection)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection}' not found or could not be deleted.",
            ) from exc
        return DeleteCollectionResponse(
            success=True,
            collection=collection,
            message=f"Collection '{collection}' deleted.",
        )

# ---------------------------------------------------------------------------
# Pick the cleaned chunk(if have) or raw chunk
# ---------------------------------------------------------------------------

def _pick_embed_text(chunk: dict) -> str:
    """Return the best text to embed for a chunk.

    Prefers ``cleaned_chunk`` (post-enrichment) when non-empty, otherwise
    falls back to ``content`` (raw chunked text).
    """
    cleaned = (chunk.get("cleaned_chunk") or "").strip()
    return cleaned if cleaned else (chunk.get("content") or "").strip()

# ---------------------------------------------------------------------------
# Collection name helpers
# ---------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _safe_token(value: str, max_len: int = 40) -> str:
    """Sanitise a string for use as part of a Qdrant collection name."""
    sanitised = _UNSAFE_RE.sub("_", value.strip().lower())
    return sanitised[:max_len]


def collection_name_for(
    filename: str,
    md_source: str | None,
    library: str | None,
    algorithm: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    prefix: str | None = None,
) -> str:
    """Build a deterministic Qdrant collection name.

    Pattern::

        {prefix}__{stem}__{md_source}__{library}-{algo}[__{size}[__{overlap}]]

    where ``prefix`` defaults to ``VECTOR_STORE_COLLECTION_PREFIX``.
    """
    settings = get_settings()
    _prefix = _safe_token(prefix or settings.VECTOR_STORE_COLLECTION_PREFIX)

    # Strip extension from filename to get the document stem
    stem = _safe_token(filename.rsplit(".", 1)[0] if "." in filename else filename)
    _md = _safe_token(md_source or "uploaded")
    _lib = _safe_token(library or "unknown")
    _algo = _safe_token(algorithm or "unknown")
    libalgo = f"{_lib}-{_algo}"

    parts = [_prefix, stem, _md, libalgo]
    if chunk_size is not None:
        parts.append(str(int(chunk_size)))
        if chunk_overlap is not None:
            parts.append(str(int(chunk_overlap)))

    return "__".join(parts)