"""
Qdrant-backed vector store for Chunky.

Responsibilities
----------------
* Build deterministic collection names from chunk-set configuration.
* Create / recreate Qdrant collections on index.
* Upsert chunk vectors with full payload for self-contained retrieval.
* ANN search with optional score threshold.
* List and delete managed collections.

All Qdrant I/O uses the synchronous HTTP client.  Async wrappers sit in
the router so the event loop stays responsive — the actual Qdrant calls
are dispatched via ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from backend.config import get_settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Module-level Qdrant client singleton — created once per process.
_client: "QdrantClient | None" = None


def _get_client() -> "QdrantClient":
    """Return (or lazily create) the module-level Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        try:
            from qdrant_client import QdrantClient as _QdrantClient  # lazy import
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. "
                "Add 'qdrant-client' to requirements.txt."
            ) from exc
        logger.info("Connecting to Qdrant at %s …", settings.QDRANT_URL)
        _client = _QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )
        logger.info("Qdrant client ready.")
    return _client


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


# ---------------------------------------------------------------------------
# Main store class
# ---------------------------------------------------------------------------


class QdrantVectorStore:
    """Manages embedding-backed collections in Qdrant."""

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        collection: str,
        filename: str,
    ) -> int:
        """Create (or recreate) *collection* and upsert all chunk vectors.

        The previous collection is deleted and recreated so re-indexing with
        the same configuration always produces a clean result.

        Args:
            chunks:     Normalised chunk dicts from the storage service.
            vectors:    Embedding vectors, one per chunk, same order as *chunks*.
            collection: Target Qdrant collection name.
            filename:   Source document filename stored in the point payload.

        Returns:
            The number of points upserted.
        """
        from qdrant_client.models import (  # lazy import
            Distance,
            PointStruct,
            VectorParams,
        )

        client = _get_client()
        vector_dim = len(vectors[0])

        # Delete existing collection (ignore 404).
        try:
            client.delete_collection(collection)
            logger.debug("Deleted existing collection '%s'.", collection)
        except Exception:
            pass  # Collection didn't exist — that's fine.

        # Recreate with cosine distance (standard for semantic search).
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d).", collection, vector_dim
        )

        # Build point structs with full payload for self-contained retrieval.
        points: list[PointStruct] = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload: dict[str, Any] = {
                "filename": filename,
                "collection": collection,
                "chunk_index": chunk.get("index", i),
                "content": chunk.get("content", ""),
                "cleaned_chunk": chunk.get("cleaned_chunk", ""),
                "title": chunk.get("title", ""),
                "context": chunk.get("context", ""),
                "summary": chunk.get("summary", ""),
                "keywords": chunk.get("keywords", []),
                "questions": chunk.get("questions", []),
                "parent_content": chunk.get("parent_content", ""),
                "parent_id": chunk.get("parent_id"),
                "metadata": chunk.get("metadata", {}),
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
            }
            points.append(PointStruct(id=i, vector=vector, payload=payload))

        # Upsert in batches to avoid large request payloads.
        batch_size = 256
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection,
                points=points[start : start + batch_size],
            )

        logger.info(
            "Indexed %d chunks into collection '%s'.", len(points), collection
        )
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Perform ANN search and return scored chunk payloads.

        Args:
            collection:      Qdrant collection name to search.
            query_vector:    Embedding of the query text.
            top_k:           Maximum number of results.
            score_threshold: Minimum cosine similarity (0–1).  Results below
                             this are dropped when specified.

        Returns:
            List of dicts with ``score`` and all chunk payload fields.
        """
        client = _get_client()

        kwargs: dict[str, Any] = {
            "collection_name": collection,
            "query_vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold

        hits = client.search(**kwargs)

        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "chunk_index": payload.get("chunk_index", 0),
                    "content": payload.get("content", ""),
                    "cleaned_chunk": payload.get("cleaned_chunk", ""),
                    "title": payload.get("title", ""),
                    "context": payload.get("context", ""),
                    "summary": payload.get("summary", ""),
                    "keywords": payload.get("keywords", []),
                    "questions": payload.get("questions", []),
                    "parent_content": payload.get("parent_content", ""),
                    "metadata": payload.get("metadata", {}),
                    "filename": payload.get("filename", ""),
                    "collection": payload.get("collection", collection),
                }
            )
        return results

    # ------------------------------------------------------------------
    # List / delete
    # ------------------------------------------------------------------

    def list_collections(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """Return all Qdrant collections, optionally filtered by name prefix.

        Args:
            prefix: When provided, only collections whose names start with
                    this prefix are returned.

        Returns:
            List of dicts with ``name``, ``points_count``, ``vector_size``.
        """
        client = _get_client()
        response = client.get_collections()
        infos: list[dict[str, Any]] = []
        for col in response.collections:
            if prefix and not col.name.startswith(prefix):
                continue
            try:
                detail = client.get_collection(col.name)
                points = detail.points_count or 0
                vec_size = 0
                if detail.config and detail.config.params and detail.config.params.vectors:
                    vp = detail.config.params.vectors
                    # VectorParams directly or a dict of named vectors
                    if hasattr(vp, "size"):
                        vec_size = vp.size
                    elif isinstance(vp, dict):
                        first = next(iter(vp.values()), None)
                        if first and hasattr(first, "size"):
                            vec_size = first.size
            except Exception:
                points = 0
                vec_size = 0
            infos.append(
                {"name": col.name, "points_count": points, "vector_size": vec_size}
            )
        return infos

    def delete_collection(self, collection: str) -> bool:
        """Delete a collection.  Returns True on success, False if not found."""
        client = _get_client()
        try:
            client.delete_collection(collection)
            logger.info("Deleted Qdrant collection '%s'.", collection)
            return True
        except Exception as exc:
            logger.warning("Failed to delete collection '%s': %s", collection, exc)
            return False
