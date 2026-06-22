"""
Embedder — wraps SentenceTransformer for Vietnamese text embeddings.

The model is loaded lazily on first use and cached as a singleton
per process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: "SentenceTransformer | None" = None


def _load_model(model_id: str) -> "SentenceTransformer":
    """Load (or return cached) embedding model."""
    global _model

    if _model is None:
        logger.info("Loading embedding model '%s'...", model_id)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed."
            ) from exc

        _model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
        )

        logger.info(
            "Embedding model '%s' loaded (dim=%d)",
            model_id,
            _model.get_sentence_embedding_dimension(),
        )

    return _model


class Embedder:
    """Vietnamese embedding wrapper."""

    def __init__(
        self,
        model_id: str | None = None,
        batch_size: int | None = None,
    ):
        settings = get_settings()

        self._model_id = (
            model_id
            or settings.VECTOR_STORE_EMBEDDING_MODEL
            or "AITeamVN/Vietnamese_Embedding"
        )

        self._batch_size = (
            batch_size
            or settings.VECTOR_STORE_EMBEDDING_BATCH_SIZE
            or 64
        )

    @property
    def dim(self) -> int:
        """
        Get the dimension of the embedding model.
        """
        return _load_model(
            self._model_id
        ).get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: The list of text strings to embed.

        Returns:
            A list of lists of floats representing the embeddings.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        model = _load_model(self._model_id)

        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string.

        Args:
            query: The query string to embed.

        Returns:
            A list of floats representing the embedding.
        """
        return self.embed([query])[0]