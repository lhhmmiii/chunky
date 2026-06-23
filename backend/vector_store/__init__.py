"""
Vector store package for Chunky.

Provides chunk embedding and Qdrant-backed semantic search/retrieval.

Public API
----------
    get_vector_store() → QdrantVectorStore
    Embedder
"""

from backend.vector_store.qdrant_store import VectorDbManager 

__all__ = ["VectorDbManager"]
