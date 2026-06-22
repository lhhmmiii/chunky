"""
SQLAlchemy ORM models for chunk set persistence.

Tables
------
chunk_sets
    Stores metadata for a single saved chunk run (library, algorithm, sizes,
    timestamps). Acts as the parent table.

chunks
    One row per individual chunk. Foreign-keyed to chunk_sets.
    Stores all enrichment fields: content, cleaned_chunk, title, context,
    summary, keywords, questions, metadata, start/end positions, and the
    parent-child fields (parent_id, parent_content).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


class ChunkSetRecord(Base):
    """Metadata for one saved chunk configuration for a document."""

    __tablename__ = "chunk_sets"

    # Unique constraint: same doc + md_source + library + algorithm +
    # chunk_size + chunk_overlap + enable_markdown_sizing always maps to
    # the same row, so re-saving with identical params is an upsert.
    __table_args__ = (
        UniqueConstraint(
            "filename",
            "md_source",
            "chunker_library",
            "chunker_type",
            "chunk_size",
            "chunk_overlap",
            "enable_markdown_sizing",
            name="uq_chunk_set_config",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    md_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunker_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunker_library: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_overlap: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enable_markdown_sizing: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    saved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(tz=timezone.utc),
    )

    # Relationship to individual chunks
    chunks: Mapped[list[ChunkRecord]] = relationship(
        "ChunkRecord",
        back_populates="chunk_set",
        cascade="all, delete-orphan",
        order_by="ChunkRecord.index",
    )


class ChunkRecord(Base):
    """One individual chunk belonging to a :class:`ChunkSetRecord`."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chunk_sets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Core content
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cleaned_chunk: Mapped[str] = mapped_column(Text, nullable=False, default="")
    start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Enrichment fields
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    context: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    keywords: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    # Parent-child chunking fields
    parent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parent_content: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Relationship back to the chunk set
    chunk_set: Mapped[ChunkSetRecord] = relationship(
        "ChunkSetRecord",
        back_populates="chunks",
    )
