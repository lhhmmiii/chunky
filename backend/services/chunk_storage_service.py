"""
Chunk storage service — persists and loads enriched chunk sets to/from disk.

Each chunk is stored with the full enriched schema:
    Chunk, CleanedChunk, Title, Context, Summary, Keywords, Questions.
Fields that have not yet been populated (pre-enrichment) are stored as empty
strings / empty lists and will be filled in by the enrichment pipeline later.

Filename format
---------------
Saved chunk files are addressable by *configuration*, not by time.  The path is::

    chunks/<stem>/<doc>_<md_source>_<library>-<algorithm>[_<size>[_<overlap>]].json

* ``<md_source>`` identifies the Markdown variant the chunks were generated
  from (``pymupdf4llm``, ``docling``, …, or ``uploaded``).  Without this
  segment, chunking the same PDF with different converters but the same
  algorithm would silently overwrite each other's saved files.
* The ``<library>-<algorithm>`` segment ALWAYS contains a hyphen so the
  parser can locate it unambiguously even when the document name itself
  contains underscores.  Wire-level chunker-type values that contain
  underscores (only ``line_based`` today) are rewritten with hyphens
  inside the filename.
* ``<size>`` and ``<overlap>`` are appended only when the chosen algorithm
  actually consumes them.

Re-saving with the same configuration overwrites in place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.config import get_settings
from backend.models.schemas import (
    ChunksVersion,
    LoadChunksResponse,
    SaveChunksRequest,
    SaveChunksResponse,
)
from backend.utils.naming import (
    KNOWN_LIBRARIES,
    KNOWN_MD_SOURCES,
    algo_from_filename_token,
    algo_to_filename_token,
    md_source_token,
    params_used_by,
    sanitise_token,
)
from backend.utils.path import safe_child_path, safe_stem as _safe_stem


def _build_chunk_filename(
    doc_name: str,
    md_source: str,
    library: str | None,
    chunker_type: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    enable_markdown_sizing: bool,
) -> str:
    """Build the deterministic chunk-file name for a given configuration."""
    lib_token = sanitise_token((library or "unknown").lower()) or "unknown"
    algo_token = algo_to_filename_token(chunker_type or "unknown") or "unknown"
    libalgo = f"{lib_token}-{algo_token}"

    parts: list[str] = [doc_name, md_source or "uploaded", libalgo]
    uses_size, uses_overlap = params_used_by(library or "", chunker_type or "", enable_markdown_sizing)
    if uses_size and chunk_size is not None:
        parts.append(str(int(chunk_size)))
        if uses_overlap and chunk_overlap is not None:
            parts.append(str(int(chunk_overlap)))
    return "_".join(parts) + ".json"


def _parse_chunk_filename(
    filename: str,
) -> tuple[str | None, str, str, int | None, int | None]:
    """Extract ``(md_source, library, algorithm, chunk_size, chunk_overlap)``.

    The library/algorithm segment is identified as the first hyphenated token
    after splitting on ``_``, preferring tokens whose left side is a known
    library.  Anything that doesn't match falls back gracefully so listing
    is never bricked by a stray legacy file.
    """
    if not filename.endswith(".json"):
        return None, "unknown", "unknown", None, None
    tokens = filename[: -len(".json")].split("_")

    libalgo_idx: int | None = None
    for i, tok in enumerate(tokens):
        if "-" not in tok:
            continue
        if tok.split("-", 1)[0] in KNOWN_LIBRARIES:
            libalgo_idx = i
            break
    if libalgo_idx is None:
        for i, tok in enumerate(tokens):
            if "-" in tok:
                libalgo_idx = i
                break
    if libalgo_idx is None:
        return None, "unknown", "unknown", None, None

    library, _, algo_token = tokens[libalgo_idx].partition("-")
    algorithm = algo_from_filename_token(algo_token) if algo_token else "unknown"

    md_source: str | None = None
    if libalgo_idx >= 1 and tokens[libalgo_idx - 1] in KNOWN_MD_SOURCES:
        md_source = tokens[libalgo_idx - 1]

    rest = tokens[libalgo_idx + 1:]
    size = int(rest[0]) if len(rest) >= 1 and rest[0].isdigit() else None
    overlap = int(rest[1]) if len(rest) >= 2 and rest[1].isdigit() else None
    return md_source, library or "unknown", algorithm or "unknown", size, overlap


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for raw in value:
            item = _as_str(raw).strip()
            if item:
                items.append(item)
        return items
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalise_chunk(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a chunk dict to snake_case, filling missing enrichment fields.

    Accepts both snake_case and legacy PascalCase keys so the function is safe
    on incoming request data (write path) and on stored JSON (read path).
    """
    return {
        "index": _as_int(raw.get("index", 0)),
        "content": _as_str(raw.get("content", raw.get("Chunk", ""))),
        "cleaned_chunk": _as_str(raw.get("cleaned_chunk", raw.get("CleanedChunk", ""))),
        "title": _as_str(raw.get("title", raw.get("Title", ""))),
        "context": _as_str(raw.get("context", raw.get("Context", ""))),
        "summary": _as_str(raw.get("summary", raw.get("Summary", ""))),
        "keywords": _as_str_list(raw.get("keywords", raw.get("Keywords", []))),
        "questions": _as_str_list(raw.get("questions", raw.get("Questions", []))),
        "metadata": _as_dict(raw.get("metadata", {})),
        "start": _as_int(raw.get("start", 0)),
        "end": _as_int(raw.get("end", 0)),
        "parent_id": raw.get("parent_id"),  # None for non-parent-child chunks
        "parent_content": _as_str(raw.get("parent_content", "")),
    }


def _md_filename_for_source(stem: str, md_source: str | None) -> str | None:
    """Reconstruct the Markdown filename a chunk file was generated from.

    Saved chunk JSON only stores ``md_source`` (the identity key) — the full
    filename is computed on read so the on-disk payload stays minimal.
    """
    if md_source is None:
        return None
    if md_source == "uploaded":
        return f"{stem}.md"
    return f"{stem}_{md_source}.md"


class ChunkStorageService:
    """Saves enriched chunk sets to deterministic, configuration-keyed files."""

    def __init__(self) -> None:
        self._chunks_dir = Path(get_settings().CHUNKS_DIR)

    def save_chunks(self, request: SaveChunksRequest) -> SaveChunksResponse:
        """Persist *request.chunks* to a configuration-keyed JSON file.

        Re-saving with the same MD source + library / algorithm / size /
        overlap overwrites the previous file deterministically.
        """
        stem = _safe_stem(request.filename)
        doc_name = sanitise_token(stem) or "doc"
        dest_dir = self._chunks_dir / stem
        dest_dir.mkdir(parents=True, exist_ok=True)

        md_source = md_source_token(request.md_filename, stem)
        dest_path = dest_dir / _build_chunk_filename(
            doc_name=doc_name,
            md_source=md_source,
            library=request.chunker_library,
            chunker_type=request.chunker_type,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            enable_markdown_sizing=request.enable_markdown_sizing,
        )

        normalised_chunks = [_normalise_chunk(c) for c in request.chunks]
        payload: dict[str, Any] = {
            "filename": request.filename,
            "md_source": md_source,
            "chunker_type": request.chunker_type,
            "chunker_library": request.chunker_library,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "enable_markdown_sizing": request.enable_markdown_sizing,
            "parent_chunk_size": request.parent_chunk_size,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_chunks": len(normalised_chunks),
            "chunks": normalised_chunks,
        }
        dest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return SaveChunksResponse(
            success=True,
            message=f"Saved {len(normalised_chunks)} chunks for '{request.filename}'",
            path=str(dest_path),
        )

    def load_chunks(self, filename: str) -> LoadChunksResponse:
        """Load the most recently *modified* saved chunk file for *filename*."""
        stem = _safe_stem(filename)
        dest_dir = self._chunks_dir / stem
        try:
            json_files = list(dest_dir.glob("*.json"))
        except (FileNotFoundError, OSError):
            json_files = []

        if not json_files:
            raise HTTPException(
                status_code=404,
                detail=f"No saved chunks found for '{filename}'",
            )
        json_files.sort(key=lambda p: p.stat().st_mtime)
        return self._read_chunk_file(json_files[-1])

    def load_chunks_by_filename(self, filename: str, chunks_filename: str) -> LoadChunksResponse:
        """Load a specific saved-chunks JSON file by its filename."""
        stem = _safe_stem(filename)
        dest_path = safe_child_path(
            self._chunks_dir / stem, chunks_filename, description="chunks filename",
        )
        if not dest_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Chunks file '{chunks_filename}' not found for '{filename}'",
            )
        return self._read_chunk_file(dest_path)

    def list_versions(self, filename: str) -> list[ChunksVersion]:
        """Return every saved-chunks JSON file for *filename*, newest first."""
        stem = _safe_stem(filename)
        dest_dir = self._chunks_dir / stem
        try:
            json_files = list(dest_dir.glob("*.json"))
        except (FileNotFoundError, OSError):
            return []
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        versions: list[ChunksVersion] = []
        for f in json_files:
            md_source, library, algorithm, size, overlap = _parse_chunk_filename(f.name)
            versions.append(ChunksVersion(
                filename=f.name,
                md_filename=_md_filename_for_source(stem, md_source),
                md_source=md_source,
                library=library,
                algorithm=algorithm,
                chunk_size=size,
                chunk_overlap=overlap,
                file_path=str(f),
            ))
        return versions

    def _read_chunk_file(self, path: Path) -> LoadChunksResponse:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            normalised = [_normalise_chunk(c) for c in payload["chunks"]]
            return LoadChunksResponse(
                chunks=normalised,
                total_chunks=payload["total_chunks"],
                filename=payload["filename"],
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Saved chunk file is corrupt: {exc}")
        except KeyError as exc:
            raise HTTPException(status_code=500, detail=f"Saved chunk file is missing field: {exc}")


# ---------------------------------------------------------------------------
# Database-backed storage service
# ---------------------------------------------------------------------------


import asyncio
import logging as _logging

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_session_factory
from backend.models.chunk_models import ChunkRecord, ChunkSetRecord

_db_logger = _logging.getLogger(__name__)


def _chunk_item_to_record(
    chunk: dict[str, Any],
    chunk_set_id: int,
) -> ChunkRecord:
    """Convert a normalised chunk dict to a :class:`ChunkRecord` ORM row."""
    return ChunkRecord(
        chunk_set_id=chunk_set_id,
        index=_as_int(chunk.get("index", 0)),
        content=_as_str(chunk.get("content", "")),
        cleaned_chunk=_as_str(chunk.get("cleaned_chunk", "")),
        title=_as_str(chunk.get("title", "")),
        context=_as_str(chunk.get("context", "")),
        summary=_as_str(chunk.get("summary", "")),
        keywords=_as_str_list(chunk.get("keywords", [])),
        questions=_as_str_list(chunk.get("questions", [])),
        metadata_=_as_dict(chunk.get("metadata", {})),
        start=_as_int(chunk.get("start", 0)),
        end=_as_int(chunk.get("end", 0)),
        parent_id=chunk.get("parent_id"),
        parent_content=_as_str(chunk.get("parent_content", "")),
    )


def _record_to_chunk_dict(record: ChunkRecord) -> dict[str, Any]:
    """Convert a :class:`ChunkRecord` ORM row to a normalised chunk dict."""
    return {
        "index": record.index,
        "content": record.content,
        "cleaned_chunk": record.cleaned_chunk,
        "title": record.title,
        "context": record.context,
        "summary": record.summary,
        "keywords": record.keywords or [],
        "questions": record.questions or [],
        "metadata": record.metadata_ or {},
        "start": record.start,
        "end": record.end,
        "parent_id": record.parent_id,
        "parent_content": record.parent_content or "",
    }


class DatabaseChunkStorageService:
    """Saves and loads enriched chunk sets via PostgreSQL (async SQLAlchemy).

    The DB schema mirrors the local-file storage semantics:
    - Re-saving with the same configuration (filename + md_source + library +
      algorithm + chunk_size + overlap) **replaces** the existing rows.
    - :meth:`list_versions` returns all configurations for a document, ordered
      newest first by ``saved_at``.
    """

    # ------------------------------------------------------------------
    # Async internals — called from async router handlers
    # ------------------------------------------------------------------

    async def _save_chunks_async(self, request: SaveChunksRequest) -> SaveChunksResponse:
        stem = _safe_stem(request.filename)
        md_source = md_source_token(request.md_filename, stem)
        normalised = [_normalise_chunk(c) for c in request.chunks]

        async with get_session_factory()() as session:
            # Upsert: delete existing set for the same config, then insert fresh.
            existing = await session.execute(
                select(ChunkSetRecord).where(
                    ChunkSetRecord.filename == request.filename,
                    ChunkSetRecord.md_source == md_source,
                    ChunkSetRecord.chunker_library == request.chunker_library,
                    ChunkSetRecord.chunker_type == request.chunker_type,
                    ChunkSetRecord.chunk_size == request.chunk_size,
                    ChunkSetRecord.chunk_overlap == request.chunk_overlap,
                    ChunkSetRecord.enable_markdown_sizing == request.enable_markdown_sizing,
                )
            )
            existing_set = existing.scalar_one_or_none()
            if existing_set is not None:
                # Cascade-delete old chunk rows, then reuse the set row.
                await session.execute(
                    delete(ChunkRecord).where(ChunkRecord.chunk_set_id == existing_set.id)
                )
                chunk_set = existing_set
                chunk_set.total_chunks = len(normalised)
                chunk_set.saved_at = __import__("datetime").datetime.now(
                    tz=__import__("datetime").timezone.utc
                )
            else:
                chunk_set = ChunkSetRecord(
                    filename=request.filename,
                    md_source=md_source,
                    chunker_type=request.chunker_type,
                    chunker_library=request.chunker_library,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                    enable_markdown_sizing=request.enable_markdown_sizing,
                    total_chunks=len(normalised),
                )
                session.add(chunk_set)

            await session.flush()  # populate chunk_set.id

            # Insert new chunk rows.
            chunk_records = [
                _chunk_item_to_record(c, chunk_set.id) for c in normalised
            ]
            session.add_all(chunk_records)
            await session.commit()

        _db_logger.info(
            "DB: saved %d chunks for '%s' (lib=%s algo=%s)",
            len(normalised), request.filename,
            request.chunker_library, request.chunker_type,
        )
        return SaveChunksResponse(
            success=True,
            message=f"Saved {len(normalised)} chunks for '{request.filename}' to database",
            path=f"db:chunk_sets/{chunk_set.id}",
        )

    async def _load_chunks_async(self, filename: str) -> LoadChunksResponse:
        """Load the most recently saved chunk set for *filename* from the DB."""
        async with get_session_factory()() as session:
            result = await session.execute(
                select(ChunkSetRecord)
                .where(ChunkSetRecord.filename == filename)
                .order_by(ChunkSetRecord.saved_at.desc())
                .limit(1)
            )
            chunk_set = result.scalar_one_or_none()
            if chunk_set is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No saved chunks found for '{filename}'",
                )
            chunks_result = await session.execute(
                select(ChunkRecord)
                .where(ChunkRecord.chunk_set_id == chunk_set.id)
                .order_by(ChunkRecord.index)
            )
            records = chunks_result.scalars().all()
            return LoadChunksResponse(
                chunks=[_record_to_chunk_dict(r) for r in records],
                total_chunks=chunk_set.total_chunks,
                filename=chunk_set.filename,
            )

    async def _load_chunks_by_id_async(
        self, filename: str, chunk_set_id: int
    ) -> LoadChunksResponse:
        async with get_session_factory()() as session:
            chunk_set = await session.get(ChunkSetRecord, chunk_set_id)
            if chunk_set is None or chunk_set.filename != filename:
                raise HTTPException(
                    status_code=404,
                    detail=f"Chunk set id={chunk_set_id} not found for '{filename}'",
                )
            chunks_result = await session.execute(
                select(ChunkRecord)
                .where(ChunkRecord.chunk_set_id == chunk_set_id)
                .order_by(ChunkRecord.index)
            )
            records = chunks_result.scalars().all()
            return LoadChunksResponse(
                chunks=[_record_to_chunk_dict(r) for r in records],
                total_chunks=chunk_set.total_chunks,
                filename=chunk_set.filename,
            )

    async def _list_versions_async(self, filename: str) -> list[ChunksVersion]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(ChunkSetRecord)
                .where(ChunkSetRecord.filename == filename)
                .order_by(ChunkSetRecord.saved_at.desc())
            )
            sets = result.scalars().all()

        stem = _safe_stem(filename)
        versions: list[ChunksVersion] = []
        for s in sets:
            versions.append(ChunksVersion(
                # Use "db:<id>" as a stable opaque filename so the router's
                # /chunks/{chunks_filename} endpoint can round-trip.
                filename=f"db:{s.id}",
                md_filename=_md_filename_for_source(stem, s.md_source),
                md_source=s.md_source,
                library=s.chunker_library or "unknown",
                algorithm=s.chunker_type or "unknown",
                chunk_size=s.chunk_size,
                chunk_overlap=s.chunk_overlap,
                file_path=f"db:chunk_sets/{s.id}",
            ))
        return versions

    # ------------------------------------------------------------------
    # Public async API — called directly by the router (no to_thread wrapper)
    # ------------------------------------------------------------------

    async def save_chunks(self, request: SaveChunksRequest) -> SaveChunksResponse:  # type: ignore[override]
        return await self._save_chunks_async(request)

    async def load_chunks(self, filename: str) -> LoadChunksResponse:  # type: ignore[override]
        return await self._load_chunks_async(filename)

    async def load_chunks_by_filename(  # type: ignore[override]
        self, filename: str, chunks_filename: str
    ) -> LoadChunksResponse:
        # chunks_filename is "db:<id>" for DB-backed sets.
        if chunks_filename.startswith("db:"):
            try:
                chunk_set_id = int(chunks_filename[3:])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid db chunk filename: {chunks_filename}",
                )
            return await self._load_chunks_by_id_async(filename, chunk_set_id)
        # Fallback: treat as a local-file lookup (migration compatibility).
        local = ChunkStorageService()
        return local.load_chunks_by_filename(filename, chunks_filename)

    async def list_versions(self, filename: str) -> list[ChunksVersion]:  # type: ignore[override]
        return await self._list_versions_async(filename)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_chunk_storage() -> "ChunkStorageService | DatabaseChunkStorageService":
    """Return the active storage service based on STORAGE_BACKEND config.

    ``STORAGE_BACKEND=db``    → :class:`DatabaseChunkStorageService` (default)
    ``STORAGE_BACKEND=local`` → :class:`ChunkStorageService` (file-based)
    """
    backend = get_settings().STORAGE_BACKEND.lower()
    if backend == "local":
        return ChunkStorageService()
    return DatabaseChunkStorageService()

