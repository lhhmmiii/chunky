"""
Text splitters backed by LangChain text-splitters.

Supports four strategies: ``token``, ``recursive``, ``character``, ``markdown``.

Install:
    pip install langchain-text-splitters tiktoken
"""

from __future__ import annotations

from typing import Callable

from fastapi import HTTPException
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from backend.models.schemas import ChunkItem, ChunkRequest, ChunkerType
from backend.registry import register_chunker
from .base import TextChunker

# Headers recognised by the Markdown splitter (H1 → H3).
_MARKDOWN_HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

_LIB = "langchain"
_LIB_LABEL = "LangChain"


class LangChainChunker(TextChunker):
    """Text splitter that delegates to LangChain's text-splitting utilities.

    Strategy is chosen at call time via :attr:`ChunkRequest.chunker_type`.

    Strategies
    ----------
    token
        :class:`~langchain_text_splitters.TokenTextSplitter` — splits on token
        boundaries using tiktoken; ideal for LLM context-window management.
    recursive
        :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter` —
        tries paragraph, sentence, word boundaries in order.
    character
        :class:`~langchain_text_splitters.CharacterTextSplitter` — splits on
        ``\\n\\n`` paragraphs, then falls back to ``chunk_size`` characters.
    markdown
        Two-phase split: headers via
        :class:`~langchain_text_splitters.MarkdownHeaderTextSplitter`, then
        optional size cap via
        :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
        (activated by ``enable_markdown_sizing``).
    """

    def chunk(self, request: ChunkRequest) -> list[ChunkItem]:
        handler = self._DISPATCH.get(request.chunker_type)
        if handler is None:
            raise HTTPException(
                status_code=400,
                detail=f"LangChainChunker does not support chunker_type='{request.chunker_type}'",
            )
        return handler(self, request)

    # ------------------------------------------------------------------
    # Private strategy methods
    # ------------------------------------------------------------------

    @register_chunker(
        library=_LIB, library_label=_LIB_LABEL,
        strategy="token", label="Token",
        description="Splits on token boundaries via tiktoken. Ideal for LLM context-window management.",
    )
    def _split_token(self, request: ChunkRequest) -> list[ChunkItem]:
        splitter = TokenTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        splits = splitter.split_text(request.content)
        # chunk_overlap is in *tokens*; build_chunks needs *characters*.
        # Measure the actual character overlap from the first adjacent pair
        # rather than estimating a token→char ratio.
        char_overlap = self.measure_char_overlap(splits)
        return self.build_chunks(request.content, splits, char_overlap)

    @register_chunker(
        library=_LIB, library_label=_LIB_LABEL,
        strategy="recursive", label="Recursive",
        description="Tries paragraph → sentence → word boundaries in order.",
    )
    def _split_recursive(self, request: ChunkRequest) -> list[ChunkItem]:
        # Use markdown-aware separators: headings, fences, horizontal rules,
        # blank lines, newlines — in that priority order.  chunk_size and
        # chunk_overlap are in characters (length_function=len is the default).
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return self.build_chunks(
            request.content,
            splitter.split_text(request.content),
            request.chunk_overlap,
        )

    @register_chunker(
        library=_LIB, library_label=_LIB_LABEL,
        strategy="character", label="Character",
        description="Splits on \\n\\n paragraphs, falls back to chunk_size characters.",
    )
    def _split_character(self, request: ChunkRequest) -> list[ChunkItem]:
        splitter = CharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separator="\n\n",
        )
        return self.build_chunks(
            request.content,
            splitter.split_text(request.content),
            request.chunk_overlap,
        )

    @register_chunker(
        library=_LIB, library_label=_LIB_LABEL,
        strategy="markdown", label="Markdown",
        description=(
            "Two-phase split: H1/H2/H3 headers first, then optional size cap "
            "via RecursiveCharacterTextSplitter (enable_markdown_sizing)."
        ),
    )
    def _split_markdown(self, request: ChunkRequest) -> list[ChunkItem]:
        """Two-phase Markdown splitting.

        Phase 1 — split on H1/H2/H3 headers via
        :class:`~langchain_text_splitters.MarkdownHeaderTextSplitter`.

        Phase 2 (optional) — apply a secondary
        :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
        to cap each section at ``chunk_size`` characters when
        ``enable_markdown_sizing`` is *True*.
        """
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS,
            strip_headers=False,
        )
        docs = md_splitter.split_text(request.content)

        if request.enable_markdown_sizing:
            # Use the same markdown-aware separators as the recursive strategy
            # so that sub-splits also respect markdown structure.
            secondary = RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )
            docs = secondary.split_documents(docs)

        # MarkdownHeaderTextSplitter preserves header text in page_content
        # (strip_headers=False), so each page_content is a verbatim substring
        # of the original — build_chunks can locate real character offsets.
        # Phase-1 splits have no overlap; phase-2 inherits request.chunk_overlap.
        char_overlap = request.chunk_overlap if request.enable_markdown_sizing else 0
        chunks = self.build_chunks(
            request.content,
            [doc.page_content for doc in docs],
            char_overlap,
        )
        # Preserve header metadata produced by MarkdownHeaderTextSplitter.
        for chunk, doc in zip(chunks, docs):
            chunk.metadata = doc.metadata
        return chunks

    @register_chunker(
        library=_LIB, library_label=_LIB_LABEL,
        strategy="parent_child", label="Parent-Child",
        description=(
            "Two-level split: text is first divided into large parent chunks "
            "(parent_chunk_size, default 3× chunk_size), then each parent is "
            "sub-split into smaller child chunks (chunk_size). Each child carries "
            "parent_id and parent_content as metadata for retrieval context."
        ),
    )
    def _split_parent_child(self, request: ChunkRequest) -> list[ChunkItem]:
        """Two-level parent-child split.

        Phase 1 — split the full text into parent chunks at ``parent_chunk_size``
        (defaults to 3× ``chunk_size`` when not set on the request).

        Phase 2 — split each parent chunk into child chunks at ``chunk_size``
        using the same overlap as the request.

        Every returned :class:`ChunkItem` represents a **child** chunk and has
        its ``parent_id`` and ``parent_content`` populated.
        """
        parent_size = request.parent_chunk_size or (request.chunk_size * 3)

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=0,  # parents do not overlap each other
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        parent_splits = parent_splitter.split_text(request.content)

        all_children: list[ChunkItem] = []
        global_child_index = 0

        for parent_id, parent_text in enumerate(parent_splits):
            child_splits = child_splitter.split_text(parent_text)

            # Map child text fragments back to character positions *within the
            # parent text* first, then add the parent's own offset in the
            # original document.
            parent_start = request.content.find(parent_text)
            if parent_start == -1:
                parent_start = 0

            child_search_start = 0
            for child_text in child_splits:
                # Find the child's position inside the parent text.
                pos_in_parent = parent_text.find(child_text, child_search_start)
                if pos_in_parent == -1:
                    pos_in_parent = child_search_start
                child_start = parent_start + pos_in_parent
                child_end = child_start + len(child_text)
                child_search_start = max(0, pos_in_parent + len(child_text) - request.chunk_overlap)

                all_children.append(
                    ChunkItem(
                        index=global_child_index,
                        content=child_text,
                        start=child_start,
                        end=child_end,
                        parent_id=parent_id,
                        parent_content=parent_text,
                        metadata={"parent_id": parent_id},
                    )
                )
                global_child_index += 1

        return all_children

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    _DISPATCH: dict[ChunkerType, Callable[[LangChainChunker, ChunkRequest], list[ChunkItem]]] = {
        ChunkerType.token: _split_token,
        ChunkerType.recursive: _split_recursive,
        ChunkerType.character: _split_character,
        ChunkerType.markdown: _split_markdown,
        ChunkerType.parent_child: _split_parent_child,
    }