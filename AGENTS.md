# AGENTS.md

## Overview

Chunky is a local, open-source workspace for preparing documents for Retrieval-Augmented Generation (RAG).

The system consists of:

* **Frontend:** React application for document inspection and workflow orchestration
* **Backend:** FastAPI service exposing conversion, chunking, enrichment, and persistence APIs
* **Storage:** Local filesystem for PDFs, Markdown files, summaries, and saved chunk sets
* **Extension system:** Decorator-based registries for PDF converters and chunking strategies

The primary goal is **inspectable RAG preprocessing**: convert → clean → chunk → enrich → inspect before indexing.

---

## Architecture

### Frontend

Responsibilities:

* Upload and manage PDFs
* Compare PDF, Markdown, and chunk outputs side-by-side
* Configure converters, chunkers, and enrichment
* Browse saved chunk versions
* Batch operations across multiple documents

The frontend discovers available converters and chunkers dynamically through:

```
GET /api/capabilities
```

Do **not** hardcode converter or chunker options in the UI.

---

### Backend

Main responsibilities:

1. PDF → Markdown conversion
2. Markdown cleanup and enrichment
3. Chunk generation
4. Chunk enrichment
5. Persisting summaries and chunk sets
6. Exposing capabilities to the frontend

High-level pipeline:

```text
PDF
 ↓
Converter
 ↓
Markdown
 ↓
Markdown enrichment
 ↓
Chunker
 ↓
Chunks
 ↓
Chunk enrichment
 ↓
Saved chunk set
```

---

## Repository Conventions

### Converters

Location:

```text
backend/converters/
```

Every converter:

* inherits from `PDFConverter`
* implements:

```python
convert(pdf_path: Path) -> str
```

* must validate file existence
* should lazily import heavyweight dependencies inside `__init__`
* should raise clear exceptions for unsupported documents

Example:

```python
@register_converter(
    name="my_converter",
    label="My Converter",
    description="Shown in UI",
)
class MyConverter(PDFConverter):
    ...
```

### Important

Adding a converter requires:

1. Creating a module under:

```text
backend/converters/
```

2. Decorating the class with:

```python
@register_converter(...)
```

3. Importing the module in:

```text
capabilities_router.py
```

The import is intentionally for side effects.

---

## Chunkers

Chunkers are registered via decorators.

A chunker registration:

```python
@register_chunker(
    library="my_lib",
    library_label="My Library",
    strategy="my_strategy",
    label="My Strategy",
    description="Shown in UI",
)
```

Chunkers should:

* be deterministic unless explicitly semantic/LLM-based
* return `list[ChunkItem]`
* preserve original ordering
* respect `chunk_size`
* respect `chunk_overlap` when supported
* document unsupported parameters clearly

If overlap is unsupported, fail explicitly or ignore with a documented warning.

---

## Capabilities Endpoint

The frontend relies on:

```text
GET /api/capabilities
```

Any new converter or chunker must appear automatically through the registry.

Agents should avoid:

* hardcoding frontend enums
* duplicating capability metadata
* maintaining parallel configuration lists

The registry is the source of truth.

---

## Markdown Enrichment

Markdown enrichment occurs **before chunking**.

Pipeline:

1. Regex cleanup
2. Optional document summary generation
3. LLM cleanup and restructuring

Guidelines:

* Preserve semantic content.
* Prefer minimal edits.
* Fix malformed tables and headers.
* Avoid hallucinating content.
* Document summaries should remain concise and factual.

---

## Chunk Enrichment

Chunk enrichment occurs **after chunking**.

Generated fields:

* `cleaned_chunk`
* `title`
* `context`
* `summary`
* `keywords`
* `questions`

Additional context may include:

* document summary
* preceding Markdown window
* following Markdown window

Guidelines:

### Title

* Short and descriptive.
* Avoid repeating section numbers.

### Context

* Describe where the chunk fits in the document.
* One sentence.

### Summary

* Summarize the chunk itself.
* One sentence.

### Keywords

* Prefer domain terms.
* Avoid generic words.

### Questions

Generate questions that:

* resemble real user queries
* can be answered directly from the chunk
* vary in phrasing

Do not generate questions requiring information outside the chunk.

---

## Persistence

Chunk sets are stored by:

* Markdown source
* Chunker library
* Chunker strategy
* Chunk size
* Chunk overlap

Persistence should be:

* deterministic
* reproducible
* independent of timestamps when possible

Agents should avoid introducing nondeterministic naming schemes.

---

## Dependency Guidelines

Prefer:

* lazy imports for heavyweight ML models
* explicit optional dependencies
* graceful degradation when dependencies are unavailable

Avoid:

* importing large ML models at module import time
* global initialization with side effects
* network calls during application startup

Models that download artifacts on first use should:

* cache locally
* report progress when possible
* fail with actionable errors

---

## API Design Principles

Backend APIs should:

* return structured Pydantic models
* provide clear error messages
* remain backward compatible where practical
* avoid exposing internal implementation details

Prefer:

```text
422 -> invalid request
404 -> missing resource
500 -> unexpected server error
```

with informative error payloads.

---

## Coding Guidelines

### General

* Prefer explicit over implicit behavior.
* Keep functions focused and small.
* Use type hints everywhere.
* Avoid hidden global state.
* Prefer composition over inheritance except for extension interfaces.

### Logging

Log:

* converter selection
* chunker selection
* enrichment progress
* model initialization
* persistence operations

Do not log:

* API keys
* full document contents
* sensitive user data

### Errors

Raise:

* actionable exceptions
* descriptive messages

Avoid:

* swallowing exceptions silently
* broad `except Exception:` unless re-raising with context

---

## Adding New Features

When adding a new feature:

1. Keep the PDF → Markdown → Chunk → Enrich pipeline explicit.
2. Preserve inspectability in the UI.
3. Expose capabilities through the registry when applicable.
4. Prefer extension points over frontend conditionals.
5. Keep outputs reproducible and easy to debug.

The guiding principle of Chunky is:

> RAG preprocessing should be observable, configurable, and reproducible rather than hidden behind a single "ingest" button.
