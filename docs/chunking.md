# Chunking

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Chunking](#extending-chunking)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/chunkers/`, `examples/chunker.ipynb`

## Purpose

Chunking is the process of splitting documents into manageable pieces for downstream processing (embedding, retrieval, etc.). This enables efficient storage, retrieval, and processing in RAG pipelines.

## Key Components

- `ragdoll/chunkers/__init__.py`: exposes `get_text_splitter` and `split_documents`.
- Example: `examples/chunker.ipynb` demonstrates chunking workflows.

## Features

- Customizable chunking strategies.
- Handles various document types and sizes.
- Integrates with ingestion and embedding pipelines.

---

## How It Works

1. **Initialization**: The chunker is initialized with a configuration and an optional text splitter.
2. **Text Splitting**: Uses LangChain's text splitters (recursive, markdown, etc.) to divide documents into chunks.
3. **Chunking Parameters**: Chunk size, overlap, and splitter type are configurable via YAML or code.

---

## Public API and Function Documentation

### `get_text_splitter`

```python
from ragdoll.chunkers import get_text_splitter

splitter = get_text_splitter(
    splitter_type="markdown",
    chunk_size=1500,
    chunk_overlap=200,
)
```

- Reads configuration from an explicit `config_manager`, supplied `config` dict, or the global `settings.get_app()` fallback.
- Returns a LangChain splitter (Markdown, RecursiveCharacter, etc.) configured with chunk size/overlap/separators.

### `split_documents`

```python
from ragdoll.chunkers import split_documents
from langchain_core.documents import Document

chunks = split_documents(
    documents=[Document(page_content="hello", metadata={"source": "demo"})],
    splitter=text_splitter,
    batch_size=50,
)
```

- Accepts LangChain `Document` objects and either an explicit splitter instance or the same keyword overrides as `get_text_splitter`.
- Cleans up nested/invalid `Document` payloads before splitting to avoid downstream crashes.

**Splitter types:**

- `markdown`: Uses header-based splitting.
- `recursive`: Uses recursive character splitting with chunk size and overlap.

---

## Usage Example

```python
from ragdoll.chunkers import get_text_splitter, split_documents
from langchain_core.documents import Document

splitter = get_text_splitter(splitter_type="recursive", chunk_size=1000, chunk_overlap=200)
docs = [Document(page_content="Your document text here.", metadata={})]
chunks = split_documents(docs, splitter=splitter)
```

---

## Extending Chunking

- Register additional LangChain splitters or custom ones by extending `get_text_splitter`.
- Wrap non-Document payloads before calling `split_documents` if you need bespoke preprocessing.

---

## Best Practices

- Tune chunk size and overlap for your use case (e.g., retrieval, summarization).
- Use markdown splitting for structured documents, recursive for plain text.
- Validate chunk boundaries for downstream model compatibility.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Embeddings](embeddings.md)
