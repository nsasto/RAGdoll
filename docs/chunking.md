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

- `chunker.py`: Implements chunking logic.
- `__init__.py`: Exposes chunker functionality.
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

### `Chunker`

#### Constructor

```python
Chunker(config: Optional[dict] = None, text_splitter: Optional[TextSplitter] = None)
```

Initializes the chunker with a config and optional text splitter. If no config is provided, loads the default config.

#### `from_config()`

Class method to instantiate a chunker from the default configuration.

#### `get_text_splitter() -> TextSplitter`

Returns a LangChain text splitter object, configured from `default_config.yaml`.

**Splitter types:**

- `markdown`: Uses header-based splitting.
- `recursive`: Uses recursive character splitting with chunk size and overlap.

---

## Usage Example

```python
from ragdoll.chunkers.chunker import Chunker
chunker = Chunker.from_config()
splitter = chunker.get_text_splitter()
chunks = splitter.split_text("Your document text here.")
```

---

## Extending Chunking

- Implement new chunkers by subclassing `Chunker` and overriding methods as needed.
- Add new splitter types by extending the logic in `get_text_splitter()`.

---

## Best Practices

- Tune chunk size and overlap for your use case (e.g., retrieval, summarization).
- Use markdown splitting for structured documents, recursive for plain text.
- Validate chunk boundaries for downstream model compatibility.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Embeddings](embeddings.md)
