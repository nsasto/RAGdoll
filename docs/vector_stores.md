# Vector Stores

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Vector Stores](#extending-vector-stores)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/vector_stores/`

## Purpose

Vector stores manage storage and retrieval of vectorized document chunks. They enable fast similarity search and are a core component of RAG pipelines.

## Key Components

- `base_vector_store.py`: Lightweight wrapper that delegates to any LangChain `VectorStore`.
- `factory.py`: Helpers for instantiating vector stores by type, config, or documents.
- `__init__.py`: Module exports for the wrapper and factory helpers.

## Features

- Pluggable backends (e.g., in-memory, database, cloud).
- Efficient similarity search.
- Integration with embedding and retrieval modules.

---

## How It Works

1. **Adding Documents**: Documents are added to the vector store and embedded as vectors.
2. **Similarity Search**: Queries are embedded and compared to stored vectors to find similar documents.
3. **Backend Flexibility**: Supports different storage backends via subclassing.

---

## Public API and Function Documentation

### `BaseVectorStore`

#### `add_documents(documents: List[Document], batch_size: Optional[int] = None) -> List[str]`

Add documents to the vector store synchronously. Returns list of document IDs.

**Parameters:**

- `documents`: List of documents to add
- `batch_size`: Optional batch size (auto-detected if not provided)

#### `async aadd_documents(documents: List[Document], batch_size: Optional[int] = None) -> List[str]`

Async wrapper for `add_documents`. Uses thread pool for non-blocking execution.

**Parameters:**

- `documents`: List of documents to add
- `batch_size`: Optional batch size (auto-detected if not provided)

#### `async add_documents_parallel(documents: List[Document], *, batch_size: Optional[int] = None, max_concurrent: int = 3, retry_failed: bool = True) -> List[str]`

Add documents with parallel embedding generation for significantly improved performance.

**Performance:** Typically 3-5x faster than sequential processing with remote embedding services.

**Parameters:**

- `documents`: List of documents to add
- `batch_size`: Size of each batch (auto-detected if not provided)
- `max_concurrent`: Maximum number of batches to process concurrently (default: 3)
- `retry_failed`: Whether to retry failed batches (default: True)

**Returns:** List of document IDs maintaining order with input documents

**Example:**

```python
# Using with EmbeddingsConfig max_concurrent setting
from ragdoll.config import Config
from ragdoll.vector_stores import create_vector_store
from ragdoll.embeddings import get_embedding_model

config = Config()
embeddings = get_embedding_model(config_manager=config)
vector_store = create_vector_store("faiss", embedding=embeddings)

# Parallel processing with configured concurrency
max_concurrent = config.embeddings_config.max_concurrent_embeddings
ids = await vector_store.add_documents_parallel(
    documents=chunks,
    max_concurrent=max_concurrent
)
```

#### `similarity_search(query: str, k: int = 4) -> List[Document]`

Search the vector store for the top-k most similar documents to the query.

#### `from_documents(documents: List[Document], embedding: Embeddings)`

Create the vector store from a list of documents and an embedding model.

---

## Usage Example

```python
from ragdoll.vector_stores import create_vector_store
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = create_vector_store(
    "chroma",  # accepts registry keys or dotted class paths
    embedding=embeddings,
    collection_name="ragdoll",
)
store.add_documents(docs)
results = store.similarity_search("What is RAG?", k=3)
```

---

## Plug-and-Play LangChain Stores

`BaseVectorStore` simply delegates to the LangChain implementation you choose. The factory resolves either a short name (e.g., `chroma`, `faiss`, `docarrayinmemory`) or a fully-qualified class path, attaches the embedding model when necessary, and returns a wrapper with a consistent interface.

- `create_vector_store(...)` instantiates an empty `VectorStore`.
- `create_vector_store_from_documents(...)` hydrates a store using LangChain's `from_documents`.
- `vector_store_from_config(...)` wires the `vector_stores` section of `default_config.yaml`.

### Configuration

```yaml
vector_stores:
  enabled: true
  default_store: chroma
  stores:
    chroma:
      params:
        collection_name: ragdoll
        persist_directory: ".ragdoll/chroma"
    faiss:
      distance_strategy: cosine
```

Within `Ragdoll`, this configuration is read via `ConfigManager.vector_store_config` (which normalizes the `vector_stores` block) and turned into a live store:

```python
from ragdoll.config.config_manager import ConfigManager
from ragdoll.vector_stores import vector_store_from_config

config = ConfigManager().vector_store_config
vector_store = vector_store_from_config(config, embedding=my_embeddings)
```

---

## Extending Vector Stores

- Register additional LangChain classes by extending `_VECTOR_STORE_REGISTRY` in `factory.py`, or provide a dotted class path in config.
- Use LangChain's existing ecosystem (Chroma, FAISS, Pinecone, Weaviate, etc.) without writing new adapters.

---

## Best Practices

- Choose a backend that matches your scale and latency needs.
- Batch add documents for efficiency.
- Regularly update and prune the store for relevance.

---

## Related Modules

- [Embeddings](embeddings.md)
- [Ingestion](ingestion.md)
