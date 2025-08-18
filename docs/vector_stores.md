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

- `base_vector_store.py`: Abstract base for vector store implementations.
- `__init__.py`: Module exports.

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

#### `add_documents(documents: List[Document])`

Add documents to the vector store.

#### `similarity_search(query: str, k: int = 4) -> List[Document]`

Search the vector store for the top-k most similar documents to the query.

#### `from_documents(documents: List[Document], embedding: Embeddings)`

Create the vector store from a list of documents and an embedding model.

---

## Usage Example

```python
from ragdoll.vector_stores.base_vector_store import BaseVectorStore
# Implement a concrete vector store subclass, then:
store = MyVectorStore()
store.add_documents(docs)
results = store.similarity_search("What is RAG?", k=3)
```

---

## Extending Vector Stores

- Subclass `BaseVectorStore` and implement required methods for new backends.
- Integrate with different embedding models as needed.

---

## Best Practices

- Choose a backend that matches your scale and latency needs.
- Batch add documents for efficiency.
- Regularly update and prune the store for relevance.

---

## Related Modules

- [Embeddings](embeddings.md)
- [Ingestion](ingestion.md)
