# Embeddings

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Embeddings](#extending-embeddings)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/embeddings/`, `examples/embeddings.ipynb`

## Purpose

Embeddings convert text chunks into vector representations for similarity search and retrieval. This enables semantic search, clustering, and other downstream tasks in RAG pipelines.

## Key Components

- `embeddings.py`: Core embedding logic.
- `__init__.py`: Module exports.
- Example: `examples/embeddings.ipynb` shows embedding usage.

## Features

- Pluggable embedding models (OpenAI, HuggingFace, etc.).
- Batch processing support.
- Integration with vector stores.

---

## How It Works

1. **Initialization**: The embeddings class is initialized with a configuration and optional embedding model.
2. **Model Selection**: Selects the embedding backend (OpenAI, HuggingFace) based on config.
3. **Embedding Generation**: Converts text or documents into vectors for storage and retrieval.

---

## Public API and Function Documentation

### `RagdollEmbeddings`

#### Constructor

```python
RagdollEmbeddings(config: Optional[dict] = None, embeddings_model: Optional[OpenAIEmbeddings] = None)
```

Initializes the embeddings class with a config and optional model. If no config is provided, loads the default config.

#### `from_config()`

Class method to instantiate from the default configuration.

#### `get_embeddings_model() -> OpenAIEmbeddings | HuggingFaceEmbeddings`

Returns the configured embedding model instance.

#### `_create_openai_embeddings(model_params: Dict[str, Any]) -> OpenAIEmbeddings`

Creates an OpenAIEmbeddings model with parameters from config.

#### `_create_huggingface_embeddings(model_params: Dict[str, Any]) -> HuggingFaceEmbeddings`

Creates a HuggingFaceEmbeddings model with parameters from config.

---

## Usage Example

```python
from ragdoll.embeddings.embeddings import RagdollEmbeddings
emb = RagdollEmbeddings.from_config()
model = emb.get_embeddings_model()
vectors = model.embed_documents(["text1", "text2"])
```

---

## Extending Embeddings

- Implement new embedding backends by adding methods to `RagdollEmbeddings`.
- Add new model types and configuration options as needed.

---

## Best Practices

- Choose embedding models that match your data and use case.
- Use batch processing for efficiency.
- Store and reuse embeddings to avoid redundant computation.

---

## Related Modules

- [Chunking](chunking.md)
- [Vector Stores](vector_stores.md)
