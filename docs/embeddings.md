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

- `ragdoll/embeddings/__init__.py`: Contains `get_embedding_model` plus provider-specific helpers.
- Example: `examples/embeddings.ipynb` shows embedding usage.

## Features

- Pluggable embedding models (OpenAI, HuggingFace, etc.).
- Batch processing support.
- Integration with vector stores.

---

## How It Works

1. **Configuration**: `ConfigManager` loads the `embeddings` block from `default_config.yaml` (or your custom config).
2. **Model Selection**: `get_embedding_model` chooses a provider based on the `default_model` entry or an explicit `model_name`.
3. **Instantiation**: Provider-specific helpers (OpenAI, HuggingFace, Google Vertex, Cohere, Fake) return a LangChain `Embeddings` object.
4. **Usage**: The returned object exposes LangChain’s standard `embed_documents`/`embed_query` methods.

---

## Public API and Function Documentation

### `get_embedding_model(model_name: Optional[str] = None, config_manager: Optional[ConfigManager] = None, provider: Optional[str] = None, **kwargs)`

Return a LangChain `Embeddings` instance based on configuration or explicit parameters.

- `model_name`: Optional alias that maps to the `embeddings.models` config section.
- `config_manager`: Provide your own `ConfigManager` instance; otherwise the default configuration is loaded.
- `provider`: Skip config lookup and directly specify `"openai"`, `"huggingface"`, `"google"`, `"cohere"`, or `"fake"`.
- `**kwargs`: Extra keyword arguments forwarded to the provider helper (and ultimately to the LangChain class).

Helper functions such as `_create_openai_embeddings`, `_create_huggingface_embeddings`, `_create_google_embeddings`, `_create_cohere_embeddings`, and `_create_fake_embeddings` encapsulate the provider-specific constructor logic. They can be reused when extending the module.

---

## Usage Example

```python
from ragdoll.embeddings import get_embedding_model
model = get_embedding_model()  # Uses the default model defined in config
vectors = model.embed_documents(["text1", "text2"])
```

---

## Extending Embeddings

- Add a new provider helper (e.g., `_create_bespoke_embeddings`) inside `ragdoll/embeddings/__init__.py`.
- Register the provider in `_initialize_model_by_provider` and update `default_config.yaml` with the necessary parameters.
- Reference the new provider via the `embeddings.models` config block.

---

## Best Practices

- Choose embedding models that match your data and use case.
- Use batch processing for efficiency.
- Store and reuse embeddings to avoid redundant computation.

---

## Related Modules

- [Chunking](chunking.md)
- [Vector Stores](vector_stores.md)
