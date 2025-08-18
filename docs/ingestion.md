# Ingestion

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Usage Example](#usage-example)
- [Extending Ingestion](#extending-ingestion)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/ingestion/`, `examples/ingestion.ipynb`

## Purpose

The ingestion module orchestrates the end-to-end pipeline for document processing. It is responsible for loading raw documents, chunking them into manageable pieces, generating embeddings, and storing the results in vector stores or other backends. This modular approach enables flexible, scalable, and reproducible data pipelines for retrieval-augmented generation (RAG) and other downstream tasks.

## Key Components

- **`base_ingestion_service.py`**: Defines the abstract base class for ingestion services, specifying the required interface and extensibility points.
- **`ingestion_service.py`**: Provides a concrete implementation of the ingestion pipeline, integrating loading, chunking, embedding, and storage.
- **Example**: `examples/ingestion.ipynb` demonstrates how to use and customize the ingestion pipeline in practice.
  ingestor.run()

## Features

- Modular ingestion steps: load, chunk, embed, store.
- Extensible for new document types, chunkers, embedders, and storage backends.
- Error handling and logging for robust pipeline execution.
- Integration with configuration and metrics modules.

---

## How It Works

1. **Loading**: Documents are loaded from various sources (files, URLs, databases, etc.).
2. **Chunking**: Loaded documents are split into smaller chunks using pluggable chunkers.
3. **Embedding**: Each chunk is converted into a vector representation using the selected embedding model.
4. **Storing**: Embeddings and metadata are stored in a vector store or other backend for later retrieval.

Each step is modular and can be replaced or extended as needed.

---

## Public API and Function Documentation

### `BaseIngestionService`

#### `ingest_documents(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]`

Abstract method to ingest documents from various sources concurrently. Each source is a dictionary with keys like `type` (e.g., "website", "pdf") and `identifier` (e.g., a URL, a file path). Returns a list of documents with metadata.

---

### `IngestionService`

#### Constructor

```python
IngestionService(
    config_path: Optional[str] = None,
    custom_loaders: Optional[Dict[str, Any]] = None,
    max_threads: Optional[int] = None,
    batch_size: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
    metrics_manager: Optional[MetricsManager] = None,
    use_cache: bool = True,
    collect_metrics: bool = False
)
```

Initializes the ingestion service with configuration, custom loaders, threading, caching, and metrics options.

#### `ingest_documents(inputs: List[str]) -> List[Dict[str, Any]]`

Ingests documents from a list of input sources (file paths, URLs, or glob patterns). Handles batching, threading, caching, and metrics. Returns a list of loaded documents with metadata.

**Example:**

```python
from ragdoll.ingestion.ingestion_service import IngestionService
ingestor = IngestionService(config_path="config.yaml")
docs = ingestor.ingest_documents(["data/*.pdf", "https://arxiv.org/abs/1234.5678"])
```

#### `clear_cache(source_type: Optional[str] = None, identifier: Optional[str] = None) -> int`

Clears cached data for a given source type and/or identifier. Returns the number of cache entries cleared.

#### `get_metrics(days: int = 30) -> Dict[str, Any]`

Returns recent and aggregate ingestion metrics for the last `days` days. Useful for monitoring pipeline performance.

---

### Utility Functions

#### `create_empty_file(filepath)`

Creates an empty file at the specified path. Example:

```python
from ragdoll.ingestion import create_empty_file
create_empty_file("output/empty.txt")
```

---

## Usage Example

```python
from ragdoll.ingestion.ingestion_service import IngestionService

ingestor = IngestionService(config_path="config.yaml")
docs = ingestor.ingest_documents(["data/*.pdf", "https://arxiv.org/abs/1234.5678"])
print(f"Loaded {len(docs)} documents")
```

See `examples/ingestion.ipynb` for a step-by-step notebook demonstration.

---

## Extending Ingestion

- **Custom Loaders**: Implement new document loaders by subclassing the loader interface and registering them via `custom_loaders` or configuration.
- **Custom Chunkers**: Add new chunking strategies by implementing the chunker interface.
- **Custom Embedders**: Integrate new embedding models as needed.
- **Custom Storage**: Support new storage backends by extending the vector store interface.

---

## Best Practices

- Use configuration files to manage pipeline settings and ensure reproducibility.
- Log all steps and errors for easier debugging and monitoring.
- Write tests for custom components to ensure reliability.
- Use modular design to swap or upgrade pipeline steps without breaking the workflow.

---

## Related Modules

- [Chunking](chunking.md)
- [Embeddings](embeddings.md)
- [Vector Stores](vector_stores.md)
- [Configuration](configuration.md)
