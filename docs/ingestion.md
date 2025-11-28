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

- **`DocumentLoaderService`** (`ragdoll/ingestion/document_loaders.py`): normalizes inputs (files, URLs, glob patterns) and hands back LangChain `Document` objects.
- **`IngestionPipeline`** (`ragdoll/pipeline/__init__.py`): orchestrates chunking, embeddings, optional entity extraction, vector-store insertion, and graph persistence/retrieval.
- **`IngestionOptions`**: configuration object for toggling vector vs. graph storage, worker counts, chunking overrides, etc.
- **Examples**: `examples/ingestion.ipynb` for a notebook walkthrough and `examples/graph_retriever_example.py` for the async graph retriever workflow.

## Features

- Modular ingestion steps: load, chunk, embed, optional entity extraction.
- Extensible for new document types, chunkers, embedders, storage backends, and graph stores.
- Built-in wiring to `EntityExtractionService`/`GraphPersistenceService` so you can persist graphs or spin up an in-memory graph retriever.
- Error handling, retries, and metrics instrumentation (when enabled in config).

---

## How It Works

1. **Loading**: Documents are loaded from various sources (files, URLs, databases, etc.).
2. **Chunking**: Loaded documents are split into smaller chunks using pluggable chunkers.
3. **Embedding**: Each chunk is converted into a vector representation using the selected embedding model.
4. **Graph creation (optional)**: When `entity_extraction.extract_entities` is enabled, chunks feed into `EntityExtractionService`, which persists the resulting graph via `GraphPersistenceService`.
5. **Graph retrieval (optional)**: If `retriever.graph.enabled` is `true`, the pipeline instantiates a GraphRetriever (NetworkX in-memory, simple, or Neo4j) and exposes it on `IngestionPipeline`.

Each step is modular and can be replaced or extended as needed.

---

## Public API and Function Documentation

### `DocumentLoaderService`

- `ingest_documents(inputs: List[str|Document]) -> List[Dict[str, Any]]`: loads heterogeneous sources (filesystem paths, glob patterns, URLs, pre-built `Document`s) and yields normalized dictionaries with `page_content` + `metadata`.
- Handles batching, threading, caching, and metrics (configured via `ingestion` section in YAML).

### `IngestionPipeline`

```python
pipeline = IngestionPipeline(
    config_manager=my_config,
    content_extraction_service=DocumentLoaderService(...),
    embedding_model=my_embeddings,
    vector_store=my_vector_store,
    options=IngestionOptions(...)
)
stats = await pipeline.ingest(sources)
retriever = pipeline.get_graph_retriever()
graph = pipeline.last_graph
```

- `IngestionOptions` toggles vector-store writes, graph-store writes, parallel entity extraction, per-component overrides, and LLM caller injection.
- After `ingest()`, the pipeline exposes:
  - `stats`: document/chunk counts, vector/graph entries, errors, whether a graph retriever is available.
  - `last_graph`: the most recent `Graph` object returned by `EntityExtractionService`.
  - `get_graph_retriever()`: returns the configured GraphRetriever (simple in-memory, NetworkX, or Neo4j).

### `ingest_from_vector_store`

```python
from ragdoll.pipeline import ingest_from_vector_store

result = await ingest_from_vector_store(
    vector_store=existing_vector_store,
    graph_store=graph_store,
    entity_service=entity_service
)
graph_retriever = result["graph_retriever"]  # Pre-configured with vector store
```

This helper function builds a knowledge graph directly from an existing vector store:

- Extracts documents from the vector store
- Creates a graph that references the same vector IDs
- Returns a GraphRetriever pre-configured with both graph store and vector store
- Useful for adding graph capabilities to existing vector stores without re-ingesting

---

## Usage Example

```python
import asyncio
from ragdoll.pipeline import IngestionPipeline, IngestionOptions
from ragdoll import settings

async def run():
    pipeline = IngestionPipeline(
        config_manager=settings.get_config_manager(),
        options=IngestionOptions(
            skip_vector_store=True,
            entity_extraction_options={
                "config": {"graph_retriever": {"enabled": True, "backend": "simple"}}
            },
        ),
    )
    stats = await pipeline.ingest(["data/manual.pdf"])
    print(stats)
    retriever = pipeline.get_graph_retriever()
    if retriever:
        print(retriever.invoke("Who collaborated with Ada Lovelace?"))

asyncio.run(run())
```

See `examples/ingestion.ipynb` for a full notebook and `examples/graph_retriever_example.py` for the dedicated graph workflow.

---

## Hybrid retrieval (vector + graph)

If you ingest with graph extraction enabled, the `Ragdoll` entry point now exposes a lightweight hybrid retriever that merges vector hits with graph nodes:

```python
result = ragdoll.query_hybrid("Who works at Contoso?")
# or: result = ragdoll.query("Who works at Contoso?", use_hybrid=True)
for doc in result["documents"]:
    print(doc.metadata.get("source_kind"), doc.page_content)
```

Defaults stay fast (small `top_k`, shallow hop expansion, no LLM in the retriever). Tuning knobs live on `RagdollRetriever`: `top_k_vector`, `top_k_graph`, `graph_hops`, and weightings for vector vs. graph scores.

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
