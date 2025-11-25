![Ragdoll](img/github-header-image.png)

[![CI](https://github.com/nsasto/RAGdoll/actions/workflows/ci.yml/badge.svg)](https://github.com/nsasto/RAGdoll/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](https://github.com/nsasto/RAGdoll/releases)
[![Stable](https://badge.fury.io/py/python-ragdoll.svg)](https://pypi.org/project/python-ragdoll/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# RAGdoll: A Flexible and Extensible RAG Framework

Welcome to Ragdoll 2.0! This release marks a significant overhaul of the Ragdoll project, focusing on enhanced flexibility, extensibility, and maintainability. We've completely refactored the core architecture to make it easier than ever to adapt Ragdoll to your specific needs and integrate it with the broader LangChain ecosystem. This document outlines the major changes and improvements you'll find in this new version.

# 🧭 Project Overview

RAGdoll 2 is an extensible framework for building Retrieval-Augmented Generation (RAG) applications. It provides a modular architecture that allows you to easily integrate various data sources, chunking strategies, embedding models, vector stores, large language models (LLMs), and graph stores. RAGdoll is designed to be flexible and fast while relying solely on open-source third-party libraries (LangChain, Chroma, spaCy, etc.). It's also designed to accomodate a broad array of file types without any initial dependency on third party hosted services using [langchain-markitdown](https://github.com/nsasto/langchain-markitdown). The loaders can easily be swapped out with any compatible lanchain loader when ready for production.

Note that RAGdoll 2 is a complete overhaul of the initial RAGdoll project and is not backwards compatible in any respect.

## How RAGdoll compares to GraphRAG-style tools

RAGdoll started as a learning project and has grown into a modular orchestrator. I've tried to focus the detail below on what is actually shipped in this repo (no performance claims or benchmarks are published yet) to help position RAGdoll in the broader landscape.

- **Scope:** RAGdoll orchestrates loaders, chunkers, embeddings, vector stores, LLMs, and an optional graph layer. The GraphRAG family (GraphRAG, NanoGraphRAG, Fast GraphRAG) is primarily graph-forward; RAGdoll adds the rest of the RAG plumbing and a demo UI.
- **Graph building:** Entities come from spaCy NER; relations come from prompt-based LLM extraction per chunk (configurable via YAML prompts/parsers). It stores a flat graph (JSON/NetworkX/Neo4j) and exposes a retriever—there is no community detection or hierarchical summarization step.
- **Retrieval:** The default `query` path is vector-only. When you call `ingest_with_graph`, you can also obtain a graph retriever (simple in-memory or Neo4j) and combine it with vector search yourself for hybrid flows; no automatic community summaries are generated. A lightweight hybrid retriever (`query(..., use_hybrid=True)` or `query_hybrid`) now merges vector hits with graph nodes for a fast graph-aware context path.
- **Config and runtime:** Everything is wired through YAML and LangChain abstractions. Defaults point at OpenAI models, but you can swap in local embeddings/LLMs and different stores (Chroma/FAISS/Neo4j) without code changes. Caching and monitoring are built in but optional.
- **Benchmarks:** Cost/speed/quality numbers depend entirely on the models and stores you pick; we have not published comparisons against GraphRAG variants, so avoid quoting figures until you run your own measurements.

## What's New

### Enhanced Features in RAGdoll 2.1

This version of RAGdoll introduces significant performance and architectural improvements:

- **Parallel Execution (NEW in 2.1):** Concurrent processing for embeddings and entity extraction with configurable rate limiting. Achieves 5-8x faster pipeline execution for typical workloads.
- **Embedding-based Graph Retrieval (NEW in 2.1):** GraphRetriever now supports embedding-based seed node selection using vector store integration, dramatically improving retrieval accuracy over fuzzy text matching.
- **Vector ID Linkage (NEW in 2.1):** Proper linking between graph nodes and vector embeddings ensures seamless hybrid retrieval without orphaned nodes.
- **Modular Retrieval Architecture (NEW in 2.1):** Clean separation between VectorRetriever, GraphRetriever, and HybridRetriever with multiple combination strategies.
- **Caching:** Store and reuse results from previous operations to avoid redundant computations.
- **Auto Loader Selection:** Includes loaders for multiple file types with Langchain-Markitdown as default, configurable to any LangChain-compatible loader.
- **Monitoring:** Track and understand the performance and behavior of your RAG applications over time.

```yaml
# Enable monitoring in config
monitor:
  enabled: true
```

## Quick Start Guide

Here's a quick example of how to get started with RAGdoll using the new LLM caller abstraction:

```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.llms import get_llm_caller

# Resolve whichever model is marked as default in config (or pass a model name).
llm_caller = get_llm_caller()

# Spin up the orchestrator with sensible defaults.
ragdoll = Ragdoll(llm_caller=llm_caller)

# Ingest a few local files (vector store + caches handled automatically).
ragdoll.ingest_data(["path/to/document.md", "path/to/notes.pdf"])

# Run a retrieval + answer round trip.
result = ragdoll.query("What is the capital of France?")
print(result["answer"])
```

Need finer control over loaders or paths? Use `settings.get_app()` (or `bootstrap_app` with overrides) to obtain the shared `AppConfig`, tweak its `config`, and pass component overrides into `Ragdoll`.

## Demo Application

RAGdoll includes an interactive web-based demo application that showcases its capabilities. The demo provides a user-friendly interface to:

- Configure RAGdoll settings using a YAML editor
- Ingest documents from various sources (files, URLs, text)
- Explore the ingestion pipeline and data transformations
- Query the RAG system and view retrieval traces
- Monitor performance metrics and caching

### Running the Demo

To run the demo application:

```bash
# Ensure dependencies are installed
pip install -e .[all]

# Start the demo server
uvicorn demo_app.main:app --reload
```

Then open your browser to `http://localhost:8000` to access the demo interface.

The demo uses FastAPI for the backend, HTMX and Alpine.js for dynamic interactions, and Tailwind CSS for styling, providing a modern, responsive experience.

## Performance & Parallel Execution

RAGdoll 2.1+ includes comprehensive parallel execution optimizations for significantly faster ingestion and entity extraction:

### Parallel Embeddings (Vector Store Layer)

**BREAKING CHANGE in 2.1:** Parallel embedding logic moved from `IngestionPipeline` to `BaseVectorStore` for better separation of concerns and reusability.

- **Concurrent batch processing**: Processes multiple embedding batches simultaneously via `add_documents_parallel()`
- **Configuration**: Set `max_concurrent_embeddings` in `EmbeddingsConfig` (YAML: `embeddings.max_concurrent_embeddings`, default: 3)
- **Automatic batching**: Intelligently splits documents into batches for optimal throughput
- **Performance gain**: 3-5x faster embedding creation compared to sequential processing
- **Retry logic**: Automatically retries failed batches sequentially for robustness

**New Async API:**

```python
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
    batch_size=10,
    max_concurrent=max_concurrent
)
```

### Parallel Entity Extraction

- **Concurrent LLM calls**: Processes multiple documents simultaneously with rate limiting (configurable via `max_concurrent_llm_calls`, default: 8)
- **Automatic parallelization**: Enabled by default for document sets with 4+ documents
- **Smart fallback**: Uses sequential processing for small batches to avoid overhead
- **Performance gain**: 5-10x faster entity extraction (limited by API rate limits)

### Configuration Example

```yaml
# In default_config.yaml or app config
embeddings:
  default_model: openai
  max_concurrent_embeddings: 5 # NEW: Controls parallel embedding batches
  models:
    openai:
      provider: openai
      model: text-embedding-3-large
```

```python
from ragdoll.pipeline import IngestionOptions

# High-speed processing (good API limits)
options = IngestionOptions(
    batch_size=20,
    max_concurrent_llm_calls=15,  # Note: max_concurrent_embeddings removed from here
    parallel_extraction=True  # Enabled by default
)

# Conservative (rate limit sensitive)
options = IngestionOptions(
    batch_size=10,
    max_concurrent_llm_calls=4
)

# Use with Ragdoll - embeddings concurrency comes from config
result = await ragdoll.ingest_with_graph(sources, options=options)
```

**Expected improvements**: 5-8x faster full pipeline execution for typical workloads. Actual speedups depend on your hardware, API rate limits, and document characteristics.

**Performance Testing**: Run `pytest tests/test_parallel_performance.py -v -s` to see detailed performance comparisons with metrics.

## Modular Retrieval Architecture

RAGdoll 2.1+ features a completely refactored retrieval system with clean separation between graph building and graph querying. The new architecture provides three composable retrievers:

### VectorRetriever

Semantic similarity search using vector embeddings:

```python
from ragdoll import VectorRetriever

vector_retriever = VectorRetriever(
    vector_store=vector_store,
    top_k=5,
    search_type="mmr"  # or "similarity", "similarity_score_threshold"
)
docs = vector_retriever.get_relevant_documents("query")
```

### GraphRetriever

Multi-hop graph traversal with BFS/DFS strategies and embedding-based seed search:

```python
from ragdoll import GraphRetriever

graph_retriever = GraphRetriever(
    graph_store=graph_store,
    vector_store=vector_store,  # Optional: enables embedding-based seed search
    embedding_model=embedding_model,  # Optional: required if vector_store provided
    top_k=5,
    max_hops=2,
    traversal_strategy="bfs",  # or "dfs"
    include_edges=True,
    seed_strategy="embedding"  # or "fuzzy" for text-based matching
)
docs = graph_retriever.get_relevant_documents("query")
```

**Key Features:**

- **Embedding-based seed search**: When configured with `vector_store` and `embedding_model`, GraphRetriever can find seed nodes by embedding similarity rather than fuzzy text matching, significantly improving retrieval accuracy
- **Automatic deduplication**: Handles multiple entities from the same document chunk that share vector IDs, ensuring efficient queries without duplicates
- **Flexible seed strategies**: Choose between embedding-based (`seed_strategy="embedding"`) or text-based fuzzy matching (`seed_strategy="fuzzy"`)
- **Vector store integration**: Seamlessly integrates with any LangChain vector store (Chroma, FAISS) to leverage existing embeddings

### HybridRetriever

Combines vector and graph retrieval with multiple strategies:

```python
from ragdoll import HybridRetriever

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    graph_retriever=graph_retriever,
    mode="rerank",  # or "concat", "weighted", "expand"
    vector_weight=0.6,
    graph_weight=0.4
)
docs = hybrid_retriever.get_relevant_documents("query")
```

### Complete Example with Graph Pipeline

```python
import asyncio
from ragdoll import Ragdoll
from ragdoll.pipeline import ingest_from_vector_store, IngestionOptions

async def main():
    ragdoll = Ragdoll()

    # Configure parallel execution for best performance
    options = IngestionOptions(
        parallel_extraction=True,          # Enabled by default
        max_concurrent_embeddings=3,       # Process 3 embedding batches concurrently
        max_concurrent_llm_calls=8,        # Limit concurrent LLM calls
        batch_size=10                      # Documents per batch
    )

    # Method 1: Ingest documents and build knowledge graph in one step
    result = await ragdoll.ingest_with_graph(
        ["path/to/docs/manual.pdf"],
        options=options
    )

    print(result["stats"])        # Ingestion metrics
    print(result["graph"])        # Pydantic Graph object
    print(result["graph_store"])  # NetworkX/Neo4j/JSON graph store
    print(result["vector_store"]) # Vector store with document embeddings

    # Method 2: Build graph from existing vector store (preserves vector_ids)
    # This ensures graph nodes reference the same embeddings as the vector store
    result = await ingest_from_vector_store(
        vector_store=existing_vector_store,
        graph_store=graph_store,
        entity_service=entity_service
    )
    graph_retriever = result["graph_retriever"]  # Pre-configured with vector store

    # Use the automatically configured hybrid retriever
    answer = ragdoll.query_hybrid("How does the widget fail-safe work?")
    print(answer["answer"])
    print(f"Retrieved {len(answer['documents'])} documents")

asyncio.run(main())
```

The helper `ingest_with_graph_sync()` wraps `asyncio.run()` for scripts that are not already running an event loop.

**Configuration:** All retrieval and performance settings are now consolidated in your config:

```yaml
# Retrieval configuration
retriever:
  vector:
    enabled: true
    top_k: 3
    search_type: "similarity"
  graph:
    enabled: true
    max_hops: 2
    traversal_strategy: "bfs"
    seed_strategy: "embedding" # Use embedding-based seed search (recommended)
    # seed_strategy: "fuzzy"    # Or use text-based fuzzy matching
    include_edges: true
  hybrid:
    mode: "concat"
    vector_weight: 0.6
    graph_weight: 0.4

# Performance settings (applied via IngestionOptions)
# These can also be configured programmatically
pipeline:
  batch_size: 10
  parallel_extraction: true
  max_concurrent_embeddings: 3
  max_concurrent_llm_calls: 8
```

See [`docs/retrieval.md`](docs/retrieval.md) for comprehensive documentation and [`examples/retrieval_examples.py`](examples/retrieval_examples.py) for complete examples.

### How Vector and Graph Stores Work Together

Ragdoll keeps both storage backends under the same orchestration surface:

- `Ragdoll.ingest_data(...)` (or the lower-level `IngestionPipeline`) always loads documents, chunks them, embeds each chunk, and writes those embeddings into the configured **vector store**.
- When `entity_extraction.extract_entities` (or `entity_extraction.graph_retriever.enabled`) is true, the same pipeline also fans out chunks to the **entity extraction service**, which generates a graph, persists it through the configured **graph store**, and can return a graph-aware retriever.
- Both flows are coordinated inside `IngestionPipeline`: it receives the shared `AppConfig`, builds the ingestion service, embedding model, vector store, and optionally graph store, and emits stats/retrievers back through `Ragdoll`.

**Building Graphs from Existing Vector Stores:**

RAGdoll 2.1+ introduces `EntityExtractionService.extract_from_vector_store()` and the corresponding `ingest_from_vector_store()` pipeline function. This allows you to:

- Extract documents directly from an existing vector store (Chroma, FAISS, or any LangChain vector store)
- Build a knowledge graph that references the **same vector IDs** as the vector store
- Create a GraphRetriever pre-configured with both the graph store and vector store for embedding-based seed search
- Avoid vector ID mismatches between graph nodes and vector store documents

This is particularly useful when you want to add graph capabilities to an existing vector store without re-ingesting all documents. The `vector_id` in each graph node's metadata matches the document ID in the vector store, enabling seamless integration between vector and graph retrieval.

**Deduplication Handling:**

Multiple entities extracted from the same document chunk naturally share the same `vector_id`. RAGdoll automatically deduplicates these shared IDs when:

- Building the embedding index (prevents duplicate ID errors from Chroma)
- Querying by embedding (returns all nodes sharing top embeddings without redundancy)
- Traversing the graph (standard graph traversal logic)

So even though `ragdoll/vector_stores` and `ragdoll/graph_stores` live in separate packages, their lifecycle is tied together via the pipeline entry points shown above.

## Installation

To install RAGdoll, follow these steps:

### Stable version install

`pip install python-ragdoll`

### Latest version install

1.  **Clone the Repository:**

```bash
    git clone https://github.com/nsasto/RAGdoll.git
    cd RAGdoll
```

2.  **Install Dependencies:**

```bash
    pip install -e .
```

This will install the required dependencies, including Langchain and Pydantic.

### Installation with optional features

RAGdoll supports optional dependency groups for different use cases:

```bash
# Base install (core functionality only)
pip install -e .

# Development tools (testing, linting, formatting)
pip install -e .[dev]

# Entity extraction and NLP features (spaCy, sentence transformers, PDF processing)
pip install -e .[entity]

# Graph database support (Neo4j, RDF)
pip install -e .[graph]

# All optional features combined
pip install -e .[all]
```

### From PyPI (recommended for production)

```bash
# Base install
pip install python-ragdoll

# With optional features
pip install python-ragdoll[all]  # or [dev], [entity], [graph]
```

## Architecture

RAGdoll's architecture is built around modular components and abstract base classes, making it highly extensible. Here's an overview of the key modules:

### Modules

- **`loaders`:** Responsible for loading data from various sources (e.g., directories, JSON files, web pages).
- **`chunkers`:** Handles the splitting of large text documents into smaller chunks.
- **`embeddings`:** Provides an interface for embedding models, allowing you to generate vector representations of text.
- **`vector_stores`:** Manages the storage and retrieval of vector embeddings.
- **`llms`:** Provides an interface to interact with different large language models.
- **`graph_stores`:** Manages the storage and querying of knowledge graphs.
- **`chains`:** Defines different types of chains, like retrieval QA (not implemented)

### Abstract Base Classes

Each module has an abstract base class (`BaseLoader`, `BaseChunker`, `BaseEmbeddings`, `BaseVectorStore`, `BaseGraphStore`, `BaseChain`) or protocol (the `BaseLLMCaller` interface) that defines a standard contract for that component type.

### Default Implementations

RAGdoll provides default implementations for most components, allowing you to quickly get started without having to write everything from scratch:

- **`Langchain-Markitdown`:** A default loader for most major file types.
  See `docs/loader_registry.md` for information on the loader registry and how
  to register custom loader classes under short names.
- **`RecursiveCharacterTextSplitter`:** A default text splitter.
- **`OpenAIEmbeddings`:** Default embeddings that use OpenAI's API.
- **`LangChain VectorStore factory`:** Plug-and-play wrapper for any LangChain vector store (Chroma, FAISS, etc.); see `docs/vector_stores.md`.
- **`OpenAILLM`**: A default OpenAI LLM.
- **`BaseGraphStore`**: A BaseGraphStore, it needs to be implemented.

## Key Design Decisions

RAGdoll 2.0 embraces LangChain's ecosystem for maximum flexibility and maintainability:

### Embeddings: LangChain Embeddings Objects

- **Decision**: Use LangChain `Embeddings` objects directly instead of creating custom embedding classes
- **Rationale**: LangChain provides robust, well-tested embedding implementations. Creating custom wrappers adds unnecessary complexity and maintenance burden.
- **Benefits**: Immediate access to all LangChain embedding providers (OpenAI, HuggingFace, etc.), automatic updates, consistent APIs.
- **Implementation**: `ragdoll.embeddings.get_embedding_model` reads your config and returns a ready-to-use LangChain embedding instance.

### Vector Stores: LangChain VectorStore Interface

- **Decision**: Accept any LangChain `VectorStore` object directly instead of requiring custom adapters
- **Rationale**: LangChain supports 40+ vector stores with consistent interfaces. Custom adapters create maintenance overhead and limit ecosystem integration.
- **Benefits**: Plug-and-play compatibility with any LangChain vector store (Chroma, FAISS, Pinecone, Weaviate, etc.), zero adapter code needed, future-proof with LangChain updates.
- **Implementation**: `BaseVectorStore` wraps LangChain `VectorStore` objects and delegates operations.

This design maximizes ecosystem compatibility while keeping RAGdoll's core orchestration logic clean and focused.

### System Diagram

For a visual walkthrough of how the ingestion, knowledge build, and query-time pieces connect, see the architecture diagram below (also available in `docs/architecture.md`):

```mermaid
graph TD
    subgraph Shared_Config["Bootstrap & Shared Services"]
        CFG["AppConfig / Config Manager"]
        CACHE["CacheManager"]
        METRICS["MetricsManager"]
        CFG --> CACHE
        CFG --> METRICS
    end

    subgraph Ingestion["Ingestion & Index Build"]
        SRC["Sources<br/>(files, URLs, loader registry)"] --> LOADER["DocumentLoaderService<br/>(auto loaders + caching + metrics)"]
        CACHE -.-> LOADER
        METRICS -.-> LOADER
        CFG --> LOADER
        LOADER --> DOCS["LangChain Documents"]
        DOCS --> CHUNK["Chunkers<br/>(split_documents)"]
        CFG --> CHUNK
        CHUNK --> EMB["Embedding Resolver<br/>(get_embedding_model)"]
        CFG --> EMB
        EMB --> VSTORE[("Vector Store<br/>Chroma/FAISS")]
        CHUNK --> ENT["EntityExtractionService<br/>(spaCy + LLM prompts)"]
        CFG --> ENT
        ENT --> GPERSIST["GraphPersistenceService<br/>(JSON/NetworkX/Neo4j)"]
        GPERSIST --> GRAPHSTORE[("Graph Store<br/>NetworkX/Neo4j")]
    end

    subgraph Query["Query & Reasoning"]
        USER["User Query"] --> RAG["Ragdoll Orchestrator"]
        CFG --> RAG

        subgraph Retrievers
            direction LR
            VR["VectorRetriever"]
            GR["GraphRetriever"]
            HR["HybridRetriever"]
        end

        RAG -- "uses" --> VR
        RAG -- "uses" --> GR
        RAG -- "uses" --> HR

        VR --> VSTORE
        GR --> GRAPHSTORE
        HR --> VR
        HR --> GR

        VR --> CONTEXT["Retrieved Chunks"]
        GR --> CONTEXT
        HR --> CONTEXT

        CONTEXT --> LLM["BaseLLMCaller"]
        RAG --> LLM
        CFG --> LLM
        LLM --> ANSWER["Answer / Structured Output"]
    end

    classDef service fill:#f9f,stroke:#333,stroke-width:1.5px;
    classDef storage fill:#dbeafe,stroke:#333,stroke-width:1.5px;
    classDef data fill:#fef3c7,stroke:#333,stroke-width:1.5px;
    classDef io fill:#fde68a,stroke:#333,stroke-width:1.5px;

    class CFG,LOADER,CHUNK,ENT,GPERSIST,RAG,LLM,VR,GR,HR service;
    class CACHE,METRICS,GRAPHSTORE,VSTORE storage;
    class DOCS,CONTEXT data;
    class SRC,USER,ANSWER io;
```

## Extensibility

RAGdoll is designed to be highly extensible. You can easily create custom components by following these steps:

1.  **Subclass the Base Class:** Create a new class that inherits from the relevant base class (e.g., `BaseLoader`, `BaseEmbeddings`).
2.  **Implement Abstract Methods:** Implement the abstract methods defined in the base class to provide your custom functionality.
3.  **Integrate into RAGdoll:** Pass an instance of your custom component to the `Ragdoll` class when you create it.

## Configuration

RAGdoll uses Pydantic to manage its configuration. This allows for:

- **Data Validation:** Automatic validation of configuration values.
- **Type Hints:** Clear type definitions for configuration settings.
- **Default Values:** Convenient default values for configuration options.

You can create a `Config` object and pass it to the `Ragdoll` class.

```python
from ragdoll import settings
from ragdoll.ragdoll import Ragdoll

# Grab the shared AppConfig (respects RAGDOLL_CONFIG_PATH when set)
app = settings.get_app()
config = app.config
vector_stores = config._config.setdefault("vector_stores", {})
vector_stores.setdefault("default_store", "chroma")
stores = vector_stores.setdefault("stores", {})
chroma_settings = stores.setdefault("chroma", {})
chroma_settings.setdefault("params", {})["persist_directory"] = "./my_vectors"

# Create Ragdoll with this configuration
ragdoll = Ragdoll(app_config=app)
```

### Entity Extraction Controls

The `entity_extraction` section of `default_config.yaml` now exposes several knobs for graph-centric workflows:

- `relationship_parsing`: choose the preferred output format (`json`, `markdown`, `auto`), optionally supply a custom parser class or schema, and pass parser-specific kwargs. This lets you tighten validation for LLM responses (e.g., point at your own Pydantic schema).
- `relationship_prompts`: declare a default prompt template plus per-provider overrides (e.g., map `"anthropic"` to a Claude-specific prompt). The service picks the prompt whose provider matches the active `BaseLLMCaller`.
- `graph_retriever`: enable creation of a graph retriever after entity extraction, select the backend (`simple` or `neo4j/langchain_neo4j`), and tune parameters like `top_k` or `include_edges`. When enabled, `EntityExtractionService` and `IngestionPipeline` expose a retriever you can plug into downstream chains.

Example excerpt:

```yaml
entity_extraction:
  relationship_parsing:
    preferred_format: "markdown"
    schema: "my_project.schemas.RelationshipListV2"
  relationship_prompts:
    default: "relationship_extraction"
    providers:
      openai: "relationship_extraction_openai"
      anthropic: "relationship_extraction_claude"
  graph_retriever:
    enabled: true
    backend: "neo4j"
    top_k: 10
```

See `docs/configuration.md` for the full field reference.

## Comparison with GraphRAG-style projects

This project began as a learning exercise; use the table below as a rough orientation (not marketing). Descriptions of other tools are based on their public docs - check their repos for specifics.

| Aspect                     | GraphRAG (Microsoft)                                                 | NanoGraphRAG                                                      | Fast GraphRAG                                       | RAGdoll (this repo)                                                                     |
| -------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Scope                      | Graph-first pipeline with multi-level community graphs and summaries | Lightweight, graph-centric variants; usually skip heavy hierarchy | Flat graph focus; optimized ingestion and traversal | Full RAG orchestrator (loaders + chunkers + embeddings + LLM + optional graph layer)    |
| Entity/Relation extraction | LLM-heavy entity + relation extraction; extensive prompts            | Simplified or minimal LLM extraction                              | Hybrid/heuristic extraction; typically flat graph   | spaCy NER + prompt-based relationship extraction per chunk; YAML prompt/parser controls |
| Graph structure            | Hierarchical communities + reports                                   | Typically flat or shallow                                         | Flat graph; no hierarchy                            | Flat graph only; persisted to JSON/NetworkX/Neo4j via `GraphPersistenceService`         |
| Summaries                  | Precomputed community summaries                                      | Often none or single-level                                        | Often skipped; sometimes on-demand                  | None precomputed; summaries come from query-time LLM calls if you add them              |
| Retrieval                  | Combines vector + hierarchical graph summaries                       | Lightweight graph or vector                                       | Flat graph traversal; vector when configured        | Vector-first; optional simple/Neo4j graph retriever from `ingest_with_graph`            |
| Defaults                   | Cloud LLMs (GPT-4/O) with hierarchical post-processing               | Small/local-friendly LLMs                                         | Async/flat graph tuned for speed                    | LangChain defaults; OpenAI models by default but fully swappable to local/open-source   |
| Benchmarks                 | Published externally                                                 | Vary by implementation                                            | Vary by implementation                              | None published here—measure with your models/stores                                     |

## Contributing

Contributions to RAGdoll are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Make your changes and write tests.
4.  Submit a pull request.

## License

RAGdoll is licensed under the [MIT License](LICENSE).
