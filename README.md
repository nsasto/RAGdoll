![Ragdoll](img/github-header-image.png)

[![CI](https://github.com/nsasto/RAGdoll/actions/workflows/ci.yml/badge.svg)](https://github.com/nsasto/RAGdoll/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-2.0.1-blue.svg)](https://github.com/nsasto/RAGdoll/releases)
[![Stable](https://badge.fury.io/py/python-ragdoll.svg)](https://pypi.org/project/python-ragdoll/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# RAGdoll: A Flexible and Extensible RAG Framework

Welcome to Ragdoll 2.0! This release marks a significant overhaul of the Ragdoll project, focusing on enhanced flexibility, extensibility, and maintainability. We've completely refactored the core architecture to make it easier than ever to adapt Ragdoll to your specific needs and integrate it with the broader LangChain ecosystem. This document outlines the major changes and improvements you'll find in this new version.

# 🧭 Project Overview

RAGdoll 2 is an extensible framework for building Retrieval-Augmented Generation (RAG) applications. It provides a modular architecture that allows you to easily integrate various data sources, chunking strategies, embedding models, vector stores, large language models (LLMs), and graph stores. RAGdoll is designed to be flexible and fast, without any third party dependencies. It's also designed to accomodate a broad array of file types without any initial dependency on third party hosted services using [langchain-markitdown](https://github.com/nsasto/langchain-markitdown). The loaders can easily be swapped out with any compatible lanchain loader when ready for production.

Note that RAGdoll 2 is a complete overhaul of the initial RAGdoll project and is not backwards compatible in any respect.

## What's New

### Enhanced Features in RAGdoll 2.0

This version of RAGdoll introduces several key features that improve the flexibility and usability of the framework:

- **Caching:** RAGdoll now supports caching, allowing you to store and reuse results from previous operations. This can significantly speed up the execution of your RAG applications by avoiding redundant computations.
- **Auto Loader Selection**: RAGdoll now includes loaders for multiple file types (not only pdf). The loader defaults to Langchain-Markitdown loaders, but can be configured to use any Lanchain compatible loader.
- **Monitoring:** A new monitoring capability has been added to RAGdoll. This allows you to track and understand the performance and behavior of your RAG applications over time.

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

### Graph Retrieval Pipeline

When you enable `entity_extraction.graph_retriever.enabled` in your config, you can trigger the full ingestion pipeline (chunking, embeddings, entity extraction, graph persistence) and retrieve a knowledge-graph-aware retriever directly from the `Ragdoll` API:

```python
import asyncio
from ragdoll.ragdoll import Ragdoll
from ragdoll.pipeline import IngestionOptions

async def main():
    ragdoll = Ragdoll()
    result = await ragdoll.ingest_with_graph(
        ["path/to/docs/manual.pdf"],
        options=IngestionOptions(parallel_extraction=False),
    )
    print(result["stats"])           # ingestion metrics
    print(result["graph"])           # pydantic Graph object
    retriever = result["graph_retriever"]
    answers = retriever.invoke("How does the widget fail-safe work?")

asyncio.run(main())
```

The helper `ingest_with_graph_sync()` wraps `asyncio.run()` for scripts that are not already running an event loop.
See `examples/graph_retriever_example.py` for a complete runnable script.

### How Vector and Graph Stores Work Together

Ragdoll keeps both storage backends under the same orchestration surface:

- `Ragdoll.ingest_data(...)` (or the lower-level `IngestionPipeline`) always loads documents, chunks them, embeds each chunk, and writes those embeddings into the configured **vector store**.
- When `entity_extraction.extract_entities` (or `entity_extraction.graph_retriever.enabled`) is true, the same pipeline also fans out chunks to the **entity extraction service**, which generates a graph, persists it through the configured **graph store**, and can return a graph-aware retriever.
- Both flows are coordinated inside `IngestionPipeline`: it receives the shared `AppConfig`, builds the ingestion service, embedding model, vector store, and optionally graph store, and emits stats/retrievers back through `Ragdoll`.

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
    %% Ingestion + Chunking
    subgraph Ingestion
        A["Input sources<br/>(files, URLs, loaders)"] --> B["Loader pipeline"]
        B --> C["Chunking Service<br/>(BaseChunkingService + plugins)"]
    end
    C --> D["Chunks (GTChunk)"]

    %% Knowledge Construction
    subgraph Knowledge_Build
        D --> E["Information Extraction<br/>(entities & relations)"]
        E --> F["Knowledge Graph Upsert<br/>(policy interface)"]
        E --> G["Embedding Pipeline<br/>(single pass)"]
        G --> H["VectorStoreAdapter<br/>(dynamic class e.g. Chroma/Hnswlib)"]
        F --> I(("Graph Storage<br/>(IGraphStorage, Neo4j, etc.)"))
        H --> J(("Vector DB"))
    end

    %% Query + Reasoning
    subgraph Query_Runtime
        Q["User Query"] --> R["State Manager<br/>(retrieval orchestration)"]
        R --> H
        R --> I
        R --> S["Context Assembly<br/>(chunks + KG facts)"]
        S --> T["Prompt Builder"]
        T --> U["LLM Caller<br/>(LangChain adapters)"]
        U --> V["Answer"]
    end

    style A fill:#ccf,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#fef3c7,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#d1fae5,stroke:#333,stroke-width:2px
    style I fill:#dbeafe,stroke:#333,stroke-width:2px
    style J fill:#dbeafe,stroke:#333,stroke-width:2px
    style Q fill:#fde68a,stroke:#333,stroke-width:2px
    style R fill:#f9f,stroke:#333,stroke-width:2px
    style S fill:#fef3c7,stroke:#333,stroke-width:2px
    style T fill:#f9f,stroke:#333,stroke-width:2px
    style U fill:#e0e7ff,stroke:#333,stroke-width:2px
    style V fill:#ccf,stroke:#333,stroke-width:2px
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
config._config["vector_store"]["params"]["persist_directory"] = "./my_vectors"

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

## Contributing

Contributions to RAGdoll are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Make your changes and write tests.
4.  Submit a pull request.

## License

RAGdoll is licensed under the [MIT License](LICENSE).
