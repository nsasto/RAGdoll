# RAGdoll Architecture Overview

The diagram below shows how ingestion, knowledge construction, and query-time reasoning compose within RAGdoll. Each block corresponds to a pluggable service or adapter, so you can swap in custom implementations as needed.

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
        EMB --> VSTORE["BaseVectorStore<br/>(Chroma/FAISS/etc.)"]
        CHUNK --> ENT["EntityExtractionService<br/>(spaCy + LLM prompts)"]
        CFG --> ENT
        ENT --> GPERSIST["GraphPersistenceService<br/>(JSON/NetworkX/Neo4j)"]
        GPERSIST --> GRAPHSTORE[("Graph Store Handle")]
        GPERSIST --> GRETR["Graph Retriever<br/>(simple/Neo4j)"]
    end

    subgraph Query["Query & Reasoning"]
        USER["User Query"] --> RAG["Ragdoll Orchestrator"]
        CFG --> RAG
        RAG --> VSTORE
        RAG --> GRETR
        VSTORE --> CONTEXT["Retrieved chunks"]
        GRETR --> CONTEXT
        RAG --> LLM["BaseLLMCaller / call_llm_sync"]
        CFG --> LLM
        LLM --> ANSWER["Answer / structured output"]
        CONTEXT --> ANSWER
    end

    classDef service fill:#f9f,stroke:#333,stroke-width:1.5px;
    classDef storage fill:#dbeafe,stroke:#333,stroke-width:1.5px;
    classDef data fill:#fef3c7,stroke:#333,stroke-width:1.5px;
    classDef io fill:#fde68a,stroke:#333,stroke-width:1.5px;

    class CFG,LOADER,CHUNK,ENT,GPERSIST,RAG,LLM service;
    class CACHE,METRICS,GRAPHSTORE,VSTORE storage;
    class DOCS,CONTEXT data;
    class SRC,USER,ANSWER io;
```

For deeper dives into each subsystem, see the dedicated docs in `docs/ingestion.md`, `docs/chunking.md`, `docs/embeddings.md`, `docs/vector_stores.md`, `docs/graph_stores.md`, and `docs/llm_integration.md`.
Recent changes worth noting:

- **LLM abstractions** are handled via `BaseLLMCaller`, so LangChain models, custom HTTP clients, or fake callers can be injected consistently. `get_llm_caller()` in `ragdoll.llms` bridges config-driven models into this interface.
- **Graph persistence** lives in `GraphPersistenceService`, which handles JSON/NetworkX/Neo4j output and exposes `create_retriever()` hooks (the new `simple` in-memory retriever and the Neo4j-backed retriever).
- **Hybrid ingestion** is performed by `IngestionPipeline` and surfaced through `Ragdoll.ingest_with_graph()` / `ingest_with_graph_sync()`, giving callers a single entry point for vector + graph indexing and retrieval.
