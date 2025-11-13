# RAGdoll Architecture Overview

The diagram below shows how ingestion, knowledge construction, and query-time reasoning compose within RAGdoll. Each block corresponds to a pluggable service or adapter, so you can swap in custom implementations as needed.

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
        E --> F["Graph Persistence<br/>(JSON, NetworkX, Neo4j)"]
        E --> G["Embedding Pipeline<br/>(single pass)"]
        G --> H["VectorStoreAdapter<br/>(Chroma, FAISS, etc.)"]
        F --> I(("Graph Storage or Retriever"))
        H --> J(("Vector DB"))
    end

    %% Query + Reasoning
    subgraph Query_Runtime
        Q["User Query"] --> R["Ragdoll Orchestrator"]
        R --> H
        R --> I
        R --> S["Context Assembly<br/>(chunks + KG facts)"]
        S --> T["Prompt Builder"]
        T --> U["BaseLLMCaller<br/>(LangChain adapters)"]
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

For deeper dives into each subsystem, see the dedicated docs in `docs/ingestion.md`, `docs/chunking.md`, `docs/embeddings.md`, `docs/vector_stores.md`, `docs/graph_stores.md`, and `docs/llm_integration.md`.
Recent changes worth noting:

- **LLM abstractions** are handled via `BaseLLMCaller`, so LangChain models, custom HTTP clients, or fake callers can be injected consistently. `get_llm_caller()` in `ragdoll.llms` bridges config-driven models into this interface.
- **Graph persistence** lives in `GraphPersistenceService`, which handles JSON/NetworkX/Neo4j output and exposes `create_retriever()` hooks (the new `simple` in-memory retriever and the Neo4j-backed retriever).
- **Hybrid ingestion** is performed by `IngestionPipeline` and surfaced through `Ragdoll.ingest_with_graph()` / `ingest_with_graph_sync()`, giving callers a single entry point for vector + graph indexing and retrieval.
