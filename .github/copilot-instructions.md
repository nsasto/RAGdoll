# RAGdoll Copilot Guide

## Posture

- Vector-first RAG orchestrator with optional graph augmentation (NetworkX/Neo4j/JSON); no community summaries or hierarchical graph steps.
- LangChain primitives everywhere (`Document`, `Embeddings`, `VectorStore`); RAGdoll wrappers stay thin for compatibility.
- Default config lives at `ragdoll/config/default_config.yaml`; override via `config_path` arg or `RAGDOLL_CONFIG_PATH`.
- Stay within the singleton `AppConfig` from `settings.get_app()` so cache/metrics/prompt templates stay shared.

## Ingestion flow (`ragdoll/pipeline/__init__.py`, `ragdoll/ragdoll.py`)

- `DocumentLoaderService` resolves loaders from config, uses cache/metrics from `AppConfig`, and accepts URLs/paths or LangChain `Document`s.
- `get_text_splitter` respects `chunking_strategy` (`none` returns `None` -> no chunking) and caches splitters; `split_documents` preserves originals when splitter is `None`.
- Embeddings via `get_embedding_model` (default `text-embedding-3-small`); concurrency comes from `embeddings_config.max_concurrent_embeddings`.
- Vector stores are built with `vector_store_from_config` (Chroma/FAISS/etc.) and wrapped in `BaseVectorStore`.
- Pipeline always writes embeddings when entity extraction is enabled: `add_documents_parallel(batch_size=IngestionOptions.batch_size, max_concurrent=embeddings_config.max_concurrent_embeddings)` then stamps `metadata.vector_id` and `metadata.embedding_timestamp`.
- Graph creation is optional; `extract_entities=False` auto-enables `skip_graph_store`. Graph persistence handled via `GraphPersistenceService`.
- `IngestionOptions` lets you skip vector/graph stores, override subcomponent configs, set `augment` (append vs replace), and pass LLM/LLM caller overrides for extraction.
- `Ragdoll.ingest_with_graph` wires the pipeline, keeps references to the graph store/graph retrievers, and exposes `query`, `query_hybrid`, and `query_pagerank`.

## Retrieval stack (`ragdoll/retrieval/*`)

- `VectorRetriever`: wrapper around LangChain search types (`similarity`, `mmr`, etc.) with `search_kwargs` passthrough.
- `GraphRetriever`: embedding-seeded BFS/DFS traversal; builds an embedding index from graph nodes by `vector_id` via `VectorStoreAdapter`; falls back to fuzzy matching if embeddings/IDs are missing; configurable via `retriever.graph` (`enabled`, `top_k`, `max_hops`, `traversal_strategy`, `hybrid_alpha`, `prebuild_index`, fallbacks).
- `PageRankGraphRetriever`: builds a bounded subgraph around seed nodes (embedding or keyword), runs personalized PageRank, dedups on `vector_id`, and falls back to vector search when empty. Controlled by `retriever.pagerank`.
- `HybridRetriever`: modes `concat`, `weighted`, `rerank`, `expand`; deduplicates merged results; can be constructed from stores or existing retrievers; config under `retriever.hybrid` (`mode`, weights, `deduplicate`).
- `RerankerRetriever`: wraps any retriever with relevance-based reranking using LLM, Cohere, or cross-encoder; over-retrieves then scores/filters documents; uses separate model from main LLM (default gpt-3.5-turbo) for cost efficiency; config under `retriever.reranker` (`enabled`, `provider`, `top_k`, `over_retrieve_multiplier`, `score_threshold`). When enabled, `Ragdoll._maybe_wrap_with_reranker` automatically wraps hybrid and PageRank retrievers.
- `Ragdoll.query` defaults to vector, but will use graph/hybrid/pagerank retrievers when requested and available; reranking is automatically applied when enabled.

## Vector stores & embeddings (`ragdoll/vector_stores/*`)

- Use the factory helpers (`vector_store_from_config`, `create_vector_store`) so embeddings are attached correctly; default registry supports Chroma/FAISS/DocArray.
- `BaseVectorStore.add_documents_parallel` replaces pipeline-specific helpers; honors backend batch limits, retries failed batches, and preserves ordering.
- `VectorStoreAdapter` normalizes embedding/document fetches by ID (Chroma/FAISS/generic). Graph and PageRank retrievers rely on IDs being aligned with the stored chunks.
- When building custom graphs, always carry `vector_id` through node metadata so retrieval can resolve embeddings and source passages.

## Entity extraction & prompts (`ragdoll/entity_extraction/*`, `ragdoll/prompts`)

- spaCy NER drives entity detection; relationship extraction is prompt-driven via LLM (`relationship_prompts.default` with provider overrides). Prompts live in `ragdoll/prompts/*.txt`.
- `EntityExtractionService.extract` parallelizes with `max_concurrent_llm_calls` (config or override) and persists via `GraphPersistenceService` (neo4j/networkx/json/custom graph object).
- `extract_from_vector_store` builds graphs directly from an existing vector store while preserving `vector_id` links; pass `embedding_model` when you need a graph retriever afterwards.

## Chunking (`ragdoll/chunkers/__init__.py`)

- Strategies: `none`, `recursive`, `character`, `markdown`, `code`, `token`. Default comes from `chunker.chunking_strategy`/`default_splitter`.
- Overlap is clamped to stay below `chunk_size`; markdown and code splitters are cached; `split_documents(..., batch_size=...)` keeps memory predictable.

## Config, env, and installs

- Config precedence: explicit `config_path` > `RAGDOLL_CONFIG_PATH` > packaged default. Avoid constructing `AppConfig` manually - use `bootstrap_app` only when you must inject overrides in tests/tools.
- Extras: `[entity]` (spaCy/PDF), `[graph]` (Neo4j/RDF), `[dev]` (pytest/linters), `[all]` pulls everything.
- Secrets can be specified as `os.environ/KEY` in YAML; `resolve_env_reference` resolves them before model creation.

## Testing cues

- Pytest is the standard; fixtures in `tests/conftest.py` provide shared `Config`, `AppConfig`, mock LLM callers, and documents - use them to avoid polluting the singleton cache.
- Skip slow runs with `pytest -m "not integration"`. Parallel embedding behavior is covered in `tests/test_vector_store_parallel.py` and `tests/test_parallel_performance.py`.

## Pitfalls to avoid

- Multiple `AppConfig` instances break caching/metrics; reuse the shared one or pass it through constructors.
- Graph retrieval needs embeddings plus `vector_id` links; if you disable vector ingestion, graph retrievers cannot seed from embeddings and will fall back to fuzzy matching.
- Keep boundary types as LangChain `Document`; when coercing dicts, ensure `page_content` and `metadata` keys exist (see `_coerce_to_document` in the pipeline).
- Benchmarks are data/model-dependent; avoid hardcoding performance claims - rerun `benchmarks/ragdoll_benchmark.py` with and without chunking for your data.
