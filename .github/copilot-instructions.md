# RAGdoll AI Agent Instructions

## Project Architecture

**RAGdoll** is a modular RAG (Retrieval-Augmented Generation) orchestrator combining vector search with optional knowledge graph capabilities. Unlike pure GraphRAG systems, RAGdoll is **vector-first with graph augmentation** - the graph enriches context rather than replacing semantic search.

### Core Design Principles

1. **LangChain Native**: Direct use of LangChain `Embeddings` and `VectorStore` objects without custom wrappers - maximum ecosystem compatibility
2. **Shared Configuration**: Single `AppConfig` instance via `ragdoll.settings.get_app()` ensures consistent config/cache/metrics across all components
3. **Async-First Pipeline**: Parallel entity extraction and embedding with configurable rate limiting (`IngestionOptions`)
4. **Modular Retrievers**: Clean separation between `VectorRetriever`, `GraphRetriever`, `PageRankGraphRetriever`, and `HybridRetriever`

### Configuration Hierarchy

```python
# Order of precedence:
# 1. Explicit config_path argument
# 2. RAGDOLL_CONFIG_PATH environment variable
# 3. ragdoll/config/default_config.yaml

from ragdoll import settings
from ragdoll.app_config import bootstrap_app

# Get shared singleton (recommended)
app = settings.get_app()  # Respects RAGDOLL_CONFIG_PATH
config = app.config

# Or bootstrap with overrides
app = bootstrap_app(config_path="custom.yaml", overrides={
    "ingestion": {"batch_size": 20},
    "embeddings": {"max_concurrent_embeddings": 5}
})
```

**Critical**: Always use `settings.get_app()` for the shared singleton. Creating multiple `AppConfig` instances defeats caching and metrics.

### Data Flow: Ingestion → Query

```
Documents → DocumentLoaderService (with caching)
         → split_documents (chunking strategies: none/recursive/markdown)
         → Embeddings (parallel batching)
         → VectorStore (Chroma/FAISS via LangChain)
         → EntityExtractionService (spaCy NER + LLM prompts)
         → GraphStore (NetworkX/Neo4j/JSON)

Query → VectorRetriever (semantic similarity)
      → GraphRetriever (BFS/DFS with embedding-based seeds)
      → HybridRetriever (concat/weighted/rerank/expand modes)
      → BaseLLMCaller (OpenAI/Anthropic/local)
      → Answer + source documents
```

**Key insight**: Vector and graph stores are built **in parallel** during ingestion via `IngestionPipeline`. Graph nodes store `vector_id` metadata linking back to embeddings for hybrid retrieval.

## Critical Workflows

### Running Benchmarks

```powershell
# Location: benchmarks/
cd benchmarks

# Create indices (vector + graph)
python ragdoll_benchmark.py -d 2wikimultihopqa -n 101 --mode hybrid --create

# Run queries on existing indices (faster)
python ragdoll_benchmark.py -d 2wikimultihopqa -n 101 --mode hybrid --benchmark

# No-chunking mode (whole-passage retrieval, typically 7% better on pre-segmented data)
python ragdoll_benchmark.py -d 2wikimultihopqa -n 101 --mode vector --no-chunking --create --benchmark

# Results saved to: benchmarks/results/ragdoll_<dataset>_<n>_<mode>[_nochunk].json
```

**Performance expectations**: Chunking HURTS performance on pre-segmented datasets like 2wikimultihopqa (46.5% no-chunk vs 39.6% chunked). Always test both modes on your data.

### Testing

```bash
# Run all tests with coverage
pytest

# Test specific module
pytest tests/test_app_config.py -v

# Performance benchmarks (parallel vs sequential)
pytest tests/test_parallel_performance.py -v -s

# Skip slow integration tests
pytest -m "not integration"
```

**Test fixtures**: `conftest.py` provides shared fixtures for `Config`, `AppConfig`, mock LLMs, and test documents. Always use fixtures to avoid singleton pollution.

### Environment Setup

```bash
# Required for OpenAI models (default)
export OPENAI_API_KEY="sk-..."

# Optional: custom config path
export RAGDOLL_CONFIG_PATH="./my_config.yaml"

# Install with all features (entity extraction, graph stores, dev tools)
pip install -e .[all]

# Or selective installs
pip install -e .[dev]      # Testing/linting only
pip install -e .[entity]   # spaCy, PDF processing
pip install -e .[graph]    # Neo4j, RDF support
```

## Project-Specific Patterns

### 1. Parallel Execution (BREAKING in 2.1+)

Parallel embedding logic **moved from `IngestionPipeline` to `BaseVectorStore`** for reusability:

```python
from ragdoll.pipeline import IngestionOptions

options = IngestionOptions(
    batch_size=20,
    max_concurrent_llm_calls=15,     # Entity extraction concurrency
    parallel_extraction=True,         # Default: True for 4+ documents
    # max_concurrent_embeddings removed - now in embeddings.max_concurrent_embeddings
)

# Embeddings concurrency controlled via config:
config.embeddings_config.max_concurrent_embeddings = 5

# Vector store calls add_documents_parallel() internally
result = await ragdoll.ingest_with_graph(sources, options=options)
```

**Why**: Enables parallel embedding in standalone vector store operations without full pipeline.

### 2. Vector ID Linking

Graph nodes MUST reference same `vector_id` as vector store documents for embedding-based seed search:

```python
# When building from existing vector store (preserves vector IDs):
from ragdoll.pipeline import ingest_from_vector_store

result = await ingest_from_vector_store(
    vector_store=existing_vector_store,
    graph_store=graph_store,
    entity_service=entity_service
)
graph_retriever = result["graph_retriever"]  # Pre-configured with vector store

# GraphRetriever with embedding seeds (recommended):
graph_retriever = GraphRetriever(
    graph_store=graph_store,
    vector_store=vector_store,           # Enables embedding-based search
    embedding_model=embedding_model,
    seed_strategy="embedding",           # vs "fuzzy" text matching
    max_hops=2
)
```

**Critical**: `EntityExtractionService.extract_from_vector_store()` automatically assigns shared `vector_id` to all entities from same chunk.

### 3. Hybrid Retrieval Modes

```python
from ragdoll.retrieval import HybridRetriever

# Mode comparison:
# - concat: Concatenates vector + graph results (fastest)
# - weighted: Scores documents by (vector_weight * v_score + graph_weight * g_score)
# - rerank: LLM-based reranking of combined results (slowest, highest quality)
# - expand: Seeds with vector, expands via graph (good for multi-hop)

hybrid = HybridRetriever(
    vector_retriever=vector_retriever,
    graph_retriever=graph_retriever,
    mode="expand",  # or concat/weighted/rerank
    vector_weight=0.6,
    graph_weight=0.4
)
```

**Benchmark finding**: All modes show similar performance (~40%) on 2wikimultihopqa because vector search already finds most ground truth. Graph expansion primarily adds **relationship metadata** to documents, not new passages.

### 4. Chunking Configuration

```yaml
# In default_config.yaml or via IngestionOptions
chunking_options:
  chunking_strategy: "none" # or "recursive", "markdown", "token"
  chunk_size: 2000
  chunk_overlap: 200
```

```python
# Programmatic override
options = IngestionOptions(
    chunking_options={
        "chunking_strategy": "none",  # Whole-passage retrieval
    }
)
```

**Decision tree**:

- Pre-segmented passages (wiki articles, Q&A pairs): `chunking_strategy: "none"`
- Long documents (PDFs, books): `chunking_strategy: "recursive"` with `chunk_size=1000-2000`
- Structured markdown: `chunking_strategy: "markdown"`

### 5. Prompt Templates

```python
from ragdoll.llms import get_prompt_template

# Templates stored in ragdoll/prompts/ as separate files
template = get_prompt_template("relationship_extraction")

# Override via AppConfig
app = bootstrap_app(prompt_templates={
    "relationship_extraction": "Extract entities: {text}"
})

# Provider-specific prompts (config: entity_extraction.relationship_prompts)
relationship_prompts:
  default: "relationship_extraction"
  providers:
    openai: "relationship_extraction_openai"
    anthropic: "relationship_extraction_claude"
```

## Integration Points

### LangChain Compatibility

- **Embeddings**: Use any `langchain_openai.OpenAIEmbeddings`, `langchain_huggingface.HuggingFaceEmbeddings`, etc. directly
- **VectorStores**: Wrap any `langchain.vectorstores.*` with `BaseVectorStore` - Chroma, FAISS, Pinecone, Weaviate all supported
- **Documents**: All pipelines use `langchain_core.documents.Document` - no custom schemas
- **Retrievers**: Implement LangChain `BaseRetriever` interface for drop-in chain compatibility

### External Services

- **OpenAI**: Default LLM/embeddings provider (requires `OPENAI_API_KEY`)
- **Neo4j**: Optional graph backend (requires `neo4j` package, connection URI in config)
- **spaCy**: Entity extraction (requires `en_core_web_sm` model: `python -m spacy download en_core_web_sm`)

## Common Pitfalls

1. **Multiple AppConfig instances**: Always use `settings.get_app()`, never create new `AppConfig()` in application code
2. **Forgetting `--no-chunking`**: Pre-segmented benchmarks need this flag for fair comparison
3. **Missing vector_id links**: When building custom graphs, ensure nodes have `vector_id` metadata for embedding-based retrieval
4. **Sync vs async**: Use `ingest_with_graph_sync()` wrapper in non-async code, or `asyncio.run(ingest_with_graph())`
5. **Cache poisoning**: Clear cache with `app.get_cache_manager().clear()` when config changes significantly

## File Locations

- **Config**: `ragdoll/config/default_config.yaml` (default), override via `RAGDOLL_CONFIG_PATH`
- **Prompts**: `ragdoll/prompts/*.txt` (relationship extraction, entity types, etc.)
- **Benchmarks**: `benchmarks/` (complete suite with datasets, metrics, comparison scripts)
- **Tests**: `tests/` (pytest with fixtures in `conftest.py`)
- **Demo App**: `demo_app/` (FastAPI + HTMX interactive UI)
- **Docs**: `docs/` (architecture, API reference, component guides)

## Key Modules Reference

- `ragdoll.ragdoll.Ragdoll`: Main orchestrator class
- `ragdoll.pipeline`: `ingest_documents`, `ingest_from_vector_store`, `IngestionOptions`
- `ragdoll.retrieval`: `VectorRetriever`, `GraphRetriever`, `HybridRetriever`, `PageRankGraphRetriever`
- `ragdoll.app_config`: `AppConfig`, `bootstrap_app` (shared config/cache/metrics)
- `ragdoll.settings`: `get_app()`, typed config accessors
- `ragdoll.embeddings`: `get_embedding_model` (LangChain factory)
- `ragdoll.llms`: `get_llm`, `get_llm_caller`, `get_prompt_template`
- `ragdoll.chunkers`: `get_text_splitter`, `split_documents`
- `ragdoll.entity_extraction`: `EntityExtractionService`, `extract_from_vector_store`

## Quick Reference

```python
# Standard ingestion flow
from ragdoll import Ragdoll
from ragdoll.pipeline import IngestionOptions

ragdoll = Ragdoll()  # Uses settings.get_app() internally
result = await ragdoll.ingest_with_graph(
    ["docs/manual.pdf"],
    options=IngestionOptions(
        batch_size=20,
        max_concurrent_llm_calls=15,
        chunking_options={"chunking_strategy": "recursive"}
    )
)

# Query with hybrid retrieval
answer = ragdoll.query_hybrid("What is the fail-safe mechanism?")
print(answer["answer"])

# Benchmark comparison (no chunking)
python benchmarks/ragdoll_benchmark.py -d 2wikimultihopqa -n 101 --mode vector --no-chunking --create --benchmark
```
