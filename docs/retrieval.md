# Retrieval Architecture

RAGdoll provides a modular, composable retrieval system that combines vector similarity search with graph-based traversal for enhanced context gathering. The retrieval architecture is LangChain-compatible and designed to be both powerful and easy to configure.

## Overview

The retrieval system consists of three main components:

1. **VectorRetriever**: Semantic similarity search using vector embeddings
2. **GraphRetriever**: Multi-hop graph traversal for relational context
3. **HybridRetriever**: Combines vector and graph retrieval strategies

All retrievers implement a common `BaseRetriever` interface, making them interchangeable and composable.

## Architecture Principles

### Separation of Concerns

- **Graph Building** (Entity Extraction): Creates knowledge graphs from documents
- **Graph Querying** (Retrieval): Searches and traverses graphs at query time

These are now distinct phases with clear boundaries, replacing the previous approach where graph retrieval was coupled with entity extraction.

### LangChain Compatibility

All retrievers implement LangChain's standard retriever interface:

```python
from ragdoll.retrieval import VectorRetriever, GraphRetriever, HybridRetriever

# All retrievers support the standard interface
documents = retriever.get_relevant_documents(query)
documents = await retriever.aget_relevant_documents(query)
```

## Configuration

### Unified Retriever Configuration

All retrieval settings are now consolidated under a single `retriever:` block in `default_config.yaml`:

```yaml
retriever:
  # Vector retrieval settings
  vector:
    enabled: true
    top_k: 3
    search_type: "similarity" # Options: "similarity", "mmr", "similarity_score_threshold"
    search_kwargs: {}

  # Graph retrieval settings
  graph:
    enabled: true
    backend: "networkx" # Options: "networkx", "neo4j", "simple"
    top_k: 5
    max_hops: 2
    include_edges: true
    traversal_strategy: "bfs" # Options: "bfs" (breadth-first), "dfs" (depth-first)
    min_score: 0.0

  # Hybrid combination settings
  hybrid:
    mode: "concat" # Options: "concat", "rerank", "weighted", "expand"
    vector_weight: 0.6
    graph_weight: 0.4
    deduplicate: true
```

### Migration from Previous Config

**Old Config** (deprecated):

```yaml
entity_extraction:
  graph_retriever:
    enabled: false
    backend: "simple"
    top_k: 5
```

**New Config**:

```yaml
retriever:
  graph:
    enabled: true
    backend: "networkx"
    top_k: 5
    max_hops: 2
```

The old `entity_extraction.graph_retriever` config has been removed to eliminate the mixing of graph creation and graph querying concerns.

## Vector Retrieval

### VectorRetriever

Performs semantic similarity search using vector embeddings stored in a vector database.

**Features:**

- Multiple search strategies (similarity, MMR, threshold-based)
- Configurable top-k results
- Support for all LangChain vector stores (FAISS, Chroma, Pinecone, etc.)

**Configuration:**

```yaml
retriever:
  vector:
    enabled: true
    top_k: 3
    search_type: "similarity"
    search_kwargs:
      score_threshold: 0.7 # For similarity_score_threshold mode
      fetch_k: 20 # For MMR mode
```

**Usage:**

```python
from ragdoll.retrieval import VectorRetriever

vector_retriever = VectorRetriever(
    vector_store=vector_store,
    top_k=5,
    search_type="mmr"  # Maximum Marginal Relevance for diversity
)

docs = vector_retriever.get_relevant_documents("What is RAGdoll?")
```

### Search Types

1. **Similarity** (default): Standard cosine similarity search
2. **MMR** (Maximum Marginal Relevance): Balances relevance and diversity
3. **Similarity Score Threshold**: Only returns documents above a score threshold

## Graph Retrieval

### GraphRetriever

Performs multi-hop graph traversal to find contextually relevant entities and relationships.

**Features:**

- Multi-hop traversal (configurable depth)
- BFS (breadth-first) or DFS (depth-first) strategies
- Relationship-aware context gathering
- Score-based seed node selection
- Decay scoring by hop distance

**Configuration:**

```yaml
retriever:
  graph:
    enabled: true
    backend: "networkx"
    top_k: 5 # Number of seed nodes
    max_hops: 2 # Traversal depth
    include_edges: true # Include relationship information
    traversal_strategy: "bfs"
    min_score: 0.0 # Minimum relevance score for seeds
```

**Usage:**

```python
from ragdoll.retrieval import GraphRetriever

graph_retriever = GraphRetriever(
    graph_store=graph_store,
    top_k=5,
    max_hops=2,
    traversal_strategy="bfs",
    include_edges=True
)

docs = graph_retriever.get_relevant_documents("Who works for Acme Corp?")
```

### How Graph Retrieval Works

1. **Seed Selection**: Find nodes matching the query using keyword scoring
2. **Traversal**: Perform BFS/DFS from seed nodes up to `max_hops` depth
3. **Scoring**: Apply decay to scores based on hop distance (0.7^depth)
4. **Document Creation**: Convert subgraph to Document objects with metadata

**Example Traversal:**

```
Query: "What does Alice work on?"

Seed Nodes: [Alice (Person)]
  └─> Hop 1: [WORKS_FOR] -> Acme Corp (Organization)
      └─> Hop 2: [PRODUCES] -> Product X (Product)
              [LOCATED_IN] -> San Francisco (Location)

Retrieved Context:
- Alice (Person): Software Engineer
  - WORKS_FOR -> Acme Corp
- Acme Corp (Organization): Technology company
  - PRODUCES -> Product X
  - LOCATED_IN -> San Francisco
```

### Traversal Strategies

**BFS (Breadth-First Search)**:

- Explores all neighbors at current depth before going deeper
- Good for finding broad context
- Default strategy

**DFS (Depth-First Search)**:

- Explores as deep as possible before backtracking
- Good for finding specific relationship chains
- Useful for causal reasoning

## Hybrid Retrieval

### HybridRetriever

Combines vector and graph retrieval for comprehensive context gathering.

**Features:**

- Multiple combination strategies
- Configurable weights for each source
- Automatic deduplication
- Metadata tracking for retrieval source

**Configuration:**

```yaml
retriever:
  hybrid:
    mode: "concat" # Combination strategy
    vector_weight: 0.6 # Weight for vector results
    graph_weight: 0.4 # Weight for graph results
    deduplicate: true # Remove duplicate documents
```

**Usage:**

```python
from ragdoll.retrieval import HybridRetriever, VectorRetriever, GraphRetriever

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    graph_retriever=graph_retriever,
    mode="rerank",
    vector_weight=0.6,
    graph_weight=0.4
)

docs = hybrid_retriever.get_relevant_documents("Explain quantum computing")
```

### Combination Modes

#### 1. Concat (Default)

Concatenates vector and graph results, with vector results first.

```python
mode: "concat"
```

**Use case**: When you want all results from both sources, prioritizing vector relevance.

#### 2. Rerank

Re-scores and re-sorts combined results using a unified scoring function.

```python
mode: "rerank"
```

**Scoring:**

- Vector docs: Base score 0.8 × position penalty
- Graph docs: Base score 0.6 × distance decay × relevance

**Use case**: When you want an intelligent blend based on multiple relevance signals.

#### 3. Weighted

Takes a weighted selection from each source based on configured weights.

```python
mode: "weighted"
vector_weight: 0.6
graph_weight: 0.4
```

**Use case**: When you want precise control over the ratio of vector vs. graph results.

#### 4. Expand

Uses vector results as core, with graph results providing related context.

```python
mode: "expand"
```

**Use case**: When vector search finds the primary content and graph adds relational context.

## Complete Example

```python
from ragdoll import Ragdoll
from ragdoll.retrieval import VectorRetriever, GraphRetriever, HybridRetriever

# Initialize RAGdoll
ragdoll = Ragdoll(config_path="config.yaml")

# Ingest documents and build graph
result = ragdoll.ingest_with_graph_sync([
    "docs/architecture.md",
    "docs/api.md"
])

# The hybrid retriever is automatically built
hybrid_retriever = ragdoll.hybrid_retriever

# Query using hybrid retrieval
docs = hybrid_retriever.get_relevant_documents(
    "What are the main components?"
)

# Or use the convenience method
result = ragdoll.query_hybrid(
    "What are the main components?",
    k=10
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['documents'])} documents")
```

## Custom Retriever Composition

You can compose retrievers independently:

```python
from ragdoll.retrieval import VectorRetriever, GraphRetriever, HybridRetriever

# Create specialized vector retriever with MMR
vector_retriever = VectorRetriever(
    vector_store=my_vector_store,
    top_k=10,
    search_type="mmr",
    search_kwargs={"fetch_k": 50}
)

# Create focused graph retriever
graph_retriever = GraphRetriever(
    graph_store=my_graph_store,
    top_k=3,
    max_hops=3,  # Deeper traversal
    traversal_strategy="dfs",  # Depth-first for chains
    min_score=0.5  # Higher threshold
)

# Combine with weighted strategy
hybrid = HybridRetriever(
    vector_retriever=vector_retriever,
    graph_retriever=graph_retriever,
    mode="weighted",
    vector_weight=0.7,
    graph_weight=0.3
)

# Use in your pipeline
docs = hybrid.get_relevant_documents("complex query")
```

## Retriever Statistics

All retrievers provide statistics via `get_stats()`:

```python
# Vector retriever stats
stats = vector_retriever.get_stats()
# {"top_k": 3, "search_type": "similarity", "document_count": 1250}

# Graph retriever stats
stats = graph_retriever.get_stats()
# {"top_k": 5, "max_hops": 2, "traversal_strategy": "bfs",
#  "node_count": 450, "edge_count": 820}

# Hybrid retriever stats
stats = hybrid_retriever.get_stats()
# {"mode": "concat", "deduplicate": true,
#  "vector_stats": {...}, "graph_stats": {...}}
```

## Best Practices

### When to Use Each Retriever

**Vector-Only**:

- Direct semantic similarity is sufficient
- No knowledge graph available
- Performance-critical scenarios

**Graph-Only**:

- Relationship-focused queries
- Exploring entity connections
- Domain with rich structured knowledge

**Hybrid**:

- Complex questions requiring both semantic and relational context
- When you have both vector and graph stores
- General-purpose RAG applications

### Configuration Tips

1. **Start Simple**: Use `mode: "concat"` first, then experiment
2. **Tune Weights**: Adjust `vector_weight` and `graph_weight` based on your domain
3. **Control Depth**: Use `max_hops: 1` for direct relationships, `2-3` for broader context
4. **Optimize Performance**: Reduce `top_k` and `max_hops` if queries are slow
5. **Enable Deduplication**: Always use `deduplicate: true` unless you need raw results

### Performance Considerations

- **Vector retrieval**: Fast (milliseconds), scales with index size
- **Graph traversal**: Moderate (10-100ms), depends on `max_hops` and graph density
- **Hybrid rerank**: Slight overhead for scoring, but improved relevance

## Troubleshooting

### No Graph Results

**Problem**: Graph retriever returns empty results

**Solutions**:

1. Check `graph.enabled: true` in config
2. Verify graph was built: `ragdoll.graph_store` is not None
3. Lower `min_score` threshold
4. Increase `top_k` for more seed nodes

### Duplicate Documents

**Problem**: Same content appears multiple times

**Solutions**:

1. Enable deduplication: `hybrid.deduplicate: true`
2. Check metadata - documents with different metadata are kept
3. Use `mode: "rerank"` which handles duplicates better

### Poor Hybrid Results

**Problem**: Hybrid retrieval seems worse than vector-only

**Solutions**:

1. Adjust weights to favor vector: `vector_weight: 0.8`
2. Try different modes: `mode: "rerank"` or `mode: "expand"`
3. Verify graph quality - run graph statistics
4. Check `include_edges: true` for richer context

## Migration Guide

### From RagdollRetriever

**Old Code**:

```python
from ragdoll.retrievers import RagdollRetriever

retriever = RagdollRetriever(
    vector_store=vs,
    graph_retriever=gr,
    mode="hybrid",
    top_k_vector=5,
    top_k_graph=5
)
```

**New Code**:

```python
from ragdoll.retrieval import VectorRetriever, HybridRetriever

vector_retriever = VectorRetriever(vector_store=vs, top_k=5)
# graph_retriever built automatically by Ragdoll
hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    graph_retriever=graph_retriever,
    mode="concat"
)
```

### From entity_extraction.graph_retriever Config

**Old Config**:

```yaml
entity_extraction:
  graph_retriever:
    enabled: true
    backend: "simple"
```

**New Config**:

```yaml
retriever:
  graph:
    enabled: true
    backend: "networkx"
    max_hops: 2
```

## See Also

- [Entity Extraction](entity_extraction.md) - Building knowledge graphs
- [Vector Stores](vector_stores.md) - Vector database configuration
- [Graph Stores](graph_stores.md) - Graph database backends
- [Examples](examples.md) - Complete retrieval examples
