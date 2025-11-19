# Retriever Refactoring Summary

## Overview

This refactoring establishes a clean, modular retrieval architecture for RAGdoll with clear separation between graph building and graph querying. All components are LangChain-compatible and follow best practices for composable RAG systems.

## Key Changes

### 1. New Retrieval Module (`ragdoll/retrieval/`)

Created a dedicated retrieval module with four main components:

- **`base.py`**: LangChain-compatible `BaseRetriever` interface
- **`vector.py`**: `VectorRetriever` for semantic similarity search
- **`graph.py`**: `GraphRetriever` with BFS/DFS traversal strategies
- **`hybrid.py`**: `HybridRetriever` combining vector and graph results

### 2. Configuration Updates

#### Unified Retriever Configuration

Consolidated all retrieval settings under a single `retriever:` block in `default_config.yaml`:

```yaml
retriever:
  vector:
    enabled: true
    top_k: 3
    search_type: "similarity"

  graph:
    enabled: true
    backend: "networkx"
    top_k: 5
    max_hops: 2
    traversal_strategy: "bfs"

  hybrid:
    mode: "concat"
    vector_weight: 0.6
    graph_weight: 0.4
```

#### Removed Legacy Config

- **Deleted**: `entity_extraction.graph_retriever` section
- **Reason**: Mixing graph creation (entity extraction) with graph querying (retrieval) was a design flaw

### 3. ConfigManager Updates

Updated `ragdoll/config/config_manager.py`:

- Added `_ensure_retriever_defaults()` method with comprehensive defaults
- Removed graph_retriever logic from `_ensure_entity_extraction_defaults()`
- Separated concerns: entity extraction builds graphs, retrieval queries them

### 4. Ragdoll Class Updates

Modified `ragdoll/ragdoll.py`:

- Removed dependency on `RagdollRetriever` (old implementation)
- Added new methods:
  - `_build_graph_retriever()`: Creates `GraphRetriever` from graph store
  - `_build_retriever()`: Creates `HybridRetriever` with vector and graph components
- Updated type hints to use new retriever classes
- Improved separation between graph building and retrieval

### 5. Package Exports

Updated `ragdoll/__init__.py` to export new retrievers:

```python
from .retrieval import (
    BaseRetriever,
    VectorRetriever,
    GraphRetriever,
    HybridRetriever,
)
```

### 6. Documentation

Created comprehensive documentation:

- **`docs/retrieval.md`**: Full retrieval architecture guide (300+ lines)

  - Configuration examples
  - Usage patterns for each retriever type
  - Combination strategies (concat, rerank, weighted, expand)
  - Best practices and troubleshooting
  - Migration guide from old implementation

- **Updated `docs/README.md`**: Added retrieval to the documentation index

### 7. Examples

Created `examples/retrieval_examples.py` with:

- Basic hybrid retrieval
- Vector-only retrieval
- Graph-only retrieval
- Custom hybrid configuration
- Strategy comparison
- Async retrieval

## Benefits

### 1. Clean Architecture

- **Separation of Concerns**: Graph building vs. graph querying are distinct
- **Composability**: Retrievers can be mixed and matched
- **Extensibility**: Easy to add new retrieval strategies

### 2. LangChain Compatibility

All retrievers implement standard LangChain interface:

```python
documents = retriever.get_relevant_documents(query)
documents = await retriever.aget_relevant_documents(query)
```

### 3. True Graph RAG

The new `GraphRetriever` provides proper graph RAG capabilities:

- Multi-hop traversal (BFS/DFS)
- Relationship-aware context
- Score decay by hop distance
- Entity expansion from seeds

Previous implementation was just keyword matching on graph nodes.

### 4. Flexible Hybrid Strategies

Four combination modes in `HybridRetriever`:

1. **Concat**: Simple concatenation (vector first)
2. **Rerank**: Unified scoring and re-sorting
3. **Weighted**: Proportional selection from each source
4. **Expand**: Vector core + graph context

### 5. Alpha-Friendly

Since RAGdoll is in alpha:

- No backward compatibility constraints
- Clean break from legacy implementation
- Updated docs and examples reflect new structure

## Migration Path

### For Users

**Old Code**:

```python
from ragdoll.retrievers import RagdollRetriever
retriever = RagdollRetriever(...)
```

**New Code**:

```python
from ragdoll import HybridRetriever
retriever = ragdoll.hybrid_retriever  # Auto-configured
```

### For Configuration

**Old Config**:

```yaml
entity_extraction:
  graph_retriever:
    enabled: false
```

**New Config**:

```yaml
retriever:
  graph:
    enabled: true
    max_hops: 2
```

## Files Changed

### Created (7 files)

- `ragdoll/retrieval/__init__.py`
- `ragdoll/retrieval/base.py`
- `ragdoll/retrieval/vector.py`
- `ragdoll/retrieval/graph.py`
- `ragdoll/retrieval/hybrid.py`
- `docs/retrieval.md`
- `examples/retrieval_examples.py`

### Modified (5 files)

- `ragdoll/config/default_config.yaml`
- `ragdoll/config/config_manager.py`
- `ragdoll/ragdoll.py`
- `ragdoll/__init__.py`
- `docs/README.md`

## Next Steps

### Recommended Follow-ups

1. **Remove `ragdoll/retrievers/`**: The old `RagdollRetriever` implementation can be deprecated
2. **Update Tests**: Create test suite for new retrieval module
3. **Performance Benchmarks**: Compare new graph traversal vs. old keyword matching
4. **Advanced Strategies**: Implement additional hybrid modes (e.g., learned reranking)
5. **Graph Store Methods**: Ensure all graph stores implement required methods for `GraphRetriever`

### Optional Enhancements

- **Caching**: Add retrieval result caching
- **Metrics**: Track retrieval latency and quality
- **Embeddings for Graph**: Use embeddings instead of keywords for seed selection
- **Similarity Threshold**: Add configurable relevance thresholds
- **Result Explanation**: Add metadata explaining why documents were retrieved

## Testing Checklist

Before considering this complete:

- [ ] Run existing tests to ensure no regressions
- [ ] Test vector retrieval with different search types
- [ ] Test graph retrieval with BFS and DFS
- [ ] Test all hybrid combination modes
- [ ] Verify configuration loading and defaults
- [ ] Test with real documents and queries
- [ ] Verify LangChain compatibility
- [ ] Check async retrieval works correctly
- [ ] Validate deduplication logic
- [ ] Test with missing graph store (vector-only fallback)

## Design Decisions

### Why Not Extend RagdollRetriever?

The old `RagdollRetriever` had several issues:

1. Tight coupling to specific graph retriever implementations
2. Mixed vector/graph logic in single class
3. Not truly LangChain-compatible
4. Limited flexibility for different combination strategies

The new architecture with separate `VectorRetriever`, `GraphRetriever`, and `HybridRetriever` provides better:

- **Testability**: Each component can be tested independently
- **Reusability**: Can use retrievers separately or combined
- **Maintainability**: Clear responsibilities for each class

### Why Remove entity_extraction.graph_retriever?

This config was a "legacy stopgap" that mixed concerns:

- **Entity extraction** should focus on building graphs
- **Retrieval** should focus on querying graphs

Having graph_retriever config under entity_extraction implied that retrieval is part of graph creation, which is architecturally incorrect.

### Why Four Hybrid Modes?

Different use cases need different strategies:

- **Concat**: Fast, simple, predictable
- **Rerank**: Intelligent blending of sources
- **Weighted**: Precise control over source ratio
- **Expand**: Vector precision + graph context

This gives users flexibility without overwhelming them (concat is default).

## Conclusion

This refactoring establishes a solid foundation for retrieval in RAGdoll. The new architecture:

- ✅ Separates graph building from graph querying
- ✅ Provides LangChain compatibility
- ✅ Implements true graph RAG with traversal
- ✅ Offers flexible hybrid strategies
- ✅ Has comprehensive documentation
- ✅ Includes practical examples

The codebase is now better positioned for future enhancements while maintaining clarity and maintainability.
