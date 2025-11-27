# Source Passage Retrieval Fix

## Problem

RAGdoll's graph retrieval was returning entity descriptions ("PERSON: Einstein\nProperties:...") instead of the original source passages, resulting in benchmark performance of 64% compared to fast-graphrag's 96%.

While graph nodes stored `vector_id` references to source documents, the `_subgraph_to_documents()` method was creating new Document objects with:

- `page_content`: Entity descriptions formatted as "EntityType: EntityName"
- `metadata`: Hardcoded `{"source": "graph_retrieval"}` without passage title

This was incompatible with benchmarks that expect passage titles (e.g., "Ava Kolker") in metadata to match against ground truth.

## Root Cause

Fast-graphrag achieves 96% by explicitly mapping graph nodes back to source corpus:

```python
corpus[chunk.metadata["id"]][0]  # Returns passage title
```

RAGdoll had the `vector_id` stored but never used it to lookup source passage metadata during retrieval.

## Solution

### 1. Added `get_documents_by_ids()` to `VectorStoreAdapter`

New method in `ragdoll/vector_stores/adapter.py` to retrieve full Document objects (not just embeddings) from the vector store:

```python
def get_documents_by_ids(self, vector_ids: List[str]) -> Dict[str, Any]:
    """
    Retrieve full documents (with metadata) for given vector IDs.

    Returns:
        Dict mapping vector_id to {"page_content": str, "metadata": dict}
    """
```

Implementations for:

- **Chroma**: Uses `collection.get(ids=vector_ids, include=["documents", "metadatas"])`
- **FAISS**: Uses `docstore.search(vec_id)` with existing index mapping
- **Generic**: Fallback returns empty dict

### 2. Updated `_subgraph_to_documents()` in `GraphRetriever`

Modified `ragdoll/retrieval/graph.py` lines 720-780 to:

1. **Extract `vector_id`** from node properties
2. **Lookup source document** using `VectorStoreAdapter.get_documents_by_ids()`
3. **Use source passage content** instead of entity description
4. **Preserve source metadata** (including title) while adding graph context

```python
# Before (returns entity description)
doc = Document(
    page_content="PERSON: Einstein\nProperties:...",
    metadata={"source": "graph_retrieval", "node_id": "..."}
)

# After (returns source passage)
doc = Document(
    page_content="Albert Einstein was born in 1879...",  # Original passage
    metadata={
        "title": "Albert Einstein",  # From vector store
        "node_id": "...",
        "entity_name": "Einstein",
        "retrieval_method": "graph_expanded"
    }
)
```

### 3. Fallback Strategy

If `vector_id` is missing or lookup fails:

- Falls back to entity description (current behavior)
- Sets `metadata["source"] = "graph_retrieval"` to distinguish from source passages
- Logs debug message for tracking

## Benefits

1. **Improved Benchmark Performance**: Graph retrieval now returns passages that match ground truth titles
2. **Better Hybrid Mode**: Vector + Graph hybrid can deduplicate properly when both return same passage
3. **Maintains Compatibility**: Fallback ensures existing functionality continues if vector_id unavailable
4. **Adds Context**: Source passages include graph metadata (entity_name, hop_distance, relevance_score)

## Expected Impact

- **Baseline (before fix)**: Hybrid expand mode = 64% perfect retrieval
- **Target (after fix)**: Expected 70-85% perfect retrieval (closing gap toward fast-graphrag's 96%)
- **Remaining gap**: May be due to domain-specific entity extraction (family_role, profession types)

## Testing

Run benchmark with:

```powershell
# Remove old index
Remove-Item -Recurse -Force db\ragdoll_2wikimultihopqa_51_graph

# Test with 25 queries
cd benchmarks
$env:HYBRID_MODE="expand"
$env:QUERY_SUBSET_SIZE="25"
python ragdoll_benchmark.py
```

Verify with diagnostic script:

```powershell
python benchmarks/test_source_passage_fix.py
```

## Files Modified

1. `ragdoll/vector_stores/adapter.py`: Added `get_documents_by_ids()` and backend implementations
2. `ragdoll/retrieval/graph.py`: Modified `_subgraph_to_documents()` to retrieve source passages
3. `benchmarks/test_source_passage_fix.py`: New diagnostic script to verify fix

## Future Optimizations

1. **Caching**: Multiple entities from same document share `vector_id` - cache lookups within single retrieval call
2. **Batch Retrieval**: Currently fetches one vector_id per node - could batch all vector_ids for one call
3. **Domain-Specific Extraction**: Add 2wikimultihopqa entity types (family_role, profession) to close remaining gap to 96%
