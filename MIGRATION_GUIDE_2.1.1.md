# Migration Guide: RAGdoll 2.1.0 → 2.1.1

## Breaking Changes: Parallel Execution Refactoring

RAGdoll 2.1.1 refactors parallel execution from the pipeline layer to the vector store layer for better separation of concerns and reusability.

---

## What Changed?

### 1. Configuration Location

**Before (2.1.0):**

```python
from ragdoll.pipeline import IngestionOptions

options = IngestionOptions(
    batch_size=10,
    max_concurrent_embeddings=5,  # ❌ Removed
    max_concurrent_llm_calls=8
)
```

**After (2.1.1+):**

```yaml
# In default_config.yaml or your config file
embeddings:
  max_concurrent_embeddings: 5 # ✅ Moved here
```

```python
from ragdoll.pipeline import IngestionOptions

options = IngestionOptions(
    batch_size=10,
    max_concurrent_llm_calls=8
    # max_concurrent_embeddings removed from here
)
```

### 2. Parallel Execution API

**Before (2.1.0):**

```python
# Internal method, not directly accessible
# Pipeline handled all parallel execution
```

**After (2.1.1+):**

```python
# New public API in BaseVectorStore
from ragdoll.vector_stores import create_vector_store
from ragdoll.embeddings import get_embedding_model

embeddings = get_embedding_model()
vector_store = create_vector_store("faiss", embedding=embeddings)

# Async parallel processing
ids = await vector_store.add_documents_parallel(
    documents=chunks,
    batch_size=10,
    max_concurrent=5
)
```

---

## Migration Scenarios

### Scenario 1: Using Default Ragdoll API ✅ No Changes Needed

If you're using the standard API, no code changes required:

```python
from ragdoll.ragdoll import Ragdoll

ragdoll = Ragdoll()
ragdoll.ingest_data(["document.pdf"])
result = ragdoll.query("What is this about?")
```

**Why it works:** Pipeline automatically uses `config.embeddings_config.max_concurrent_embeddings`

---

### Scenario 2: Custom IngestionOptions ⚠️ Minor Change

**Before:**

```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.pipeline import IngestionOptions

options = IngestionOptions(
    batch_size=20,
    max_concurrent_embeddings=10,  # ❌
    max_concurrent_llm_calls=15
)

ragdoll = Ragdoll()
await ragdoll.ingest_with_graph(sources, options=options)
```

**After:**

```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.pipeline import IngestionOptions

# Set max_concurrent_embeddings in config instead
ragdoll = Ragdoll()
ragdoll.config_manager.embeddings_config.max_concurrent_embeddings = 10

options = IngestionOptions(
    batch_size=20,
    max_concurrent_llm_calls=15
)

await ragdoll.ingest_with_graph(sources, options=options)
```

---

### Scenario 3: Direct Vector Store Usage ✅ New Capabilities

**Before:**

```python
vector_store = create_vector_store("faiss", embedding=embeddings)
ids = vector_store.add_documents(chunks)  # Sequential
```

**After (use parallel for speedup):**

```python
vector_store = create_vector_store("faiss", embedding=embeddings)

# Async parallel processing - 3-5x faster!
ids = await vector_store.add_documents_parallel(
    chunks,
    max_concurrent=5
)
```

---

### Scenario 4: Configuration File Updates

**Before (2.1.0 config):**

```yaml
embeddings:
  enabled: true
  default_model: "text-embedding-3-small"
  models:
    text-embedding-3-small:
      provider: "openai"
      model: "text-embedding-3-small"
```

**After (2.1.1+ config):**

```yaml
embeddings:
  enabled: true
  default_model: "text-embedding-3-small"
  max_concurrent_embeddings: 3 # ✅ Add this line
  models:
    text-embedding-3-small:
      provider: "openai"
      model: "text-embedding-3-small"
```

---

## Step-by-Step Migration

### Step 1: Update Configuration Files

Add `max_concurrent_embeddings` to your embeddings config:

```yaml
embeddings:
  max_concurrent_embeddings: 3 # Conservative default
  # Increase to 5-10 if you have good API rate limits
```

### Step 2: Update Code Using IngestionOptions

Remove `max_concurrent_embeddings` parameter:

```diff
options = IngestionOptions(
    batch_size=20,
-   max_concurrent_embeddings=10,
    max_concurrent_llm_calls=15
)
```

If you need to set it programmatically:

```python
config.embeddings_config.max_concurrent_embeddings = 10
```

### Step 3: Test Your Application

Run your existing tests to ensure everything works:

```bash
python -m pytest tests/
```

### Step 4: (Optional) Use New Async API

For direct vector store operations, consider using the new parallel API:

```python
# Old (still works)
ids = vector_store.add_documents(chunks)

# New (faster)
ids = await vector_store.add_documents_parallel(chunks, max_concurrent=5)
```

---

## Performance Tuning

### Recommended Settings by Use Case

#### Conservative (Rate-limited APIs)

```yaml
embeddings:
  max_concurrent_embeddings: 2
```

#### Balanced (Default)

```yaml
embeddings:
  max_concurrent_embeddings: 3
```

#### Aggressive (Good rate limits)

```yaml
embeddings:
  max_concurrent_embeddings: 8
```

#### Local Embeddings (High throughput)

```yaml
embeddings:
  max_concurrent_embeddings: 10
```

### Testing Performance

Run the performance comparison to see actual speedups:

```bash
python -m pytest tests/test_parallel_performance.py::TestPerformanceComparison::test_side_by_side_comparison -v -s
```

Expected output:

```
Sequential time:     2.50s
Parallel time:       0.83s
Speedup factor:      3.01x
```

---

## Troubleshooting

### Issue: AttributeError on max_concurrent_embeddings

**Error:**

```python
AttributeError: 'IngestionOptions' object has no attribute 'max_concurrent_embeddings'
```

**Solution:**
Remove the parameter from `IngestionOptions()` and set it in config instead:

```python
# Remove this:
options = IngestionOptions(max_concurrent_embeddings=5)

# Use this:
config.embeddings_config.max_concurrent_embeddings = 5
options = IngestionOptions()
```

### Issue: Config not loading max_concurrent_embeddings

**Error:**

```python
KeyError: 'max_concurrent_embeddings'
```

**Solution:**
Add the setting to your YAML config file:

```yaml
embeddings:
  max_concurrent_embeddings: 3
```

Or it will use the default value (3) from `EmbeddingsConfig`.

### Issue: Slower performance after upgrade

**Possible Causes:**

1. Default `max_concurrent_embeddings=3` might be too conservative
2. Your API has rate limits being hit

**Solutions:**

- Increase concurrency: `max_concurrent_embeddings: 5`
- Check API rate limit logs
- Monitor concurrent requests in performance tests

---

## Validation Checklist

- [ ] Updated config files with `max_concurrent_embeddings`
- [ ] Removed `max_concurrent_embeddings` from `IngestionOptions`
- [ ] Tested existing ingestion pipelines
- [ ] Ran unit tests (`pytest tests/`)
- [ ] (Optional) Tested new async API
- [ ] (Optional) Ran performance tests to validate speedup

---

## Benefits of This Change

✅ **Better Architecture**: Parallel logic belongs in storage layer, not orchestration
✅ **Reusability**: Any code can use parallel embeddings, not just pipeline
✅ **Configuration**: Persistent config vs transient runtime options
✅ **Standards**: Aligns with LangChain async patterns
✅ **Performance**: Same 3-5x speedup, now more accessible

---

## Need Help?

- Review [PARALLEL_REFACTORING_SUMMARY.md](./PARALLEL_REFACTORING_SUMMARY.md) for technical details
- Check [docs/vector_stores.md](./docs/vector_stores.md) for API documentation
- Run performance tests to validate: `pytest tests/test_parallel_performance.py -v -s`
- Open an issue on GitHub if you encounter problems

---

## Timeline

- **2.1.0**: Initial parallel execution in pipeline (deprecated approach)
- **2.1.1**: Refactored to vector store layer (current, breaking change)
- **2.2.0**: (Future) Additional performance optimizations and monitoring
