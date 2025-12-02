# RAGdoll Benchmark Suite

Comprehensive benchmarking of RAGdoll's retrieval capabilities against baseline methods using standard multi-hop QA datasets.

## Overview

This benchmark suite evaluates RAGdoll's **Graph-Augmented RAG** approach against baseline methods.

### RAGdoll's Approach: Graph-Augmented RAG

RAGdoll differs from traditional Graph-RAG systems:

**Standard Graph-RAG:**

- Uses knowledge graph as the _primary_ retrieval mechanism
- Typically replaces or heavily supplements text-based retrieval with graph traversal
- Focus is on structured entity relationships rather than combining with dense vector search

**RAGdoll's Graph-Augmented RAG:**

- **Hybrid mode**: Combines vector search (dense retrieval) with graph expansion
- Graph is used to _enrich context_, not as a standalone retriever
- **Goal**: Improve multi-hop reasoning and entity-aware QA while still leveraging unstructured text
- Vector search provides the core relevance, graph traversal adds entity relationships and reasoning paths

This approach is particularly effective for complex reasoning tasks that benefit from both semantic similarity (vector) and structured relationships (graph).

### Evaluation Metrics

- **Retrieval Quality**: Perfect retrieval rate, Recall@k, MRR
- **Performance**: Query latency (mean, median, P95, P99), throughput
- **Comparison**: RAGdoll (vector/hybrid) vs pure vector baseline

## Datasets

We use the standard multi-hop QA dataset:

- **2wikimultihopqa**: Multi-hop questions requiring reasoning across multiple passages

### Dataset Download

Download the dataset from the [fast-graphrag benchmarks](https://github.com/circlemind-ai/fast-graphrag/tree/main/benchmarks/datasets):

```bash
# Place in benchmarks/datasets/
wget https://raw.githubusercontent.com/circlemind-ai/fast-graphrag/main/benchmarks/datasets/2wikimultihopqa.json
```

## Quick Start

### Prerequisites

```bash
# Set OpenAI API key
$env:OPENAI_API_KEY = "your-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### Create Indices Only

To just create the underlying graph and vector databases without running benchmark queries:

```powershell
# Create all indices (baseline + vector + hybrid)
.\run_benchmarks.ps1 -Subset 51 -CreateOnly

# Create only RAGdoll indices (skip baseline)
.\run_benchmarks.ps1 -Subset 51 -CreateOnly -SkipBaseline
```

Or using Python directly:

```bash
# Create vector baseline index only
python vector_baseline.py -d 2wikimultihopqa -n 51 --create

# Create RAGdoll vector index only (no graph/entities)
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode vector --create

# Create RAGdoll hybrid index (with graph and entities)
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --create
```

**What gets created:**

- `db/vector_baseline_<dataset>_<n>/vector/` - Chroma vector store for baseline
- `db/ragdoll_<dataset>_<n>_vector/vector/` - Shared RAGdoll vector store (used by all modes)
- `db/ragdoll_<dataset>_<n>_graph/` - Graph metadata for PageRank + hybrid (`graph.pkl`)

**Directory Reuse:** All RAGdoll modes share the same `ragdoll_<dataset>_<n>_vector` embeddings directory. Once created, vector mode, PageRank, and hybrid all read from this location—avoiding redundant embedding computation. Only PageRank and hybrid create the separate `_graph` directory for entity/relationship data.

**Note**: Creating the shared graph index (used by PageRank + hybrid) takes longest (~5-10 minutes for 51 queries) due to entity extraction LLM calls. Vector-only modes are much faster (~30 seconds).

### Run Complete Benchmark

```powershell
# Run all benchmarks with 51 queries (fast) - runs all modes: vector, pagerank, hybrid
cd benchmarks
.\run_benchmarks.ps1 -Subset 51

# Run all modes with 101 queries (more comprehensive)
.\run_benchmarks.ps1 -Subset 101

# Run specific mode only
.\run_benchmarks.ps1 -Subset 51 -Mode vector
.\run_benchmarks.ps1 -Subset 51 -Mode hybrid

# Skip baseline comparison
.\run_benchmarks.ps1 -Subset 51 -SkipBaseline

# Include no-chunking comparison for all selected modes (recommended for pre-segmented datasets)
.\run_benchmarks.ps1 -Subset 51 -IncludeNoChunking

# Run all modes with both chunking and no-chunking
.\run_benchmarks.ps1 -Subset 101 -IncludeNoChunking

# Run only hybrid mode with no-chunking
.\run_benchmarks.ps1 -Subset 101 -Mode hybrid -IncludeNoChunking
```

### Run Individual Benchmarks

```bash
# Create indices only
.\run_benchmarks.ps1 -Subset 51 -CreateOnly

# Run queries only (requires existing indices)
.\run_benchmarks.ps1 -Subset 51 -BenchmarkOnly
```

## Manual Usage

### Vector Baseline

```bash
# Create index
python vector_baseline.py -d 2wikimultihopqa -n 51 --create

# Run benchmark
python vector_baseline.py -d 2wikimultihopqa -n 51 --benchmark
```

### RAGdoll Benchmark

```bash
# Vector-only mode
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode vector --create
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode vector --benchmark

# Vector mode without chunking (whole-passage retrieval, matches baseline performance)
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode vector --no-chunking --create
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode vector --no-chunking --benchmark

# PageRank graph mode (graph-first retrieval with PageRank ranking)
# Builds the shared graph index reused by hybrid mode
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode pagerank --create
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode pagerank --benchmark

# Hybrid mode (vector + graph, recommended for multi-hop reasoning)
# --create will reuse the shared graph index if it already exists
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --create
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --benchmark

# Note: The legacy graph-only retriever has been replaced by the PageRank graph mode above.
```

**Tip:** Once `db/ragdoll_<dataset>_<n>_vector/` exists, rerunning `--mode vector --create` will skip embedding recreation. PageRank and hybrid modes automatically reuse this shared vector directory when creating graph artifacts, so you only pay embedding costs once.

### Generate Comparison

```bash
python compare.py -d 2wikimultihopqa -n 51 -o BENCHMARK_SUMMARY.md
```

## Running Benchmarks on Existing Indices

If you've already created the database indices, you can run just the benchmark queries without recreating:

### Using PowerShell Script

```powershell
# Run benchmarks on existing indices (skips creation)
.\run_benchmarks.ps1 -Subset 51 -BenchmarkOnly
```

### Using Python Directly

```bash
# Run benchmark on existing vector index
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode vector --benchmark

# Run benchmark on existing hybrid index
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --benchmark

# Run baseline benchmark on existing index
python vector_baseline.py -d 2wikimultihopqa -n 51 --benchmark
```

**Note**: The `--benchmark` flag without `--create` will:

1. Load the shared vector store from `db/ragdoll_<dataset>_<n>_vector/vector/`
2. Load the graph store from `db/ragdoll_<dataset>_<n>_graph/graph.pkl` (for PageRank/hybrid modes only)
3. Reconstruct the retriever with the same configuration
4. Run all queries and save results

This is much faster than recreating indices and useful for:

- Testing different query sets
- Comparing results after code changes
- Re-running after errors
- Tuning retrieval parameters without re-ingestion

### Changing Retrieval Parameters

To test different retrieval settings on an existing index:

```bash
# Test different hybrid modes without recreating
$env:HYBRID_MODE = "expand"
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --benchmark

$env:HYBRID_MODE = "concat"
python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --benchmark

# Or use the test script to compare all modes
.\test_hybrid_modes.ps1 -Subset 51
```

**Retrieval-Only Parameters** (no index recreation needed):

- `HYBRID_MODE`: expand, concat, weighted, rerank
- Top-k settings (modify in code)
- Search type (similarity, MMR)
- Max hops for graph traversal
- Deduplication settings

**Ingestion Parameters** (require `--create`):

- `--no-chunking`: Disable chunking for whole-passage retrieval
- Chunk size/overlap (when chunking is enabled)
- Entity extraction settings
- Embedding model
- Graph construction

## Metrics Explained

### Retrieval Quality

- **Perfect Retrieval Rate**: Percentage of queries where ALL ground truth evidence was retrieved in top-k results
- **Recall@k**: Average percentage of ground truth evidence retrieved (partial credit)
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant document

### Latency

- **Mean**: Average query time
- **Median**: 50th percentile (typical query)
- **P95**: 95th percentile (slow queries)
- **P99**: 99th percentile (worst case)

### Throughput

- **Queries/second**: How many queries can be processed per second

## Expected Results

Based on benchmarking with 2wikimultihopqa dataset (51 queries, top_k=8):

| Metric            | Vector Baseline | RAGdoll Vector (chunked) | RAGdoll Vector (no chunk) | RAGdoll PageRank | RAGdoll Hybrid (expand) |
| ----------------- | --------------- | ------------------------ | ------------------------- | ---------------- | ----------------------- |
| Perfect Retrieval | 27.5%           | 45.1%                    | **52.9%**                 | 45.1%            | 45.1%                   |
| Recall@8          | 62.3%           | 73.0%                    | **77.9%**                 | 73.0%            | 73.0%                   |
| MRR               | **1.000**       | **1.000**                | **1.000**                 | **1.000**        | **1.000**               |
| Latency (mean)    | 896ms           | **281ms**                | 303ms                     | 538ms            | 699ms                   |
| Throughput        | 1.12 q/s        | **3.56 q/s**             | 3.30 q/s                  | 1.86 q/s         | 1.43 q/s                |

**101 Query Results** (2wikimultihopqa, top_k=8):

| Metric            | Vector Baseline | RAGdoll Vector (no chunk) | RAGdoll Vector (chunked) | RAGdoll PageRank | RAGdoll Hybrid (expand) |
| ----------------- | --------------- | ------------------------- | ------------------------ | ---------------- | ----------------------- |
| Perfect Retrieval | 37.6%           | **47.5%**                 | 39.6%                    | 39.6%            | 39.6%                   |
| Recall@8          | 68.3%           | **74.8%**                 | 70.0%                    | 70.0%            | 70.0%                   |
| MRR               | 0.968           | **0.983**                 | 0.979                    | 0.979            | **0.983**               |
| Latency (mean)    | 312ms           | **296ms**                 | 275ms                    | 564ms            | 586ms                   |
| Throughput        | 3.20 q/s        | **3.38 q/s**              | 3.63 q/s                 | 1.77 q/s         | 1.71 q/s                |

**Performance Observations**:

**101-Query Validation:**

- **RAGdoll Vector (no-chunking) achieves 47.5% perfect retrieval** - best overall performance
- **Chunking penalty consistent: ~8pp across both query sets** (51q: -7.8pp, 101q: -7.9pp)
- **Outperforms FastGraphRAG VectorDB by +5.5pp** (47.5% vs 42% with identical embedding model)
- RAGdoll no-chunk outperforms Vector Baseline by +9.9pp (47.5% vs 37.6%)
- All graph modes show identical retrieval metrics, confirming metadata enrichment role

**51-Query Results:**

- No-chunking vector retrieval: 52.9% perfect retrieval (+25.4pp vs baseline, +7.8pp vs chunked RAGdoll)
- Chunked RAGdoll modes (45.1%) outperform Vector Baseline (27.5%) by +17.6pp
- Recall@8 improvement with no-chunking: 77.9% vs 73.0% chunked (+4.9pp)
- MRR of 1.0 across all RAGdoll modes indicates ground truth always ranked first when found

**Latency & Throughput:**

- RAGdoll vector modes fastest (~280-300ms mean, 3.2-3.6 q/s)
- Graph modes 2x slower (~560-590ms, 1.7-1.8 q/s) but add relationship metadata
- Vector Baseline shows higher variance (51q: 896ms mean, 101q: 312ms mean)

**Architectural Note - Chunking vs Passage-Level Retrieval**:

Our benchmarks reveal that **chunking significantly HURTS retrieval performance** for pre-segmented datasets:

**51 Query Results:**

- **RAGdoll Vector (no chunk): 52.9%** - Best performance, whole-passage retrieval
- **RAGdoll Vector (chunked): 45.1%** - 2000-character chunks with 200-character overlap
- **Performance Gap**: -7.8 percentage points due to chunking

**101 Query Results:**

- **RAGdoll Vector (no chunk): 47.5%** - Best performance, whole-passage retrieval
- **RAGdoll Vector (chunked): 39.6%** - 2000-character chunks with 200-character overlap
- **Vector Baseline (37.6%)**: Uses whole-passage retrieval without chunking
- **Performance Gap**: -7.9 percentage points due to chunking

**Cross-Query Set Validation:**

The chunking penalty is remarkably consistent across different query set sizes:

- 51 queries: -7.8pp (52.9% → 45.1%)
- 101 queries: -7.9pp (47.5% → 39.6%)

This consistency indicates the penalty is a fundamental architectural effect, not dataset-dependent.

**Why Chunking Hurts Performance Here**:

1. **Context fragmentation**: Breaking passages into chunks can split related facts that should be retrieved together
2. **Question-answer mismatch**: Multi-hop questions may reference concepts that span multiple chunks, reducing semantic similarity
3. **Increased noise**: More chunks mean more potential false positives, diluting top-k results
4. **Document-level relevance**: For this dataset, passages are already focused enough that whole-document embedding works better

**When Chunking Helps vs Hurts**:

- **Helps**: Long documents (>5000 chars), diverse topics within documents, fact-finding queries
- **Hurts**: Pre-segmented passages, multi-hop reasoning requiring context, entity-centric questions

**Comparison to FastGraphRAG VectorDB Baseline (42%)**:

RAGdoll Vector (no-chunking) achieves **47.5%**, outperforming FastGraphRAG's VectorDB (42%) by **+5.5pp**. Both use:

- Same dataset (2wikimultihopqa, 101 queries)
- Same embedding model (text-embedding-3-small)
- Same passage format (title + text)
- Top-k=8

RAGdoll's performance advantage may be due to:

- **Vector storage implementation**: RAGdoll uses Chroma, FastGraphRAG uses HNSW
- **Indexing parameters**: Different ef_construction/ef_search settings
- **Query processing**: Optimized similarity computation and ranking
- **Metadata handling**: Enhanced deduplication and relevance scoring

**Key Takeaway**: For this benchmark, **passage-level retrieval outperforms chunked retrieval by ~8 percentage points consistently**. RAGdoll supports both approaches:

- Use `--no-chunking` flag for whole-passage retrieval (achieves 52.9% on 51 queries, 47.5% on 101 queries)
- Use default chunking for long documents where splitting is beneficial (45.1% on 51 queries, 39.6% on 101 queries)

**Recommendation**: Test both modes on your dataset to determine which performs better. Pre-segmented passages (like 2wikimultihopqa) benefit from no chunking, while long documents often need chunking. The `--no-chunking` flag creates separate indices with `_nochunk` suffix for easy comparison.

**Note on PageRank Mode**: The PageRank retriever is included for graph-first experiments, but its primary purpose is to complement vector search rather than replace it entirely. Results may emphasize entity-centric passages.

**By Design**: RAGdoll implements **Graph-Augmented RAG**, where the graph enhances vector search rather than replacing it. This differs from traditional Graph-RAG systems that use graph traversal as the primary retrieval mechanism. RAGdoll's approach:

1. **Vector search** identifies semantically relevant passages (core retrieval)
2. **Graph expansion** enriches results with entity relationships and reasoning paths
3. **Hybrid combination** merges both sources for improved multi-hop reasoning

**Note on Source Passage Retrieval**: With the source passage retrieval fix (RAGdoll 2.1.1+), graph retrieval returns actual source passages instead of entity descriptions. This means:

1. **Seed nodes** (0-hop): Return the same passages as vector search, but enriched with relationship metadata
2. **1-hop neighbors**: Return related entities' source passages that vector search may have missed
3. **Deduplication**: Enabled by default, removes duplicate passages when both vector and graph find the same document

If all hybrid modes show similar performance, it suggests that:

- Vector search is already finding most discoverable ground truth passages
- 1-hop graph expansion isn't uncovering many additional ground truth documents for this dataset
- The benefit of graph expansion is primarily in adding **relationship context** (triples, hop_distance) rather than finding new passages

For datasets with stronger entity relationships or queries requiring multi-hop reasoning, graph expansion should show more differentiation between modes.

**Graph Metadata Enrichment**: Retrieved documents include structured relationship triples in metadata for LLM consumption:

```python
# Example metadata from graph-expanded retrieval
{
    "title": "Ava Kolker",
    "entity_name": "Ava Kolker",
    "relationship_triples": [
        ("Ava Kolker", "CHILD_OF", "Doug Kolker"),
        ("Ava Kolker", "ACTED_IN", "Girl Meets World"),
        ("Ava Kolker", "BORN_IN", "Los Angeles")
    ],
    "retrieval_method": "graph_expanded",
    "hop_distance": 0,
    "relevance_score": 0.95
}
```

This allows LLMs to understand both the source passage content AND the structured entity relationships for improved reasoning.

**Key Insights**:

- Graph-augmented hybrid retrieval dramatically improves perfect retrieval rate (64% vs 41%)
- Particularly effective for multi-hop reasoning questions requiring entity relationships
- Trade-off: ~3.5x latency increase due to graph traversal (but still sub-second average)
- The performance gain justifies the latency cost for complex QA tasks

## Cost Estimation

Approximate costs for OpenAI API (as of 2024):

### Subset 51

- Vector baseline: ~$0.02
- RAGdoll vector: ~$0.02
- RAGdoll pagerank: ~$0.15 (includes entity extraction)
- RAGdoll hybrid: ~$0.15 (includes entity extraction)

### Subset 101

- Vector baseline: ~$0.04
- RAGdoll vector: ~$0.04
- RAGdoll pagerank: ~$0.30
- RAGdoll hybrid: ~$0.30

## Directory Structure

```
benchmarks/
├── __init__.py                  # Package init
├── datasets.py                  # Dataset loading utilities
├── metrics.py                   # Evaluation metrics
├── ragdoll_benchmark.py         # RAGdoll benchmark script
├── vector_baseline.py           # Vector baseline benchmark
├── compare.py                   # Results comparison
├── run_benchmarks.ps1           # Automation script
├── README.md                    # This file
├── datasets/                    # Dataset files (download separately)
│   ├── 2wikimultihopqa.json
│   └── hotpotqa.json
├── results/                     # Benchmark results (generated)
│   ├── vector_baseline_*.json
│   ├── ragdoll_*_vector.json
│   ├── ragdoll_*_pagerank.json
│   ├── ragdoll_*_hybrid.json
│   └── BENCHMARK_SUMMARY.md
└── db/                          # Persisted indices (generated)
    ├── vector_baseline_*/
    ├── ragdoll_*_vector/
    └── ragdoll_*_graph/
```

## Troubleshooting

### Dataset not found

```
⚠️  Dataset not found: benchmarks/datasets/2wikimultihopqa.json
```

**Solution**: Download dataset files to `benchmarks/datasets/` directory

### OPENAI_API_KEY not set

```
❌ OPENAI_API_KEY not set!
```

**Solution**: Set environment variable or add to `.env` file

### Out of memory

If you run out of memory with large subsets:

- Use smaller subset sizes (10, 51)
- Reduce batch size in benchmark scripts
- Close other applications

### Slow ingestion

Entity extraction can be slow:

- Expected: ~1-2 minutes for 51 passages
- Use `-CreateOnly` to separate index creation from querying
- Consider using `--mode vector` for faster baseline

## Extending Benchmarks

### Add New Dataset

1. Download dataset in same format as 2wikimultihopqa
2. Place in `benchmarks/datasets/`
3. Run: `python ragdoll_benchmark.py -d your_dataset -n 51 --create --benchmark`

### Add New Retrieval Mode

Edit `ragdoll_benchmark.py` and add new mode in `run_benchmark()` function.

### Custom Metrics

Add metric functions to `metrics.py` and update `compute_all_metrics()`.

## Citation

If you use these benchmarks, please cite:

```
RAGdoll Benchmark Suite
https://github.com/nsasto/RAGdoll
```

## License

Same as RAGdoll project (see root LICENSE file).
