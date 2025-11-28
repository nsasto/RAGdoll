# RAGdoll Benchmark Results

## Retrieval Quality

| Method | Perfect Retrieval | Recall@8 | MRR |
|--------|------------------|----------|-----|
| Vector Baseline | 46.5% | 74.5% | 0.979 |
| RAGdoll (vector) | 39.6% | 70.0% | 0.979 |
| RAGdoll (pagerank) | 39.6% | 70.0% | 0.979 |
| RAGdoll (hybrid) | 39.6% | 70.0% | 0.983 |

## Multi-Hop Performance

| Method | Perfect Retrieval | Recall@8 | MRR |
|--------|------------------|----------|-----|
| Vector Baseline | 46.5% | 74.5% | 0.979 |
| RAGdoll (vector) | 39.6% | 70.0% | 0.979 |
| RAGdoll (pagerank) | 39.6% | 70.0% | 0.979 |
| RAGdoll (hybrid) | 39.6% | 70.0% | 0.983 |

## Latency (milliseconds)

| Method | Mean | Median | P95 | P99 |
|--------|------|--------|-----|-----|
| Vector Baseline | 302 | 236 | 567 | 1160 |
| RAGdoll (vector) | 302 | 258 | 543 | 1189 |
| RAGdoll (pagerank) | 635 | 563 | 902 | 2407 |
| RAGdoll (hybrid) | 662 | 564 | 893 | 2013 |

## Throughput

| Method | Queries/sec |
|--------|-------------|
| Vector Baseline | 3.32 |
| RAGdoll (vector) | 3.31 |
| RAGdoll (pagerank) | 1.57 |
| RAGdoll (hybrid) | 1.51 |

## Improvement vs Vector Baseline

| Method | Accuracy Improvement | Latency Overhead |
|--------|---------------------|------------------|
| RAGdoll (vector) | +-14.9% | +0.1% |
| RAGdoll (pagerank) | +-14.9% | +110.7% |
| RAGdoll (hybrid) | +-14.9% | +119.4% |