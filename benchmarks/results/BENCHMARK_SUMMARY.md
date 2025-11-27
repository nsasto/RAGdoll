# RAGdoll Benchmark Results

## Retrieval Quality

| Method | Perfect Retrieval | Recall@8 | MRR |
|--------|------------------|----------|-----|
| Vector Baseline | 27.5% | 62.7% | 1.000 |
| RAGdoll (vector) | 45.1% | 73.0% | 1.000 |
| RAGdoll (pagerank) | 45.1% | 73.0% | 1.000 |
| RAGdoll (hybrid) | 47.1% | 74.0% | 1.000 |

## Multi-Hop Performance

| Method | Perfect Retrieval | Recall@8 | MRR |
|--------|------------------|----------|-----|
| Vector Baseline | 27.5% | 62.7% | 1.000 |
| RAGdoll (vector) | 45.1% | 73.0% | 1.000 |
| RAGdoll (pagerank) | 45.1% | 73.0% | 1.000 |
| RAGdoll (hybrid) | 47.1% | 74.0% | 1.000 |

## Latency (milliseconds)

| Method | Mean | Median | P95 | P99 |
|--------|------|--------|-----|-----|
| Vector Baseline | 341 | 319 | 505 | 904 |
| RAGdoll (vector) | 365 | 301 | 693 | 1130 |
| RAGdoll (pagerank) | 857 | 587 | 1174 | 6405 |
| RAGdoll (hybrid) | 691 | 620 | 1098 | 1685 |

## Throughput

| Method | Queries/sec |
|--------|-------------|
| Vector Baseline | 2.93 |
| RAGdoll (vector) | 2.74 |
| RAGdoll (pagerank) | 1.17 |
| RAGdoll (hybrid) | 1.45 |

## Improvement vs Vector Baseline

| Method | Accuracy Improvement | Latency Overhead |
|--------|---------------------|------------------|
| RAGdoll (vector) | +64.3% | +7.2% |
| RAGdoll (pagerank) | +64.3% | +151.5% |
| RAGdoll (hybrid) | +71.4% | +102.8% |