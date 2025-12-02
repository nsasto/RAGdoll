# RAGdoll Benchmark Results

## Retrieval Quality

| Method | Perfect Retrieval | Recall@8 | MRR |
|--------|------------------|----------|-----|
| Vector Baseline | 25.7% | 60.9% | 0.956 |
| RAGdoll (vector) | 39.6% | 70.0% | 0.979 |
| RAGdoll (pagerank) | 39.6% | 70.0% | 0.979 |
| RAGdoll (hybrid) | 39.6% | 70.0% | 0.983 |

## Multi-Hop Performance

| Method | Perfect Retrieval | Recall@8 | MRR |
|--------|------------------|----------|-----|
| Vector Baseline | 25.7% | 60.9% | 0.956 |
| RAGdoll (vector) | 39.6% | 70.0% | 0.979 |
| RAGdoll (pagerank) | 39.6% | 70.0% | 0.979 |
| RAGdoll (hybrid) | 39.6% | 70.0% | 0.983 |

## Latency (milliseconds)

| Method | Mean | Median | P95 | P99 |
|--------|------|--------|-----|-----|
| Vector Baseline | 318 | 254 | 737 | 992 |
| RAGdoll (vector) | 301 | 253 | 450 | 986 |
| RAGdoll (pagerank) | 557 | 516 | 833 | 1270 |
| RAGdoll (hybrid) | 605 | 528 | 916 | 1214 |

## Throughput

| Method | Queries/sec |
|--------|-------------|
| Vector Baseline | 3.14 |
| RAGdoll (vector) | 3.33 |
| RAGdoll (pagerank) | 1.79 |
| RAGdoll (hybrid) | 1.65 |

## Improvement vs Vector Baseline

| Method | Accuracy Improvement | Latency Overhead |
|--------|---------------------|------------------|
| RAGdoll (vector) | +53.8% | +-5.5% |
| RAGdoll (pagerank) | +53.8% | +75.1% |
| RAGdoll (hybrid) | +53.8% | +90.2% |