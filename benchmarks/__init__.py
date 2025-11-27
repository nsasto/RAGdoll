"""
RAGdoll Benchmark Suite

Compare RAGdoll's retrieval performance against baseline methods using
standard multi-hop QA datasets (2wikimultihopqa, HotpotQA).

Key metrics:
- Perfect retrieval rate (all evidence retrieved)
- Recall@k (percentage of evidence retrieved)
- Mean Reciprocal Rank (MRR)
- Query latency (mean, median, p95, p99)
- Throughput (queries/second)
- Cost estimation (API calls)
"""

__version__ = "0.1.0"
