"""
Evaluation metrics for RAGdoll benchmarks.

Implements retrieval quality metrics and latency statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np


@dataclass
class LatencyStats:
    """Track and compute latency statistics."""

    latencies: List[float] = field(default_factory=list)

    def add(self, latency_ms: float):
        """Add a latency measurement."""
        self.latencies.append(latency_ms)

    def get_stats(self) -> Dict[str, float]:
        """
        Compute latency statistics.

        Returns:
            Dict with mean, median, percentiles, min, max, std
        """
        if not self.latencies:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
            }

        arr = np.array(self.latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "std_ms": float(np.std(arr)),
        }

    def get_throughput(self) -> float:
        """
        Calculate queries per second.

        Returns:
            Throughput in queries/second
        """
        if not self.latencies:
            return 0.0

        total_time_s = sum(self.latencies) / 1000.0  # Convert ms to seconds
        return len(self.latencies) / total_time_s if total_time_s > 0 else 0.0


def compute_perfect_retrieval_rate(
    results: List[Dict[str, any]],
) -> Tuple[float, float]:
    """
    Compute perfect retrieval rate (all evidence retrieved in top-k).

    Args:
        results: List of query results with 'evidence' and 'ground_truth' keys

    Returns:
        Tuple of (overall_rate, multihop_rate)
    """
    perfect_count = 0

    for result in results:
        ground_truth = set(result["ground_truth"])
        predicted = set(result["evidence"])

        # Perfect retrieval = all ground truth evidence retrieved
        if ground_truth.issubset(predicted):
            perfect_count += 1

    rate = perfect_count / len(results) if results else 0.0
    return rate


def compute_recall_at_k(results: List[Dict[str, any]]) -> float:
    """
    Compute average recall@k (what % of evidence was retrieved).

    Args:
        results: List of query results

    Returns:
        Average recall
    """
    recalls = []

    for result in results:
        ground_truth = set(result["ground_truth"])
        predicted = set(result["evidence"])

        if not ground_truth:
            continue

        # Recall = intersection / total ground truth
        recall = len(ground_truth & predicted) / len(ground_truth)
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def compute_mrr(results: List[Dict[str, any]]) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        results: List of query results with 'evidence' and 'ground_truth' keys

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for result in results:
        ground_truth = set(result["ground_truth"])
        predicted = result["evidence"]  # Should be list in order

        # Find first relevant document
        for rank, pred in enumerate(predicted, start=1):
            if pred in ground_truth:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def filter_multihop_results(
    results: List[Dict[str, any]], multihop_questions: Set[str]
) -> List[Dict[str, any]]:
    """
    Filter results to only multihop questions.

    Args:
        results: All results
        multihop_questions: Set of multihop question strings

    Returns:
        Filtered results
    """
    return [r for r in results if r["question"] in multihop_questions]


def compute_all_metrics(
    results: List[Dict[str, any]],
    latency_stats: LatencyStats,
    multihop_questions: Set[str] = None,
) -> Dict[str, any]:
    """
    Compute all evaluation metrics.

    Args:
        results: Query results
        latency_stats: Latency measurements
        multihop_questions: Optional set of multihop questions (deprecated, use multi_hop field in results)

    Returns:
        Dict with all metrics
    """
    metrics = {
        "num_queries": len(results),
        "perfect_retrieval_rate": compute_perfect_retrieval_rate(results),
        "recall_at_k": compute_recall_at_k(results),
        "mrr": compute_mrr(results),
        "latency": latency_stats.get_stats(),
        "throughput_qps": latency_stats.get_throughput(),
    }

    # Add multihop-specific metrics using the multi_hop field in results
    multihop_results = [r for r in results if r.get("multi_hop", False)]
    if multihop_results:
        metrics["multihop"] = {
            "num_queries": len(multihop_results),
            "perfect_retrieval_rate": compute_perfect_retrieval_rate(multihop_results),
            "recall_at_k": compute_recall_at_k(multihop_results),
            "mrr": compute_mrr(multihop_results),
        }

    return metrics


def format_metrics_table(metrics: Dict[str, any], method_name: str) -> str:
    """
    Format metrics as a readable table.

    Args:
        metrics: Computed metrics
        method_name: Name of the method

    Returns:
        Formatted string
    """
    lines = [
        "\n" + "=" * 70,
        f"{method_name} Results",
        "=" * 70,
        f"Queries: {metrics['num_queries']}",
        f"",
        f"RETRIEVAL QUALITY:",
        f"  Perfect Retrieval Rate: {metrics['perfect_retrieval_rate']:.1%}",
        f"  Recall@k:               {metrics['recall_at_k']:.1%}",
        f"  MRR:                    {metrics['mrr']:.3f}",
    ]

    if "multihop" in metrics:
        mh = metrics["multihop"]
        lines.extend(
            [
                f"",
                f"MULTI-HOP ONLY ({mh['num_queries']} queries):",
                f"  Perfect Retrieval Rate: {mh['perfect_retrieval_rate']:.1%}",
                f"  Recall@k:               {mh['recall_at_k']:.1%}",
                f"  MRR:                    {mh['mrr']:.3f}",
            ]
        )

    lat = metrics["latency"]
    lines.extend(
        [
            f"",
            f"LATENCY:",
            f"  Mean:    {lat['mean_ms']:>6.1f} ms",
            f"  Median:  {lat['median_ms']:>6.1f} ms",
            f"  P95:     {lat['p95_ms']:>6.1f} ms",
            f"  P99:     {lat['p99_ms']:>6.1f} ms",
            f"  Min:     {lat['min_ms']:>6.1f} ms",
            f"  Max:     {lat['max_ms']:>6.1f} ms",
            f"",
            f"THROUGHPUT: {metrics['throughput_qps']:.2f} queries/second",
            "=" * 70,
        ]
    )

    return "\n".join(lines)
