"""
Compare benchmark results and generate summary reports.
"""

import json
from pathlib import Path
from typing import Dict, List
import argparse


def load_result(file_path: Path) -> Dict:
    """Load a benchmark result file."""
    with open(file_path, "r") as f:
        return json.load(f)


def generate_comparison_table(results: Dict[str, Dict]) -> str:
    """
    Generate markdown comparison table.

    Args:
        results: Dict mapping method name to result data

    Returns:
        Markdown formatted table
    """
    lines = [
        "# RAGdoll Benchmark Results",
        "",
        "## Retrieval Quality",
        "",
        "| Method | Perfect Retrieval | Recall@8 | MRR |",
        "|--------|------------------|----------|-----|",
    ]

    for method, data in results.items():
        metrics = data["metrics"]
        lines.append(
            f"| {method} | "
            f"{metrics['perfect_retrieval_rate']:.1%} | "
            f"{metrics['recall_at_k']:.1%} | "
            f"{metrics['mrr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Multi-Hop Performance",
            "",
            "| Method | Perfect Retrieval | Recall@8 | MRR |",
            "|--------|------------------|----------|-----|",
        ]
    )

    for method, data in results.items():
        metrics = data["metrics"]
        if "multihop" in metrics:
            mh = metrics["multihop"]
            lines.append(
                f"| {method} | "
                f"{mh['perfect_retrieval_rate']:.1%} | "
                f"{mh['recall_at_k']:.1%} | "
                f"{mh['mrr']:.3f} |"
            )
        else:
            lines.append(f"| {method} | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Latency (milliseconds)",
            "",
            "| Method | Mean | Median | P95 | P99 |",
            "|--------|------|--------|-----|-----|",
        ]
    )

    for method, data in results.items():
        lat = data["metrics"]["latency"]
        lines.append(
            f"| {method} | "
            f"{lat['mean_ms']:.0f} | "
            f"{lat['median_ms']:.0f} | "
            f"{lat['p95_ms']:.0f} | "
            f"{lat['p99_ms']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Throughput",
            "",
            "| Method | Queries/sec |",
            "|--------|-------------|",
        ]
    )

    for method, data in results.items():
        qps = data["metrics"]["throughput_qps"]
        lines.append(f"| {method} | {qps:.2f} |")

    # Add speedup analysis
    if "Vector Baseline" in results and len(results) > 1:
        baseline_rate = results["Vector Baseline"]["metrics"]["perfect_retrieval_rate"]

        lines.extend(
            [
                "",
                "## Improvement vs Vector Baseline",
                "",
                "| Method | Accuracy Improvement | Latency Overhead |",
                "|--------|---------------------|------------------|",
            ]
        )

        baseline_latency = results["Vector Baseline"]["metrics"]["latency"]["mean_ms"]

        for method, data in results.items():
            if method == "Vector Baseline":
                continue

            rate = data["metrics"]["perfect_retrieval_rate"]
            if baseline_rate > 0:
                improvement = ((rate - baseline_rate) / baseline_rate) * 100
                improvement_str = f"+{improvement:.1f}%"
            else:
                improvement_str = "N/A" if rate == 0 else f"{rate:.1%}"

            latency = data["metrics"]["latency"]["mean_ms"]
            overhead = ((latency - baseline_latency) / baseline_latency) * 100

            lines.append(f"| {method} | " f"{improvement_str} | " f"+{overhead:.1f}% |")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare Benchmark Results")
    parser.add_argument(
        "-d", "--dataset", default="2wikimultihopqa", help="Dataset name"
    )
    parser.add_argument("-n", type=int, default=51, help="Subset size")
    parser.add_argument(
        "-o", "--output", default="BENCHMARK_SUMMARY.md", help="Output file name"
    )

    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"

    # Load all results for this dataset/subset
    results = {}

    # Load vector baseline
    baseline_file = results_dir / f"vector_baseline_{args.dataset}_{args.n}.json"
    if baseline_file.exists():
        results["Vector Baseline"] = load_result(baseline_file)

    # Load RAGdoll results
    mode_labels = {
        "vector": "RAGdoll (vector)",
        "pagerank": "RAGdoll (pagerank)",
        "hybrid": "RAGdoll (hybrid)",
    }

    for mode, label in mode_labels.items():
        result_file = results_dir / f"ragdoll_{args.dataset}_{args.n}_{mode}.json"
        if result_file.exists():
            results[label] = load_result(result_file)

    if not results:
        print(f"⚠️  No results found for {args.dataset} (subset={args.n})")
        print(f"   Looking in: {results_dir}")
        return

    # Generate comparison
    print(f"\n📊 Comparing {len(results)} methods...")

    markdown = generate_comparison_table(results)

    # Save to file
    output_path = results_dir / args.output
    with open(output_path, "w") as f:
        f.write(markdown)

    print(f"✅ Comparison saved to: {output_path}")

    # Also print to console
    print("\n" + markdown)


if __name__ == "__main__":
    main()
