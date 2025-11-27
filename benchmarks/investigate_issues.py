"""
Investigate PageRank mode performance and multi-hop detection.
"""

import json
from pathlib import Path
from datasets import load_dataset, get_queries, get_corpus


def check_graph_mode_results():
    """Check what's in the PageRank-only results."""
    print("\n" + "=" * 70)
    print("INVESTIGATING PAGERANK MODE")
    print("=" * 70)

    results_file = Path("results/ragdoll_2wikimultihopqa_51_pagerank.json")
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print(
            "   Run: python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode pagerank --benchmark"
        )
        return

    with open(results_file) as f:
        data = json.load(f)

    print(f"\nTotal queries: {len(data['results'])}")
    print(f"Perfect retrievals: {data['metrics']['perfect_retrieval_rate']:.1%}")
    print(f"Recall@8: {data['metrics']['recall_at_k']:.1%}")

    # Check first few results
    print("\nFirst 3 query results:")
    for i, result in enumerate(data["results"][:3]):
        print(f"\n  Query {i+1}: {result['question'][:80]}...")
        print(f"  Ground truth: {list(result['ground_truth'])[:3]}")

        evidence = result.get("evidence", [])
        if isinstance(evidence, list):
            print(f"  Retrieved: {evidence[:3] if evidence else '[]'}")
        else:
            print(f"  Retrieved: (not a list: {type(evidence)})")

        print(f"  Multi-hop: {result.get('multi_hop', 'N/A')}")

    # Count how many queries returned nothing
    empty_count = sum(1 for r in data["results"] if not r.get("evidence"))
    print(f"\n❌ Queries with no results: {empty_count}/{len(data['results'])}")

    if empty_count == len(data["results"]):
        print("\n⚠️  DIAGNOSIS: PageRank retriever returning nothing for all queries!")
        print("   Possible causes:")
        print("   1. Seed selection not finding relevant chunks")
        print("   2. Graph store not loaded correctly")
        print("   3. Graph has no nodes/edges")
        print("   4. Vector IDs not linking correctly")
        print("\n   Try: python diagnose_graph.py to check graph structure")


def check_dataset_multi_hop():
    """Check if dataset has multi-hop flags."""
    print("\n" + "=" * 70)
    print("INVESTIGATING MULTI-HOP DETECTION")
    print("=" * 70)

    datasets_dir = Path(__file__).parent / "datasets"

    # Load a small sample
    dataset = load_dataset("2wikimultihopqa", datasets_dir, subset=10)
    queries = get_queries(dataset)

    print(f"\nLoaded {len(queries)} queries")

    # Check for type field
    multi_hop_count = 0
    types = {}

    for query in queries:
        qtype = query.query_type
        types[qtype] = types.get(qtype, 0) + 1

        if query.is_multihop:
            multi_hop_count += 1

    print(f"\nQuery types found:")
    for qtype, count in types.items():
        print(f"  {qtype}: {count}")

    print(f"\nMulti-hop queries: {multi_hop_count}/{len(queries)}")

    if multi_hop_count > 0:
        print("\n✅ Dataset HAS multi-hop information (type field)")
        print("✅ Query objects now extract multi-hop flags")
        print("\n   Next: Re-run benchmarks to populate multi-hop metrics")
    else:
        print("\n⚠️  No multi-hop queries found in sample")

    # Show example
    print("\nExample query:")
    example = queries[0]
    print(f"  Question: {example.question[:100]}...")
    print(f"  Type: {example.query_type}")
    print(f"  Is multi-hop: {example.is_multihop}")
    print(f"  Evidence count: {len(example.evidence)}")


def check_existing_results():
    """Check if existing results have multi-hop data."""
    print("\n" + "=" * 70)
    print("CHECKING EXISTING RESULTS")
    print("=" * 70)

    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ No results directory found")
        return

    result_files = list(results_dir.glob("*.json"))
    print(f"\nFound {len(result_files)} result files")

    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            continue

        has_multi_hop = "multi_hop" in results[0]
        multi_hop_count = sum(1 for r in results if r.get("multi_hop", False))

        print(f"\n{result_file.name}:")
        print(f"  Has multi_hop field: {has_multi_hop}")
        if has_multi_hop:
            print(f"  Multi-hop queries: {multi_hop_count}/{len(results)}")
        else:
            print(f"  ⚠️  Missing multi_hop field - re-run benchmark to add it")


if __name__ == "__main__":
    check_graph_mode_results()
    check_dataset_multi_hop()
    check_existing_results()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("\n1. Multi-hop detection is now fixed - re-run benchmarks:")
    print(
        "   python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --benchmark"
    )
    print("\n2. For PageRank mode diagnosis, run:")
    print("   python diagnose_graph.py")
    print("\n3. Then regenerate comparison:")
    print("   python compare.py -d 2wikimultihopqa -n 51")
