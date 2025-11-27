"""Utility to diagnose Recall@8 contributions for benchmark results."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_results(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("results", []), data.get("metrics", {})


def query_recall(entry: Dict[str, Any]) -> Tuple[float, int]:
    ground = set(filter(None, entry.get("ground_truth", [])))
    retrieved = list(filter(None, entry.get("evidence", [])))
    if not ground:
        return 0.0, 0
    overlap = ground.intersection(set(retrieved))
    missing = len(ground.difference(overlap))
    recall = len(overlap) / len(ground)
    duplicates = max(0, len(retrieved) - len(set(retrieved)))
    return recall, duplicates if missing else duplicates


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def inspect_file(path: Path) -> None:
    results, stored_metrics = load_results(path)
    if not results:
        print(f"No results in {path}")
        return

    recalls = []
    duplicate_counts = []
    multi_hop_recalls = []
    per_query = []

    for entry in results:
        recall, duplicates = query_recall(entry)
        recalls.append(recall)
        duplicate_counts.append(duplicates)
        per_query.append(
            {
                "question": entry.get("question", "")[:80],
                "recall": recall,
                "multi_hop": bool(entry.get("multi_hop")),
                "ground_truth": entry.get("ground_truth", []),
                "evidence": entry.get("evidence", []),
            }
        )
        if entry.get("multi_hop"):
            multi_hop_recalls.append(recall)

    mean_recall = sum(recalls) / len(recalls)
    print("=" * 80)
    print(f"File: {path.name}")
    print(f"Queries: {len(results)}")
    if stored_metrics:
        print(
            "Stored Recall@8:",
            stored_metrics.get("recall_at_k", "N/A"),
        )
    print("Computed Recall@8:", format_pct(mean_recall))
    if multi_hop_recalls:
        multi_mean = sum(multi_hop_recalls) / len(multi_hop_recalls)
        print("Multi-hop Recall@8:", format_pct(multi_mean))

    recall_hist = Counter(round(value, 2) for value in recalls)
    print("\nRecall distribution (rounded to 2 decimals):")
    for recall_value, count in recall_hist.most_common():
        print(f"  {recall_value:.2f}: {count}")

    duplicate_total = sum(duplicate_counts)
    print(f"\nTotal duplicate titles in retrieved evidence: {duplicate_total}")

    low_recall = [item for item in per_query if item["recall"] < 1.0]
    low_recall.sort(key=lambda item: item["recall"])

    if low_recall:
        print("\nLowest recall queries (up to 5 shown):")
        for entry in low_recall[:5]:
            missing = set(entry["ground_truth"]).difference(set(entry["evidence"]))
            print("-" * 40)
            print(f"Recall: {format_pct(entry['recall'])}")
            print(f"Multi-hop: {entry['multi_hop']}")
            print("Ground truth:", entry["ground_truth"])
            print("Retrieved:", entry["evidence"])
            if missing:
                print("Missing:", sorted(missing))
            print(f"Question: {entry['question']}...")


def resolve_paths(argument: str) -> List[Path]:
    base = Path(argument)
    if base.is_dir():
        return sorted(base.glob("*.json"))
    if base.exists():
        return [base]
    raise FileNotFoundError(f"Path not found: {argument}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Recall@8 contributions.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result files or directories to inspect",
    )
    args = parser.parse_args()

    for item in args.paths:
        try:
            paths = resolve_paths(item)
        except FileNotFoundError as exc:
            print(exc)
            continue
        for path in paths:
            inspect_file(path)


if __name__ == "__main__":
    main()
