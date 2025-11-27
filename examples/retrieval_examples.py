"""
RAGdoll Retrieval Examples (self-contained).

Loads the bundled comparison sample, ingests it with vector + graph stores, and
demonstrates each retriever (vector, graph, hybrid, pagerank) with timing and a
simple keyword-based precision/recall proxy. Uses FakeEmbeddings so no API keys
are required.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document

# Ensure local package imports when running from examples/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragdoll import Ragdoll
from ragdoll.settings import bootstrap_app
from ragdoll.pipeline import IngestionOptions


# Globals shared across examples
load_dotenv(ROOT / ".env")
CONFIG_PATH = ROOT / "examples" / "app_config_demo.yaml"
APP_CONFIG = bootstrap_app(str(CONFIG_PATH))
# Ensure pagerank retriever is enabled for the demo
_raw = getattr(APP_CONFIG.config, "_config", {})
_raw.setdefault("retriever", {}).setdefault("pagerank", {})
_raw["retriever"]["pagerank"]["enabled"] = True
SAMPLE_FILE = Path(__file__).parent / "retriever_comparison_sample.txt"


def _build_ragdoll(with_graph: bool = True) -> Ragdoll:
    """Initialize Ragdoll from config and ingest the sample corpus."""
    ragdoll = Ragdoll(app_config=APP_CONFIG)
    if with_graph:
        graph_dir = ROOT / "demo_state"
        graph_dir.mkdir(exist_ok=True)
        ragdoll.ingest_with_graph_sync(
            [str(SAMPLE_FILE)],
            options=IngestionOptions(
                graph_store_options={
                    "store_type": "networkx",
                    "output_file": str(graph_dir / "retrieval_examples_graph.pkl"),
                }
            ),
        )
    else:
        ragdoll.ingest_data([str(SAMPLE_FILE)])
    return ragdoll


def _print_docs(docs: List[Document]) -> None:
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("retrieval_source", "unknown")
        print(f"  {i}. [{source}] {doc.page_content[:120]}...")


def _evaluate_keywords(docs: List[Document], keywords: List[str]) -> Dict[str, float]:
    content = " ".join(doc.page_content for doc in docs).lower()
    found = sum(1 for kw in keywords if kw.lower() in content)
    recall = found / len(keywords) if keywords else 0.0
    precision = found / max(len(docs) * len(keywords), 1)
    return {"precision": precision, "recall": recall}


def example_basic_hybrid():
    print("\n=== Basic Hybrid Retrieval ===")
    ragdoll = _build_ragdoll(with_graph=True)
    query = "What are the main components of RAGdoll?"
    result = ragdoll.query_hybrid(query, k=5)
    print(f"Answer: {result['answer']}")
    print(f"Retrieved {len(result['documents'])} docs:")
    _print_docs(result["documents"])


def example_vector_only():
    print("\n=== Vector Retrieval ===")
    ragdoll = _build_ragdoll(with_graph=False)
    docs = ragdoll.vector_store.similarity_search("How do I configure embeddings?", k=5)
    print(f"Retrieved {len(docs)} docs:")
    _print_docs(docs)


def example_graph_only():
    print("\n=== Graph Retrieval ===")
    ragdoll = _build_ragdoll(with_graph=True)
    docs = ragdoll.graph_retriever.get_relevant_documents(
        "What entities are related to Kubernetes?", top_k=5
    )
    print(f"Retrieved {len(docs)} docs:")
    _print_docs(docs)


def example_hybrid_custom():
    print("\n=== Hybrid Retrieval (custom weights) ===")
    ragdoll = _build_ragdoll(with_graph=True)
    ragdoll.hybrid_retriever.mode = "concat"
    docs = ragdoll.hybrid_retriever.get_relevant_documents(
        "Explain the relationship between graph traversal and vector search", top_k=5
    )
    print(f"Retrieved {len(docs)} docs:")
    _print_docs(docs)


def example_comparison():
    """
    Compare retrievers with timing and keyword coverage (precision/recall proxy).
    """
    print("\n=== Retriever Comparison (vector / graph / hybrid / pagerank) ===")
    ragdoll = _build_ragdoll(with_graph=True)

    test_cases = [
        {
            "query": "Which companies attended the summit?",
            "keywords": ["Microsoft", "Google", "Amazon"],
        },
        {
            "query": "How are Kubernetes and distributed systems connected?",
            "keywords": ["Kubernetes", "distributed systems"],
        },
        {
            "query": "What does the PageRank retriever do?",
            "keywords": ["PageRank", "graph"],
        },
    ]

    for case in test_cases:
        query = case["query"]
        keywords = case["keywords"]
        print(f"\nQuery: {query}")

        timings = []
        # Vector
        start = time.perf_counter()
        v_docs = ragdoll.vector_store.similarity_search(query, k=5)
        timings.append(("vector", time.perf_counter() - start, v_docs))
        # Graph
        start = time.perf_counter()
        g_docs = ragdoll.graph_retriever.get_relevant_documents(query, top_k=5)
        timings.append(("graph", time.perf_counter() - start, g_docs))
        # Hybrid
        start = time.perf_counter()
        h_docs = ragdoll.hybrid_retriever.get_relevant_documents(query, top_k=5)
        timings.append(("hybrid", time.perf_counter() - start, h_docs))
        # PageRank
        p_docs: List[Document] = []
        if ragdoll.pagerank_retriever:
            start = time.perf_counter()
            p_docs = ragdoll.pagerank_retriever.get_relevant_documents(query, top_k=5)
            timings.append(("pagerank", time.perf_counter() - start, p_docs))
        else:
            timings.append(("pagerank", 0.0, []))

        for name, elapsed, docs in timings:
            metrics = _evaluate_keywords(docs, keywords)
            print(
                f"  {name:<8} time={elapsed*1000:6.2f} ms  "
                f"precision~={metrics['precision']:.2f}  recall~={metrics['recall']:.2f}  "
                f"docs={len(docs)}"
            )
        print("  Top results (vector / hybrid / pagerank):")
        for label, docs in [
            ("vector", v_docs),
            ("hybrid", h_docs),
            ("pagerank", p_docs),
        ]:
            if docs:
                print(f"    {label:<8}: {docs[0].page_content[:80]}...")


def main():
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"Sample corpus missing at {SAMPLE_FILE}. Add the text file and retry."
        )

    print(
        """
Available examples:
  1 - Basic Hybrid Retrieval
  2 - Vector-Only Retrieval
  3 - Graph-Only Retrieval
  4 - Hybrid Retrieval (custom)
  5 - Retriever Comparison (latency + keyword coverage)
"""
    )

    choice = sys.argv[1] if len(sys.argv) > 1 else "5"
    if choice == "1":
        example_basic_hybrid()
    elif choice == "2":
        example_vector_only()
    elif choice == "3":
        example_graph_only()
    elif choice == "4":
        example_hybrid_custom()
    else:
        print("No param specified. Running default: Retriever Comparison\n")
        example_comparison()


if __name__ == "__main__":
    main()
