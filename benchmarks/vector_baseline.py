"""
Vector baseline benchmark using Chroma + OpenAI embeddings.

Provides a fair comparison baseline for pure semantic search.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import xxhash
from tqdm import tqdm
from dotenv import load_dotenv

from datasets import (
    load_dataset,
    get_corpus,
    get_queries,
    get_multihop_questions,
    estimate_costs,
)
from metrics import LatencyStats, compute_all_metrics, format_metrics_table

from ragdoll.embeddings import get_embedding_model
from ragdoll.vector_stores import vector_store_from_config
from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.retrieval import VectorRetriever

# Suppress verbose HTTP logging from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def create_vector_index(
    corpus: Dict[int, tuple],
    working_dir: Path,
) -> Dict[str, Any]:
    """
    Create vector-only index.

    Args:
        corpus: Dict mapping hash to (title, text)
        working_dir: Directory for persisting data

    Returns:
        Dict with vector_store, embeddings
    """
    print(f"\nCreating vector index...")

    # Prepare documents
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=f"{title}\n\n{text}", metadata={"title": title, "id": hash_id}
        )
        for hash_id, (title, text) in corpus.items()
    ]

    # Create embeddings
    embeddings = get_embedding_model()

    # Configure vector store
    vector_config = VectorStoreConfig(
        enabled=True,
        store_type="chroma",
        params={
            "collection_name": "vector_baseline",
            "persist_directory": str(working_dir / "vector"),
        },
    )

    vector_store = vector_store_from_config(vector_config, embedding=embeddings)

    # Add documents
    print(f"Adding {len(documents)} documents...")
    start = time.time()
    vector_store.add_documents(documents)
    duration = time.time() - start

    print(f"✅ Index created in {duration:.1f}s")

    return {
        "vector_store": vector_store,
        "embeddings": embeddings,
        "ingestion_time": duration,
    }


async def benchmark_queries(
    queries: List[Any],
    retriever: VectorRetriever,
    corpus: Dict[int, tuple],
    top_k: int = 8,
) -> tuple:
    """
    Run benchmark queries and track latency.

    Args:
        queries: List of Query objects
        retriever: VectorRetriever instance
        corpus: Corpus mapping for looking up titles
        top_k: Number of results to retrieve

    Returns:
        Tuple of (results, latency_stats)
    """
    results = []
    latency_stats = LatencyStats()

    print(f"\nRunning {len(queries)} queries...")

    for query in tqdm(queries, desc="Queries"):
        start = time.perf_counter()

        # Run retrieval
        docs = retriever.get_relevant_documents(query.question)[:top_k]

        latency_ms = (time.perf_counter() - start) * 1000
        latency_stats.add(latency_ms)

        # Extract retrieved titles
        retrieved_titles = []
        for doc in docs:
            title = doc.metadata.get("title") or doc.metadata.get("source", "")
            if title:
                retrieved_titles.append(title)
            elif doc.page_content:
                first_line = doc.page_content.split("\n")[0]
                retrieved_titles.append(first_line[:100])

        results.append(
            {
                "question": query.question,
                "answer": query.answer,
                "evidence": retrieved_titles[:top_k],
                "ground_truth": [e[0] for e in query.evidence],
                "latency_ms": latency_ms,
                "multi_hop": query.is_multihop,  # Add multi-hop flag
            }
        )

    return results, latency_stats


async def run_benchmark(
    dataset_name: str,
    subset: int,
    working_dir: Path,
    create: bool = False,
    benchmark: bool = False,
):
    """
    Run complete benchmark workflow.

    Args:
        dataset_name: Dataset name
        subset: Subset size
        working_dir: Working directory
        create: Whether to create index
        benchmark: Whether to run benchmark
    """
    datasets_dir = Path(__file__).parent / "datasets"

    # Load dataset
    print(f"\nLoading dataset: {dataset_name} (subset={subset})")
    dataset = load_dataset(dataset_name, datasets_dir, subset)
    corpus = get_corpus(dataset, dataset_name)
    queries = get_queries(dataset)

    print(f"Loaded {len(corpus)} passages, {len(queries)} queries")

    # Estimate costs
    costs = estimate_costs(len(corpus), len(queries), extract_entities=False)

    print(f"\n💰 Estimated costs:")
    print(f"   Embeddings: ${costs['embedding_cost']:.3f}")
    print(f"   Total: ${costs['total_cost']:.3f}")

    # Create index
    if create:
        index_data = await create_vector_index(corpus, working_dir)

        # Save metadata
        metadata = {
            "dataset": dataset_name,
            "subset": subset,
            "mode": "vector_baseline",
            "num_passages": len(corpus),
            "ingestion_time": index_data["ingestion_time"],
        }

        with open(working_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # Run benchmark
    if benchmark:
        # Load index
        embeddings = get_embedding_model()

        vector_config = VectorStoreConfig(
            enabled=True,
            store_type="chroma",
            params={
                "collection_name": "vector_baseline",
                "persist_directory": str(working_dir / "vector"),
            },
        )

        vector_store = vector_store_from_config(vector_config, embedding=embeddings)
        retriever = VectorRetriever(vector_store=vector_store, top_k=8)

        # Run queries
        results, latency_stats = await benchmark_queries(
            queries,
            retriever,
            corpus,
            top_k=8,
        )

        # Load multihop questions if available
        multihop_questions = set(
            get_multihop_questions(dataset_name, subset, datasets_dir)
        )

        # Compute metrics
        metrics = compute_all_metrics(results, latency_stats, multihop_questions)

        # Print results
        print(format_metrics_table(metrics, "Vector Baseline"))

        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        output_file = results_dir / f"vector_baseline_{dataset_name}_{subset}.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "results": results,
                    "metrics": metrics,
                    "config": {
                        "dataset": dataset_name,
                        "subset": subset,
                        "mode": "vector_baseline",
                        "top_k": 8,
                    },
                },
                f,
                indent=2,
            )

        print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Vector Baseline Benchmark")
    parser.add_argument(
        "-d", "--dataset", default="2wikimultihopqa", help="Dataset to use"
    )
    parser.add_argument("-n", type=int, default=51, help="Subset of corpus to use")
    parser.add_argument("-c", "--create", action="store_true", help="Create the index")
    parser.add_argument(
        "-b", "--benchmark", action="store_true", help="Run benchmark queries"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Please set it in .env or environment.")
        return

    # Setup working directory
    working_dir = (
        Path(__file__).parent / "db" / f"vector_baseline_{args.dataset}_{args.n}"
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    asyncio.run(
        run_benchmark(
            args.dataset,
            args.n,
            working_dir,
            create=args.create,
            benchmark=args.benchmark,
        )
    )


if __name__ == "__main__":
    main()
