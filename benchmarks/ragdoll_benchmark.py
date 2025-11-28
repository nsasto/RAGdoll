"""
RAGdoll benchmark script.

Tests RAGdoll's vector, PageRank graph, and hybrid retrieval modes against
multi-hop QA datasets.
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

from ragdoll.app_config import bootstrap_app
from ragdoll.pipeline import ingest_documents, IngestionOptions
from ragdoll.retrieval import (
    VectorRetriever,
    HybridRetriever,
    PageRankGraphRetriever,
)

# Map retrieval modes to on-disk directory suffixes (graph modes share the same
# persisted graph index, while all modes share the vector store directory).
MODE_WORKDIR_SUFFIX = {
    "vector": "vector",
    "pagerank": "graph",
    "hybrid": "graph",
}

# Suppress verbose HTTP logging from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
from ragdoll.embeddings import get_embedding_model
from ragdoll.llms import get_llm_caller

# Configure logging to show progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


async def create_ragdoll_index(
    corpus: Dict[int, tuple],
    working_dir: Path,
    vector_dir: Path,
    mode: str = "hybrid",
    extract_entities: bool = True,
    no_chunking: bool = False,
) -> Dict[str, Any]:
    """
    Create RAGdoll index with ingestion pipeline.

    Args:
        corpus: Dict mapping hash to (title, text)
        working_dir: Directory for persisting graph metadata
        vector_dir: Directory for shared vector store
        mode: Retrieval mode (vector, graph, hybrid)
        extract_entities: Whether to extract entities
        no_chunking: If True, disable chunking (whole-passage retrieval)

    Returns:
        Dict with vector_store, graph_store, embeddings
    """
    print(f"\nCreating RAGdoll index (mode={mode}, entities={extract_entities})...")

    # Prepare documents
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=f"{title}\n\n{text}", metadata={"title": title, "id": hash_id}
        )
        for hash_id, (title, text) in corpus.items()
    ]

    # Get LLM caller if entities are needed
    llm_caller = None
    if extract_entities:
        llm_caller = get_llm_caller()
        if llm_caller is None:
            print("⚠️  Unable to initialize LLM, entity extraction may fail")

    # Ensure directories exist before ingestion
    vector_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Configure ingestion
    options = IngestionOptions(
        batch_size=20,
        max_concurrent_llm_calls=15,  # Increase concurrency for faster entity extraction
        extract_entities=extract_entities,
        skip_graph_store=not extract_entities,  # Skip graph store when not extracting entities
        collect_metrics=False,  # Disable metrics collection for clean benchmarks
        chunking_options={
            "chunking_strategy": "none" if no_chunking else "recursive",
            "chunk_size": 2000,  # Larger chunks = fewer LLM calls
            "chunk_overlap": 200,
        },
        vector_store_options={
            "store_type": "chroma",
            "params": {
                "collection_name": "ragdoll_benchmark",
                "persist_directory": str(vector_dir / "vector"),
            },
        },
        llm_caller=llm_caller,
    )

    if extract_entities:
        options.graph_store_options = {
            "store_type": "networkx",
            "output_file": str(working_dir / "graph.pkl"),
        }
        options.entity_extraction_options = {
            "entity_types": ["Person", "Organization", "Location", "Event"],
            "use_llm": True,
        }

    # Pass documents directly to ingestion (no need to write files)
    start = time.time()
    result = await ingest_documents(documents, options=options)
    duration = time.time() - start

    print(f"✅ Ingestion complete in {duration:.1f}s")
    stats = result.get("stats", {})
    print(f"   Documents: {stats.get('documents_processed')}")
    print(f"   Chunks: {stats.get('chunks_created')}")
    if extract_entities:
        print(f"   Entities: {stats.get('entities_extracted')}")
        print(f"   Relationships: {stats.get('relationships_extracted')}")

    return {
        "vector_store": result.get("vector_store"),
        "graph_store": result.get("graph_store"),
        "embeddings": get_embedding_model(),
        "ingestion_time": duration,
        "stats": stats,
    }


def index_exists(vector_dir: Path, working_dir: Path, extract_entities: bool) -> bool:
    """Check whether an index already exists for the requested configuration."""

    vector_collection = vector_dir / "vector"
    if not vector_collection.exists():
        return False

    if extract_entities:
        return (working_dir / "graph.pkl").exists()

    return True


async def benchmark_queries(
    queries: List[Any],
    retriever: Any,
    corpus: Dict[int, tuple],
    top_k: int = 8,
) -> tuple:
    """
    Run benchmark queries and track latency.

    Args:
        queries: List of Query objects
        retriever: Retriever instance
        corpus: Corpus mapping for looking up titles
        top_k: Number of results to retrieve

    Returns:
        Tuple of (results, latency_stats)
    """
    results = []
    latency_stats = LatencyStats()

    # Create reverse lookup: title -> hash
    title_to_hash = {title: hash_id for hash_id, (title, _) in corpus.items()}

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
            # Try to get title from metadata
            title = doc.metadata.get("title") or doc.metadata.get("source", "")
            if title:
                retrieved_titles.append(title)
            # Fallback: extract from content
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
    mode: str,
    working_dir: Path,
    vector_dir: Path,
    create: bool = False,
    benchmark: bool = False,
    no_chunking: bool = False,
):
    """
    Run complete benchmark workflow.

    Args:
        dataset_name: Dataset name
        subset: Subset size
        mode: Retrieval mode (vector, graph, hybrid)
        working_dir: Directory used for mode-specific artifacts (graph metadata)
        vector_dir: Directory containing the shared vector store
        create: Whether to create index
        benchmark: Whether to run benchmark
        no_chunking: If True, disable chunking for whole-passage retrieval
    """
    datasets_dir = Path(__file__).parent / "datasets"

    # Load dataset
    print(f"\nLoading dataset: {dataset_name} (subset={subset})")
    dataset = load_dataset(dataset_name, datasets_dir, subset)
    corpus = get_corpus(dataset, dataset_name)
    queries = get_queries(dataset)

    print(f"Loaded {len(corpus)} passages, {len(queries)} queries")

    # Estimate costs
    extract_entities = mode in ["pagerank", "hybrid"]
    costs = estimate_costs(len(corpus), len(queries), extract_entities=extract_entities)

    print(f"\n💰 Estimated costs:")
    print(f"   Embeddings: ${costs['embedding_cost']:.3f}")
    if extract_entities:
        print(f"   Entity extraction: ${costs['entity_extraction_cost']:.3f}")
    print(f"   Total: ${costs['total_cost']:.3f}")

    metadata_path = working_dir / "metadata.json"

    # Create index
    if create:
        if index_exists(vector_dir, working_dir, extract_entities):
            print(
                f"\nℹ️  Existing index detected at {working_dir}. "
                "Skipping ingestion. Delete the directory to rebuild."
            )
            if not metadata_path.exists():
                metadata = {
                    "dataset": dataset_name,
                    "subset": subset,
                    "mode": mode,
                    "num_passages": len(corpus),
                    "ingestion_time": None,
                    "stats": {},
                    "vector_directory": str(vector_dir),
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
        else:
            index_data = await create_ragdoll_index(
                corpus,
                working_dir,
                vector_dir,
                mode=mode,
                extract_entities=extract_entities,
            )

            # Save index metadata
            metadata = {
                "dataset": dataset_name,
                "subset": subset,
                "mode": mode,
                "num_passages": len(corpus),
                "ingestion_time": index_data["ingestion_time"],
                "stats": index_data["stats"],
                "vector_directory": str(vector_dir),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    # Run benchmark
    if benchmark:
        # Load index
        from ragdoll.vector_stores import vector_store_from_config
        from ragdoll.config.base_config import VectorStoreConfig

        embeddings = get_embedding_model()

        vector_config = VectorStoreConfig(
            enabled=True,
            store_type="chroma",
            params={
                "collection_name": "ragdoll_benchmark",
                "persist_directory": str(vector_dir / "vector"),
            },
        )

        from ragdoll.vector_stores import vector_store_from_config

        vector_store = vector_store_from_config(vector_config, embedding=embeddings)

        # Create retriever based on mode
        if mode == "vector":
            retriever = VectorRetriever(vector_store=vector_store, top_k=8)
        elif mode == "pagerank":
            # Load graph store
            from ragdoll.graph_stores import GraphStoreWrapper
            import pickle

            with open(working_dir / "graph.pkl", "rb") as f:
                nx_graph = pickle.load(f)

            graph_store = GraphStoreWrapper(
                store_type="networkx", store_impl=nx_graph, config={}
            )

            retriever = PageRankGraphRetriever(
                graph_store=graph_store,
                vector_store=vector_store,
                embedding_model=embeddings,
                top_k=8,
            )
        else:  # hybrid
            # Load graph store
            from ragdoll.graph_stores import GraphStoreWrapper
            import pickle

            with open(working_dir / "graph.pkl", "rb") as f:
                nx_graph = pickle.load(f)

            graph_store = GraphStoreWrapper(
                store_type="networkx", store_impl=nx_graph, config={}
            )

            # Use "expand" mode - vector results seed graph expansion for best passage retrieval
            # Testing showed: expand=64%, concat=64%, rerank=56%, weighted=52%
            hybrid_mode = os.getenv("HYBRID_MODE", "expand")

            retriever = HybridRetriever(
                vector_store=vector_store,
                graph_store=graph_store,
                embedding_model=embeddings,
                mode=hybrid_mode,
                vector_weight=0.7,  # Higher weight on vector for passage retrieval
                graph_weight=0.3,  # Lower weight on graph for context enhancement
                top_k=8,
                max_hops=2,
            )

            print(f"  Using hybrid mode: {hybrid_mode}")

        # Run queries
        print(f"\nRunning {len(queries)} queries...")
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
        print(format_metrics_table(metrics, f"RAGdoll ({mode})"))

        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        chunk_suffix = "_nochunk" if no_chunking else ""
        output_file = (
            results_dir / f"ragdoll_{dataset_name}_{subset}_{mode}{chunk_suffix}.json"
        )

        config_output = {
            "dataset": dataset_name,
            "subset": subset,
            "mode": mode,
            "top_k": 8,
            "no_chunking": no_chunking,
        }

        # Add hybrid_mode if applicable
        if mode == "hybrid":
            config_output["hybrid_mode"] = hybrid_mode

        with open(output_file, "w") as f:
            json.dump(
                {
                    "results": results,
                    "metrics": metrics,
                    "config": config_output,
                },
                f,
                indent=2,
            )

        print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAGdoll Benchmark")
    parser.add_argument(
        "-d", "--dataset", default="2wikimultihopqa", help="Dataset to use"
    )
    parser.add_argument("-n", type=int, default=51, help="Subset of corpus to use")
    parser.add_argument("-c", "--create", action="store_true", help="Create the index")
    parser.add_argument(
        "-b", "--benchmark", action="store_true", help="Run benchmark queries"
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["vector", "pagerank", "hybrid"],
        help="Retrieval mode",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Disable chunking (use whole-passage retrieval)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Please set it in .env or environment.")
        return

    base_dir = Path(__file__).parent / "db"

    # Add suffix for no-chunking mode to keep indices separate
    chunk_suffix = "_nochunk" if args.no_chunking else ""
    vector_dir = base_dir / f"ragdoll_{args.dataset}_{args.n}_vector{chunk_suffix}"

    if args.mode == "vector":
        working_dir = vector_dir
    else:
        suffix = MODE_WORKDIR_SUFFIX.get(args.mode, args.mode)
        working_dir = (
            base_dir / f"ragdoll_{args.dataset}_{args.n}_{suffix}{chunk_suffix}"
        )

    vector_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    asyncio.run(
        run_benchmark(
            args.dataset,
            args.n,
            args.mode,
            working_dir,
            vector_dir,
            create=args.create,
            benchmark=args.benchmark,
            no_chunking=args.no_chunking,
        )
    )


if __name__ == "__main__":
    main()
