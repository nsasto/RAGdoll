"""
Quick test to verify graph retrieval returns source passages instead of entity descriptions.

This script tests the fix by:
1. Loading an existing graph store (if available)
2. Performing a graph retrieval
3. Checking if returned documents have passage titles instead of "graph_retrieval"
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragdoll.embeddings import get_embedding_model
from ragdoll.graph_stores import get_graph_store
from ragdoll.retrieval import GraphRetriever
from ragdoll.vector_stores import create_vector_store
from dotenv import load_dotenv

load_dotenv(override=True)


def test_source_passage_retrieval():
    """Test if graph retrieval returns source passages."""

    print("=" * 80)
    print("Testing Graph Retrieval Source Passage Fix")
    print("=" * 80)

    base_dir = Path(__file__).parent / "db"

    # Check if shared graph index exists (try 25-query first, then 51-query)
    graph_candidates = [
        base_dir / "ragdoll_2wikimultihopqa_25_graph",
        base_dir / "ragdoll_2wikimultihopqa_51_graph",
        base_dir / "ragdoll_2wikimultihopqa_25_hybrid",
        base_dir / "ragdoll_2wikimultihopqa_51_hybrid",
    ]

    graph_dir = next((path for path in graph_candidates if path.exists()), None)

    if not graph_dir:
        expected = ", ".join(str(path) for path in graph_candidates[:2])
        print(f"\n❌ Graph index not found at: {expected}")
        print("   Run the benchmark first to create the index.")
        return False

    print(f"\n✅ Found graph index at: {graph_dir}")

    # Load graph store
    graph_store_path = graph_dir / "graph.pkl"
    if not graph_store_path.exists():
        print(f"\n❌ Graph store not found at: {graph_store_path}")
        return False

    print(f"✅ Loading graph store from: {graph_store_path}")
    graph_store = get_graph_store(
        store_type="networkx", input_file=str(graph_store_path)
    )

    # Get node count
    if hasattr(graph_store, "nodes"):
        node_count = len(list(graph_store.nodes()))
        print(f"   Graph has {node_count} nodes")

    # Load vector store (shared across modes)
    vector_candidates = [
        base_dir / "ragdoll_2wikimultihopqa_25_vector",
        base_dir / "ragdoll_2wikimultihopqa_51_vector",
        graph_dir,
    ]
    vector_dir = next(
        (path for path in vector_candidates if (path / "vector").exists()), None
    )

    if not vector_dir:
        expected = ", ".join(str(path) for path in vector_candidates[:2])
        print(f"\n❌ Vector index not found at: {expected}")
        print("   Run the benchmark first to create the vector index.")
        return False

    vector_store_path = vector_dir / "vector"

    print(f"✅ Loading vector store from: {vector_store_path}")
    embeddings = get_embedding_model()

    # Determine collection name from db_path
    collection_name = vector_dir.name  # e.g., "ragdoll_2wikimultihopqa_25_vector"

    vector_store = create_vector_store(
        "chroma",
        embedding=embeddings,
        persist_directory=str(vector_store_path),
        collection_name=collection_name,
    )

    # Create graph retriever
    print("\n✅ Creating GraphRetriever with vector store")
    graph_retriever = GraphRetriever(
        graph_store=graph_store,
        vector_store=vector_store,
        embedding_model=embeddings,
        top_k=3,
        max_hops=1,
    )

    # Test query
    test_query = "Who is the father of Ava Kolker?"
    print(f"\n🔍 Testing query: '{test_query}'")

    results = graph_retriever.get_relevant_documents(test_query)

    print(f"\n📊 Retrieved {len(results)} documents:")
    print("-" * 80)

    source_passages = 0
    entity_descriptions = 0

    for i, doc in enumerate(results, 1):
        metadata = doc.metadata
        content_preview = doc.page_content[:100].replace("\n", " ")

        # Check if this is a source passage or entity description
        has_title = "title" in metadata
        has_source = "source" in metadata
        retrieval_method = metadata.get(
            "retrieval_method", metadata.get("source", "unknown")
        )

        if has_title or retrieval_method == "graph_expanded":
            source_passages += 1
            status = "✅ SOURCE PASSAGE"
        elif has_source and metadata["source"] == "graph_retrieval":
            entity_descriptions += 1
            status = "❌ ENTITY DESCRIPTION"
        else:
            status = "⚠️  UNKNOWN"

        print(f"\n{i}. {status}")
        print(f"   Metadata: {dict(list(metadata.items())[:5])}")
        print(f"   Content: {content_preview}...")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Source Passages: {source_passages}/{len(results)} ({source_passages/len(results)*100:.1f}%)"
    )
    print(
        f"Entity Descriptions: {entity_descriptions}/{len(results)} ({entity_descriptions/len(results)*100:.1f}%)"
    )

    if source_passages > 0:
        print("\n✅ SUCCESS: Graph retrieval is returning source passages!")
        return True
    else:
        print("\n❌ FAILED: Graph retrieval is still returning entity descriptions")
        return False


if __name__ == "__main__":
    success = test_source_passage_retrieval()
    sys.exit(0 if success else 1)
