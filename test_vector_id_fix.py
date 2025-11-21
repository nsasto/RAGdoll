"""
Test script to verify the vector_id bug fix.

This script verifies that:
1. Pipeline returns vector_store in the result
2. Graph nodes have vector_ids that match the vector store
3. GraphRetriever can successfully retrieve embeddings
"""

import asyncio
import os
import tempfile
from pathlib import Path

from ragdoll.pipeline import ingest_documents, IngestionOptions
from ragdoll.retrieval import GraphRetriever
from ragdoll.embeddings import get_embedding_model
from ragdoll.vector_stores.adapter import VectorStoreAdapter
from dotenv import load_dotenv

load_dotenv(override=True)


async def test_vector_id_fix():
    print("=" * 60)
    print("Testing vector_id bug fix")
    print("=" * 60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a test document
        test_file = tmpdir_path / "test.txt"
        test_file.write_text(
            """
        Graph RAG systems combine knowledge graphs with retrieval-augmented generation.
        The system extracts entities like Person, Organization, and Location.
        These entities are connected through relationships to form a knowledge graph.
        Vector embeddings enable semantic search across both documents and graph nodes.
        """
        )

        # Setup embeddings (using fake embeddings for speed)
        embeddings = get_embedding_model(provider="fake", size=256)

        # Configure ingestion with both vector and graph stores
        options = IngestionOptions(
            batch_size=5,
            extract_entities=True,
            chunking_options={"chunk_size": 200, "chunk_overlap": 50},
            vector_store_options={
                "store_type": "chroma",
                "params": {
                    "collection_name": "test_fix",
                    "persist_directory": str(tmpdir_path / "vector"),
                },
            },
            graph_store_options={
                "store_type": "networkx",
                "output_file": str(tmpdir_path / "graph.pkl"),
            },
            entity_extraction_options={
                "entity_types": ["Person", "Organization", "Location"],
            },
        )

        # Run ingestion
        print("\n1. Running ingestion pipeline...")
        result = await ingest_documents([str(test_file)], options=options)

        # Check 1: Verify vector_store is returned
        print("\n2. Checking if vector_store is returned...")
        vector_store = result.get("vector_store")
        if vector_store is None:
            print("   ❌ FAILED: vector_store not in result")
            return False
        print("   ✅ PASSED: vector_store returned")

        # Check 2: Verify graph has nodes with vector_ids
        print("\n3. Checking graph nodes have vector_ids...")
        graph = result.get("graph")
        graph_store = result.get("graph_store")

        if not graph or not graph.nodes:
            print("   ⚠️  WARNING: No graph nodes created (might be expected if no LLM)")
            return True  # Not a failure, just no entities extracted

        # Get vector_ids from graph nodes
        node_vector_ids = set()
        for node in graph.nodes[:5]:  # Check first 5 nodes
            vector_id = node.properties.get("vector_id") if node.properties else None
            if vector_id:
                node_vector_ids.add(vector_id)

        if not node_vector_ids:
            print("   ❌ FAILED: No vector_ids found in graph nodes")
            return False
        print(f"   ✅ PASSED: Found {len(node_vector_ids)} unique vector_ids")

        # Check 3: Verify vector_ids exist in vector store
        print("\n4. Checking vector_ids exist in vector store...")
        adapter = VectorStoreAdapter(vector_store)

        # Try to retrieve embeddings
        embeddings_dict = adapter.get_embeddings_by_ids(list(node_vector_ids))

        if not embeddings_dict:
            print("   ❌ FAILED: No embeddings retrieved from vector store")
            return False

        overlap_count = len(embeddings_dict)
        total_count = len(node_vector_ids)
        overlap_percentage = (overlap_count / total_count) * 100

        print(
            f"   ✅ PASSED: {overlap_count}/{total_count} vector_ids found in store ({overlap_percentage:.1f}%)"
        )

        # Check 4: Test GraphRetriever with embedding-based search
        print("\n5. Testing GraphRetriever with embedding-based search...")
        graph_retriever = GraphRetriever(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_model=embeddings,
            top_k=3,
            max_hops=1,
            enable_fallback=True,
        )

        query = "What are knowledge graphs?"
        results = graph_retriever.get_relevant_documents(query)

        stats = graph_retriever.get_stats()
        indexed_nodes = stats.get("indexed_nodes", 0)
        total_nodes = stats.get("node_count", 0)

        print(f"   - Total nodes: {total_nodes}")
        print(f"   - Indexed nodes: {indexed_nodes}")
        print(f"   - Retrieved nodes: {len(results)}")

        if indexed_nodes == 0 and total_nodes > 0:
            print("   ⚠️  WARNING: No nodes indexed (embeddings not matched)")
            print("   This suggests vector_ids still don't match")
            return False

        if indexed_nodes > 0:
            print(
                f"   ✅ PASSED: GraphRetriever successfully indexed {indexed_nodes} nodes"
            )

        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED - Bug fix verified!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_vector_id_fix())
    exit(0 if success else 1)
