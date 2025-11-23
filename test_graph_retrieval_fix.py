"""
Test script to verify graph retrieval with embeddings works after VectorStoreAdapter fix.
"""

import asyncio
import sys
from pathlib import Path
import tempfile

from langchain_core.documents import Document

from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.embeddings import get_embedding_model
from ragdoll.vector_stores import vector_store_from_config
from ragdoll.pipeline import ingest_from_vector_store, IngestionOptions


async def test_graph_retrieval_with_embeddings():
    """Test that graph retrieval successfully indexes nodes and retrieves results."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        print("=" * 70)
        print("Testing Graph Retrieval with Embeddings")
        print("=" * 70)

        # 1. Create vector store with documents
        print("\n1. Creating vector store...")
        embeddings = get_embedding_model(provider="fake", size=256)

        vector_config = VectorStoreConfig(
            enabled=True,
            store_type="chroma",
            params={
                "collection_name": "graph_retrieval_test",
                "persist_directory": str(tmpdir_path / "vector"),
            },
        )

        vector_store = vector_store_from_config(vector_config, embedding=embeddings)

        test_docs = [
            Document(
                page_content="Python is a programming language created by Guido van Rossum.",
                metadata={"source": "doc1"},
            ),
            Document(
                page_content="Machine learning is a field of artificial intelligence.",
                metadata={"source": "doc2"},
            ),
            Document(
                page_content="Neural networks are used in deep learning systems.",
                metadata={"source": "doc3"},
            ),
        ]

        vector_ids = vector_store.add_documents(test_docs)
        print(f"   ✅ Added {len(test_docs)} documents to vector store")

        # 2. Extract entities and create graph
        print("\n2. Extracting entities and building graph...")

        options = IngestionOptions(
            batch_size=5,
            entity_extraction_options={
                "entity_types": ["Person", "Organization", "Technology"],
            },
            graph_store_options={
                "store_type": "networkx",
                "output_file": str(tmpdir_path / "graph.pkl"),
            },
        )

        result = await ingest_from_vector_store(
            vector_store=vector_store,
            embedding_model=embeddings,
            options=options,
        )

        graph = result.get("graph")
        graph_retriever = result.get("graph_retriever")
        stats = result.get("stats", {})

        print(f"   Entities extracted: {stats.get('entities_extracted', 0)}")
        print(f"   Relationships found: {stats.get('relationships_extracted', 0)}")

        # 3. Check that nodes are indexed
        print("\n3. Checking embedding index...")

        if graph_retriever:
            retriever_stats = graph_retriever.get_stats()
            node_count = retriever_stats.get("node_count", 0)
            indexed_nodes = retriever_stats.get("indexed_nodes", 0)

            print(f"   Total nodes: {node_count}")
            print(f"   Indexed nodes: {indexed_nodes}")

            if indexed_nodes == 0:
                print("   ❌ FAILED: No nodes were indexed with embeddings!")
                return False
            elif indexed_nodes < node_count:
                print(f"   ⚠️  WARNING: Only {indexed_nodes}/{node_count} nodes indexed")
            else:
                print(f"   ✅ All {indexed_nodes} nodes successfully indexed!")

        # 4. Test retrieval
        print("\n4. Testing graph retrieval...")

        if graph_retriever:
            test_query = "What technologies are mentioned?"
            results = graph_retriever.get_relevant_documents(test_query)

            print(f"   Query: '{test_query}'")
            print(f"   Retrieved: {len(results)} nodes")

            if len(results) == 0:
                print("   ❌ FAILED: No results returned!")
                return False

            print("\n   Sample results:")
            for idx, doc in enumerate(results[:5], start=1):
                node_type = doc.metadata.get("node_type", "unknown")
                node_id = doc.metadata.get("node_id", "unknown")
                score = doc.metadata.get("relevance_score", 0)
                print(f"   {idx}. [{node_type}] {node_id} (score: {score:.3f})")

            print("\n   ✅ Graph retrieval working!")

        # Cleanup
        try:
            if hasattr(vector_store, "_store") and hasattr(
                vector_store._store, "_client"
            ):
                del vector_store._store._client
            del vector_store
        except:
            pass

        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_graph_retrieval_with_embeddings())
    sys.exit(0 if success else 1)
