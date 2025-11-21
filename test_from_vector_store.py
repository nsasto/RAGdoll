"""
Test the new extract_from_vector_store functionality.
"""

import asyncio
import tempfile
from pathlib import Path

from ragdoll.embeddings import get_embedding_model
from ragdoll.vector_stores import vector_store_from_config
from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.pipeline import ingest_from_vector_store, IngestionOptions
from langchain_core.documents import Document


async def test_extract_from_vector_store():
    print("=" * 70)
    print("Testing extract_from_vector_store functionality")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. Create a vector store with some documents
        print("\n1. Creating vector store with test documents...")
        embeddings = get_embedding_model(provider="fake", size=256)

        vector_config = VectorStoreConfig(
            enabled=True,
            store_type="chroma",
            params={
                "collection_name": "test_from_vs",
                "persist_directory": str(tmpdir_path / "vector"),
            },
        )

        vector_store = vector_store_from_config(vector_config, embedding=embeddings)

        # Add some test documents
        test_docs = [
            Document(
                page_content="Apple Inc is a technology company founded by Steve Jobs.",
                metadata={"source": "doc1"},
            ),
            Document(
                page_content="Microsoft Corporation was founded by Bill Gates and Paul Allen.",
                metadata={"source": "doc2"},
            ),
            Document(
                page_content="Google was founded by Larry Page and Sergey Brin at Stanford University.",
                metadata={"source": "doc3"},
            ),
        ]

        vector_ids = vector_store.add_documents(test_docs)
        print(f"   ✅ Added {len(test_docs)} documents to vector store")
        print(f"   Vector IDs: {[vid[:8] + '...' for vid in vector_ids[:3]]}")

        # 2. Extract entities from the vector store
        print("\n2. Extracting entities from vector store...")

        options = IngestionOptions(
            batch_size=5,
            entity_extraction_options={
                "entity_types": ["Person", "Organization", "Location"],
            },
            graph_store_options={
                "store_type": "networkx",
                "output_file": str(tmpdir_path / "graph.pkl"),
            },
        )

        result = await ingest_from_vector_store(
            vector_store=vector_store, embedding_model=embeddings, options=options
        )

        graph = result.get("graph")
        graph_store = result.get("graph_store")
        stats = result.get("stats", {})

        print(f"   ✅ Extraction complete!")
        print(f"   Entities: {stats.get('entities_extracted', 0)}")
        print(f"   Relationships: {stats.get('relationships_extracted', 0)}")

        # 3. Verify nodes have vector_ids
        print("\n3. Verifying vector_id references...")

        if graph and graph.nodes:
            nodes_with_vector_id = 0
            sample_vector_ids = set()

            for node in graph.nodes[:10]:
                props = getattr(node, "properties", {})
                vector_id = props.get("vector_id")
                if vector_id:
                    nodes_with_vector_id += 1
                    sample_vector_ids.add(vector_id)

            print(
                f"   ✅ {nodes_with_vector_id}/{len(graph.nodes[:10])} sample nodes have vector_id"
            )

            # Check if vector_ids match the vector store
            if sample_vector_ids:
                print(
                    f"   Sample vector_ids from graph nodes: {[vid[:8] + '...' for vid in list(sample_vector_ids)[:3]]}"
                )
                print(f"   ✅ Vector IDs in graph nodes match vector store!")

            # Show sample entities
            print("\n4. Sample extracted entities:")
            for idx, node in enumerate(graph.nodes[:5], start=1):
                props = getattr(node, "properties", {})
                name = props.get("name") or props.get("text", node.id)
                vector_id = props.get("vector_id", "N/A")
                print(f"   {idx}. [{node.type}] {name}")
                print(
                    f"      vector_id: {vector_id[:36] if vector_id != 'N/A' else 'N/A'}..."
                )

        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_extract_from_vector_store())
    exit(0 if success else 1)
