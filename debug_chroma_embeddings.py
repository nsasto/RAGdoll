"""
Minimal debug script to test Chroma embedding retrieval.
"""

import logging
from pathlib import Path
import tempfile

from langchain_core.documents import Document

from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.embeddings import get_embedding_model
from ragdoll.vector_stores import vector_store_from_config
from ragdoll.vector_stores.adapter import VectorStoreAdapter

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)

    print("=" * 70)
    print("Debug: Chroma Embedding Retrieval")
    print("=" * 70)

    # Create vector store
    print("\n1. Creating vector store...")
    embeddings = get_embedding_model(provider="fake", size=256)

    vector_config = VectorStoreConfig(
        enabled=True,
        store_type="chroma",
        params={
            "collection_name": "debug_test",
            "persist_directory": str(tmpdir_path / "vector"),
        },
    )

    vector_store = vector_store_from_config(vector_config, embedding=embeddings)

    test_docs = [
        Document(page_content="Test document one.", metadata={"source": "doc1"}),
        Document(page_content="Test document two.", metadata={"source": "doc2"}),
    ]

    vector_ids = vector_store.add_documents(test_docs)
    print(f"Added {len(test_docs)} documents")
    print(f"Vector IDs: {vector_ids}")

    # Test direct Chroma access
    print("\n2. Testing direct Chroma collection access...")
    store = vector_store._store if hasattr(vector_store, "_store") else vector_store
    print(f"Store type: {type(store)}")
    print(f"Has _collection: {hasattr(store, '_collection')}")

    if hasattr(store, "_collection"):
        collection = store._collection
        print(f"Collection type: {type(collection)}")

        # Try get without embeddings
        results_no_embed = collection.get(ids=vector_ids)
        print(f"\nResults without embeddings:")
        print(f"  IDs: {results_no_embed.get('ids', [])}")
        print(f"  Has 'embeddings' key: {'embeddings' in results_no_embed}")

        # Try get with embeddings
        results_with_embed = collection.get(ids=vector_ids, include=["embeddings"])
        print(f"\nResults with embeddings:")
        print(f"  IDs: {results_with_embed.get('ids', [])}")
        print(f"  Has 'embeddings' key: {'embeddings' in results_with_embed}")
        print(f"  Embeddings value: {results_with_embed.get('embeddings')}")

        if results_with_embed.get("embeddings"):
            first_embedding = results_with_embed["embeddings"][0]
            print(f"  First embedding type: {type(first_embedding)}")
            print(
                f"  First embedding length: {len(first_embedding) if first_embedding else 0}"
            )

    # Test adapter
    print("\n3. Testing VectorStoreAdapter...")
    adapter = VectorStoreAdapter(vector_store)
    print(f"Adapter backend: {adapter._backend}")

    embeddings_dict = adapter.get_embeddings_by_ids(vector_ids)
    print(f"\nAdapter retrieved {len(embeddings_dict)} embeddings")
    for vid, emb in embeddings_dict.items():
        print(f"  {vid[:8]}... -> embedding shape: {emb.shape}")

    print("\n" + "=" * 70)
    print("Debug complete!")
    print("=" * 70)
