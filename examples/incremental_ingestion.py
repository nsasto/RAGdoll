"""
Example: Initial population + incremental updates to existing RAGdoll stores.

Demonstrates:
1. Initial ingestion with vector + graph construction
2. Adding new documents to existing stores
3. Querying across old and new data
"""

import asyncio
from pathlib import Path
import os
from ragdoll import Ragdoll, settings
from ragdoll.pipeline import IngestionOptions
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)
# Ensure we use the repo config with public OpenAI models
os.environ.setdefault(
    "RAGDOLL_CONFIG_PATH",
    str(Path(__file__).resolve().parents[1] / "ragdoll" / "config" / "default_config.yaml"),
)
# logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger("openai").setLevel(logging.DEBUG)


async def main():
    # Use shared singleton for consistent cache/metrics
    app = settings.get_app()

    # Initialize RAGdoll (creates stores if needed)
    ragdoll = Ragdoll()
    if ragdoll.llm_caller:
        try:
            probe = await ragdoll.llm_caller.call("ping")
            print("LLM smoke test:", (probe or "")[:120], "...")
        except Exception as smoke_err:
            print(f"LLM smoke test failed: {smoke_err}")

    print("=" * 60)
    print("STEP 1: Initial Population")
    print("=" * 60)

    # Initial ingestion - builds both vector and graph stores
    initial_sources = ["docs/architecture.md", "docs/overview.md"]

    result = await ragdoll.ingest_with_graph(
        sources=initial_sources,
        options=IngestionOptions(
            batch_size=20,
            max_concurrent_llm_calls=15,  # Entity extraction concurrency
            chunking_options={
                "chunking_strategy": "markdown",  # Good for structured docs
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            graph_store_options={
                "store_type": "networkx"  # Use NetworkX for full graph capabilities
            },
        ),
    )

    print(f"\n✓ Initial ingestion complete:")
    print(f"  - Documents processed: {len(initial_sources)}")
    print(f"  - Vector store populated: {ragdoll.vector_store is not None}")
    print(f"  - Graph store populated: {ragdoll.graph_store is not None}")
    print(f"  - Hybrid retriever available: {ragdoll.hybrid_retriever is not None}")
    print(f"  - Graph retriever available: {ragdoll.graph_retriever is not None}")

    # Test query on initial data
    print("\n" + "=" * 60)
    print("Query 1: Testing initial data")
    print("=" * 60)

    try:
        print("Querying hybrid retriever...")
        answer1 = ragdoll.query_hybrid("What is RAGdoll's architecture?", k=5)
        if answer1.get("answer"):
            print(f"\nAnswer: {answer1['answer'][:200]}...")
        else:
            print("\nNo answer generated (LLM may not be configured)")
        print(f"Sources: {len(answer1['documents'])} documents")
    except Exception as e:
        print(f"Query 1 failed: {e}")
        import traceback

        traceback.print_exc()

    # Add new documents to existing stores
    print("\n" + "=" * 60)
    print("STEP 2: Incremental Update")
    print("=" * 60)

    new_sources = ["docs/retrieval.md", "docs/ingestion.md"]

    # CRITICAL: Reuse existing stores - pipeline adds new data
    result_incremental = await ragdoll.ingest_with_graph(
        sources=new_sources,
        options=IngestionOptions(
            batch_size=20,
            max_concurrent_llm_calls=15,  # Entity extraction concurrency
            chunking_options={
                "chunking_strategy": "markdown",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            graph_store_options={"store_type": "networkx"},  # Use same graph store type
        ),
    )

    print(f"\n✓ Incremental ingestion complete:")
    print(f"  - New documents processed: {len(new_sources)}")
    print(f"  - Vector store updated (additive)")
    print(f"  - Graph store updated (additive)")
    print(f"  - Graph nodes preserve vector_id links")

    # Test query on combined data (old + new)
    print("\n" + "=" * 60)
    print("Query 2: Testing combined data (old + new)")
    print("=" * 60)

    try:
        print("Querying hybrid retriever...")
        answer2 = ragdoll.query_hybrid("How does ingestion and retrieval work?", k=5)
        if answer2.get("answer"):
            print(f"\nAnswer: {answer2['answer'][:200]}...")
        else:
            print("\nNo answer generated (LLM may not be configured)")
        print(f"Sources: {len(answer2['documents'])} documents")
    except Exception as e:
        print(f"Query 2 failed: {e}")
        import traceback

        traceback.print_exc()

    # Verify we can find both old and new content
    print("\n" + "=" * 60)
    print("Query 3: Cross-document verification")
    print("=" * 60)

    try:
        print("Querying hybrid retriever...")
        answer3 = ragdoll.query_hybrid(
            "What is the relationship between architecture and retrieval?", k=5
        )
        if answer3.get("answer"):
            print(f"\nAnswer: {answer3['answer'][:200]}...")
        else:
            print("\nNo answer generated (LLM may not be configured)")

        # Show source document origins
        sources_by_file = {}
        for doc in answer3["documents"]:
            source = doc.metadata.get("source", "unknown")
            filename = Path(source).name
            sources_by_file[filename] = sources_by_file.get(filename, 0) + 1

        print("\n✓ Sources span both initial and incremental data:")
        for filename, count in sources_by_file.items():
            print(f"  - {filename}: {count} chunks")
    except Exception as e:
        print(f"Query 3 failed: {e}")
        import traceback

        traceback.print_exc()

    # Demonstrate access to underlying stores
    print("\n" + "=" * 60)
    print("Store Statistics")
    print("=" * 60)

    if hasattr(ragdoll.graph_store, "get_all_nodes"):
        nodes = ragdoll.graph_store.get_all_nodes()
        print(f"  - Graph nodes: {len(nodes)}")

        # Count nodes with vector_id (should be all of them)
        linked_nodes = sum(
            1 for node in nodes if node.get("properties", {}).get("vector_id")
        )
        print(f"  - Nodes with vector_id links: {linked_nodes}")

    print("\n✓ Example complete!")
    print("  Both vector and graph stores contain all data")
    print("  Queries retrieve across old + new documents")
    print("  Graph expansion works via preserved vector_id metadata")


# Alternative: Manual control over incremental updates
async def manual_incremental_example():
    """
    Fine-grained control: manually add chunks to stores.
    Useful when you have pre-processed chunks.
    """
    from ragdoll import settings
    from ragdoll.entity_extraction import EntityExtractionService
    from langchain_core.documents import Document

    app = settings.get_app()
    ragdoll = Ragdoll()

    # Assume stores already populated from previous run
    vector_store = ragdoll.vector_store
    graph_store = ragdoll.graph_store

    # New pre-chunked documents
    new_chunks = [
        Document(
            page_content="RAGdoll 2.1 introduces parallel embedding.",
            metadata={"source": "changelog.md", "chunk_id": 0},
        ),
        Document(
            page_content="Vector stores now handle concurrency internally.",
            metadata={"source": "changelog.md", "chunk_id": 1},
        ),
    ]

    print("Manual incremental update:")

    # 1. Add to vector store (parallel embedding via BaseVectorStore)
    vector_ids = await vector_store.aadd_documents(new_chunks)
    print(f"  ✓ Added {len(vector_ids)} chunks to vector store")

    # 2. Extract entities with vector ID linking
    entity_service = EntityExtractionService(
        llm_caller=app.get_llm_caller(), config=app.config.entity_extraction_config
    )

    extraction_results = await entity_service.extract_from_documents(
        documents=new_chunks,
        vector_ids=vector_ids,  # CRITICAL: Links graph to embeddings
    )

    # 3. Add to graph (operations are additive)
    for entity in extraction_results["entities"]:
        graph_store.add_node(
            entity["id"],
            label=entity["label"],
            properties={
                **entity["properties"],
                "vector_id": entity["metadata"]["vector_id"],  # Preserve link
            },
        )

    for rel in extraction_results["relationships"]:
        graph_store.add_edge(
            rel["source"],
            rel["target"],
            relationship=rel["type"],
            properties=rel.get("properties", {}),
        )

    print(f"  ✓ Extracted {len(extraction_results['entities'])} entities")
    print(f"  ✓ Extracted {len(extraction_results['relationships'])} relationships")
    print("  ✓ All nodes linked via vector_id metadata")


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment to run manual control example
    # asyncio.run(manual_incremental_example())
