"""
RAGdoll Retrieval Examples

Demonstrates the new modular retrieval architecture with vector, graph,
and hybrid retrieval strategies.
"""

from ragdoll import Ragdoll, VectorRetriever, GraphRetriever, HybridRetriever


def example_basic_retrieval():
    """Basic example using the Ragdoll orchestrator."""

    # Initialize RAGdoll with config
    ragdoll = Ragdoll(config_path="app_config.yaml")

    # Ingest documents and build knowledge graph
    result = ragdoll.ingest_with_graph_sync(
        ["docs/architecture.md", "docs/configuration.md", "examples/README.md"]
    )

    print(f"Ingested {result['stats'].get('total_documents', 0)} documents")
    print(f"Graph has {result['stats'].get('entities_extracted', 0)} entities")

    # Use the automatically configured hybrid retriever
    query = "What are the main components of RAGdoll?"
    result = ragdoll.query_hybrid(query, k=5)

    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")
    print(f"\nRetrieved {len(result['documents'])} documents:")
    for i, doc in enumerate(result["documents"], 1):
        source = doc.metadata.get("retrieval_source", "unknown")
        print(f"  {i}. [{source}] {doc.page_content[:100]}...")


def example_vector_only():
    """Example using vector retrieval only."""

    ragdoll = Ragdoll()

    # Ingest without graph extraction
    documents = ragdoll.ingest_data(["docs/README.md"])

    # Create vector retriever directly
    vector_retriever = VectorRetriever(
        vector_store=ragdoll.vector_store,
        top_k=5,
        search_type="mmr",  # Maximum Marginal Relevance for diversity
    )

    # Query
    docs = vector_retriever.get_relevant_documents("How do I configure embeddings?")

    print(f"Found {len(docs)} relevant documents")
    for doc in docs:
        print(f"- {doc.metadata.get('source', 'unknown')}")


def example_graph_only():
    """Example using graph retrieval only."""

    ragdoll = Ragdoll()

    # Ingest with graph extraction
    result = ragdoll.ingest_with_graph_sync(["docs/entity_extraction.md"])

    # Access the graph retriever
    graph_retriever = ragdoll.graph_retriever

    if graph_retriever:
        # Query for entity relationships
        docs = graph_retriever.get_relevant_documents(
            "What entities are related to LLM?"
        )

        print(f"Found {len(docs)} graph nodes")
        for doc in docs:
            node_type = doc.metadata.get("node_type", "Entity")
            hop_dist = doc.metadata.get("hop_distance", 0)
            print(f"- {node_type} (distance: {hop_dist})")
            print(f"  {doc.page_content[:150]}")
    else:
        print("Graph retriever not available")


def example_custom_hybrid():
    """Example with custom hybrid retriever configuration."""

    ragdoll = Ragdoll()
    result = ragdoll.ingest_with_graph_sync(["docs/*.md"])

    # Create custom vector retriever with specific settings
    vector_retriever = VectorRetriever(
        vector_store=ragdoll.vector_store,
        top_k=10,
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7},
    )

    # Create custom graph retriever with deep traversal
    graph_retriever = GraphRetriever(
        graph_store=ragdoll.graph_store,
        top_k=3,
        max_hops=3,  # Deep traversal
        traversal_strategy="dfs",  # Depth-first for chains
        include_edges=True,
        min_score=0.5,
    )

    # Combine with weighted strategy
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        mode="rerank",  # Re-score combined results
        vector_weight=0.7,
        graph_weight=0.3,
        deduplicate=True,
    )

    # Query
    docs = hybrid_retriever.get_relevant_documents(
        "Explain the relationship between entities and graphs"
    )

    print(f"Retrieved {len(docs)} documents using custom hybrid strategy")

    # Get statistics
    stats = hybrid_retriever.get_stats()
    print("\nRetriever Configuration:")
    print(f"  Mode: {stats['mode']}")
    print(f"  Vector top_k: {stats['vector_stats']['top_k']}")
    print(f"  Graph max_hops: {stats['graph_stats']['max_hops']}")


def example_comparison():
    """Compare different retrieval strategies."""

    ragdoll = Ragdoll()
    result = ragdoll.ingest_with_graph_sync(["docs/retrieval.md"])

    query = "What is graph traversal?"

    print(f"Query: {query}\n")

    # 1. Vector-only
    vector_docs = ragdoll.vector_store.similarity_search(query, k=3)
    print(f"Vector-only: {len(vector_docs)} documents")

    # 2. Graph-only
    if ragdoll.graph_retriever:
        graph_docs = ragdoll.graph_retriever.get_relevant_documents(query)
        print(f"Graph-only: {len(graph_docs)} documents")

    # 3. Hybrid concat
    ragdoll.hybrid_retriever.mode = "concat"
    concat_docs = ragdoll.hybrid_retriever.get_relevant_documents(query)
    print(f"Hybrid (concat): {len(concat_docs)} documents")

    # 4. Hybrid rerank
    ragdoll.hybrid_retriever.mode = "rerank"
    rerank_docs = ragdoll.hybrid_retriever.get_relevant_documents(query)
    print(f"Hybrid (rerank): {len(rerank_docs)} documents")

    # Show top result from each strategy
    print("\nTop result from each strategy:")
    print(f"\nVector: {vector_docs[0].page_content[:100]}...")
    if ragdoll.graph_retriever and graph_docs:
        print(f"\nGraph: {graph_docs[0].page_content[:100]}...")
    print(f"\nReranked: {rerank_docs[0].page_content[:100]}...")


def example_async_retrieval():
    """Example using async retrieval."""

    import asyncio

    async def retrieve_async():
        ragdoll = Ragdoll()
        await ragdoll.ingest_with_graph(["docs/README.md"])

        # Async retrieval
        docs = await ragdoll.hybrid_retriever.aget_relevant_documents(
            "What is RAGdoll?"
        )

        print(f"Retrieved {len(docs)} documents asynchronously")
        return docs

    # Run async function
    docs = asyncio.run(retrieve_async())
    return docs


if __name__ == "__main__":
    print("=" * 70)
    print("RAGdoll Retrieval Examples")
    print("=" * 70)

    print("\n1. Basic Hybrid Retrieval")
    print("-" * 70)
    example_basic_retrieval()

    print("\n\n2. Vector-Only Retrieval")
    print("-" * 70)
    example_vector_only()

    print("\n\n3. Graph-Only Retrieval")
    print("-" * 70)
    example_graph_only()

    print("\n\n4. Custom Hybrid Configuration")
    print("-" * 70)
    example_custom_hybrid()

    print("\n\n5. Strategy Comparison")
    print("-" * 70)
    example_comparison()

    print("\n\n6. Async Retrieval")
    print("-" * 70)
    example_async_retrieval()
