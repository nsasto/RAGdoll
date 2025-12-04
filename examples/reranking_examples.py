"""
Example: Using RerankerRetriever to improve retrieval quality

This example demonstrates how to enable and use reranking to filter
and improve the quality of retrieved documents.
"""

from ragdoll import Ragdoll
from ragdoll.retrieval import VectorRetriever, RerankerRetriever
from langchain_core.documents import Document

# Example 1: Automatic reranking via config
print("=" * 60)
print("Example 1: Automatic Reranking (via config)")
print("=" * 60)

# In your config.yaml, set:
# retriever:
#   reranker:
#     enabled: true
#     provider: "llm"
#     top_k: 5
#     over_retrieve_multiplier: 2

# When reranker is enabled in config, all Ragdoll queries use it automatically
ragdoll = Ragdoll()

# This will:
# 1. Retrieve 10 documents (5 * 2)
# 2. Score each with LLM
# 3. Return top 5 most relevant
result = ragdoll.query("What are the main components of RAGdoll?", k=5)

print(f"Answer: {result['answer']}")
print(f"Documents returned: {len(result['documents'])}")
for i, doc in enumerate(result["documents"], 1):
    score = doc.metadata.get("rerank_score", "N/A")
    print(f"  {i}. Score: {score:.2f} - {doc.page_content[:80]}...")


# Example 2: Manual reranking with custom retriever
print("\n" + "=" * 60)
print("Example 2: Manual Reranking")
print("=" * 60)

# Assuming you have a vector store and documents loaded
from ragdoll.vector_stores import vector_store_from_config
from ragdoll.embeddings import get_embedding_model
from ragdoll.app_config import bootstrap_app

app_config = bootstrap_app()
embedding_model = get_embedding_model(config_manager=app_config.config_manager)

# Create sample documents
sample_docs = [
    Document(
        page_content="RAGdoll is a RAG orchestration framework",
        metadata={"source": "doc1"},
    ),
    Document(
        page_content="It supports vector and graph retrieval",
        metadata={"source": "doc2"},
    ),
    Document(page_content="Weather forecast for tomorrow", metadata={"source": "doc3"}),
    Document(
        page_content="Entity extraction creates knowledge graphs",
        metadata={"source": "doc4"},
    ),
    Document(page_content="Random unrelated content", metadata={"source": "doc5"}),
]

# Create a simple vector store
vector_store = vector_store_from_config(
    config_manager=app_config.config_manager,
    embeddings=embedding_model,
)
vector_store.add_documents(sample_docs)

# Create base retriever
base_retriever = VectorRetriever(
    vector_store=vector_store, top_k=10, search_type="similarity"
)

# Wrap with reranker
reranker = RerankerRetriever(
    base_retriever=base_retriever,
    provider="llm",
    top_k=3,
    over_retrieve_multiplier=2,
    score_threshold=0.5,  # Filter documents below 0.5 relevance
    log_scores=True,
)

# Query
docs = reranker.get_relevant_documents("What features does RAGdoll support?")

print(f"\nReranked Results (top {len(docs)}):")
for i, doc in enumerate(docs, 1):
    score = doc.metadata.get("rerank_score", "N/A")
    print(f"{i}. Score: {score:.2f}")
    print(f"   Source: {doc.metadata.get('source')}")
    print(f"   Content: {doc.page_content}")
    print()


# Example 3: Comparing with and without reranking
print("=" * 60)
print("Example 3: With vs Without Reranking")
print("=" * 60)

query = "Tell me about RAGdoll's retrieval capabilities"

# Without reranking
print("\nWithout Reranking:")
vector_docs = base_retriever.get_relevant_documents(query, top_k=3)
for i, doc in enumerate(vector_docs, 1):
    print(f"{i}. {doc.page_content[:60]}...")

# With reranking
print("\nWith Reranking:")
reranked_docs = reranker.get_relevant_documents(query, top_k=3)
for i, doc in enumerate(reranked_docs, 1):
    score = doc.metadata.get("rerank_score", "N/A")
    print(f"{i}. (Score: {score:.2f}) {doc.page_content[:60]}...")


# Example 4: Different reranking providers
print("\n" + "=" * 60)
print("Example 4: Different Reranking Providers")
print("=" * 60)

print("\nLLM Provider (gpt-3.5-turbo):")
print("  - Cost: Medium")
print("  - Speed: Slow")
print("  - Quality: High")
print("  - Use case: Best semantic understanding")

print("\nCohere Provider (rerank-english-v3):")
print("  - Cost: Low")
print("  - Speed: Fast")
print("  - Quality: High")
print("  - Use case: Production, high volume")
print("  - Note: Requires COHERE_API_KEY environment variable")

print("\nCross-Encoder Provider (local model):")
print("  - Cost: Free")
print("  - Speed: Fast")
print("  - Quality: Good")
print("  - Use case: Local deployment, no API calls")
print("  - Note: Requires sentence-transformers package")


# Example 5: Monitoring reranking performance
print("\n" + "=" * 60)
print("Example 5: Reranker Statistics")
print("=" * 60)

stats = reranker.get_stats()
print("\nReranker Configuration:")
for key, value in stats.items():
    print(f"  {key}: {value}")


print("\n" + "=" * 60)
print("Configuration Tips")
print("=" * 60)
print(
    """
1. Set over_retrieve_multiplier based on your needs:
   - 2x: Good balance (retrieve 10, return 5)
   - 3x: Better coverage but slower (retrieve 15, return 5)
   - 1.5x: Faster with less reranking overhead

2. Adjust score_threshold to filter noise:
   - 0.0: No filtering (default)
   - 0.3-0.5: Moderate filtering
   - 0.6+: Aggressive filtering (only highly relevant docs)

3. Choose provider based on use case:
   - Development: cross-encoder (free, local)
   - Production (quality): llm with gpt-3.5-turbo
   - Production (speed): cohere
   - Production (cost): cross-encoder

4. Enable log_scores during debugging to see relevance scores
"""
)
