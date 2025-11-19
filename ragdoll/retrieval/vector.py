"""
Vector Retriever

Semantic similarity-based retrieval using vector stores.
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from ragdoll.retrieval.base import BaseRetriever


class VectorRetriever(BaseRetriever):
    """
    Vector-based semantic similarity retriever.

    Uses LangChain VectorStore implementations (FAISS, Chroma, etc.)
    for semantic search over document embeddings.

    Args:
        vector_store: LangChain VectorStore instance
        top_k: Number of documents to retrieve
        search_type: Type of search ("similarity", "mmr", or "similarity_score_threshold")
        search_kwargs: Additional search parameters
    """

    def __init__(
        self,
        vector_store,
        top_k: int = 3,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve documents via vector similarity search.

        Args:
            query: Query string to search for
            **kwargs: Override top_k or other search parameters

        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            return []

        # Allow runtime override of parameters
        k = kwargs.get("top_k", self.top_k)
        search_type = kwargs.get("search_type", self.search_type)

        # Merge runtime kwargs with configured search_kwargs
        merged_kwargs = {**self.search_kwargs, **kwargs}
        merged_kwargs["k"] = k

        try:
            if search_type == "similarity":
                documents = self.vector_store.similarity_search(query, k=k)
            elif search_type == "mmr":
                documents = self.vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=merged_kwargs.get("fetch_k", k * 2)
                )
            elif search_type == "similarity_score_threshold":
                score_threshold = merged_kwargs.get("score_threshold", 0.5)
                documents = self.vector_store.similarity_search_with_relevance_scores(
                    query, k=k, score_threshold=score_threshold
                )
                # Extract documents from (doc, score) tuples
                documents = [doc for doc, score in documents]
            else:
                # Default to basic similarity search
                documents = self.vector_store.similarity_search(query, k=k)

            return documents

        except Exception as e:
            print(f"Vector retrieval error: {e}")
            return []

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Args:
            query: Query string to search for
            **kwargs: Override top_k or other search parameters

        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            return []

        k = kwargs.get("top_k", self.top_k)
        search_type = kwargs.get("search_type", self.search_type)

        merged_kwargs = {**self.search_kwargs, **kwargs}
        merged_kwargs["k"] = k

        try:
            if search_type == "similarity":
                documents = await self.vector_store.asimilarity_search(query, k=k)
            elif search_type == "mmr":
                documents = await self.vector_store.amax_marginal_relevance_search(
                    query, k=k, fetch_k=merged_kwargs.get("fetch_k", k * 2)
                )
            else:
                documents = await self.vector_store.asimilarity_search(query, k=k)

            return documents

        except Exception as e:
            print(f"Async vector retrieval error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "top_k": self.top_k,
            "search_type": self.search_type,
        }

        # Try to get document count if vector store supports it
        try:
            if hasattr(self.vector_store, "__len__"):
                stats["document_count"] = len(self.vector_store)
            elif hasattr(self.vector_store, "index") and hasattr(
                self.vector_store.index, "ntotal"
            ):
                # FAISS-specific
                stats["document_count"] = self.vector_store.index.ntotal
        except:
            pass

        return stats
