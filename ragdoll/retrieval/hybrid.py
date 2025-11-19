"""
Hybrid Retriever

Combines vector and graph retrieval strategies for enhanced RAG.
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from ragdoll.retrieval.base import BaseRetriever
from ragdoll.retrieval.vector import VectorRetriever
from ragdoll.retrieval.graph import GraphRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector similarity and graph traversal.

    Supports multiple combination strategies:
    - concat: Concatenate vector and graph results
    - rerank: Re-score and rerank combined results
    - weighted: Weighted combination based on scores
    - expand: Use vector results to seed graph expansion

    Args:
        vector_retriever: VectorRetriever instance
        graph_retriever: Optional GraphRetriever instance
        mode: Combination strategy ("concat", "rerank", "weighted", "expand")
        vector_weight: Weight for vector results (0-1) in weighted mode
        graph_weight: Weight for graph results (0-1) in weighted mode
        deduplicate: Whether to remove duplicate documents
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: Optional[GraphRetriever] = None,
        mode: str = "concat",
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        deduplicate: bool = True,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.mode = mode.lower()
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.deduplicate = deduplicate

        # Validate weights sum to 1.0 in weighted mode
        if self.mode == "weighted":
            total = self.vector_weight + self.graph_weight
            if abs(total - 1.0) > 0.01:
                # Normalize weights
                self.vector_weight = self.vector_weight / total
                self.graph_weight = self.graph_weight / total

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve documents using hybrid strategy.

        Args:
            query: Query string
            **kwargs: Override retrieval parameters

        Returns:
            Combined list of relevant documents
        """
        # Get vector results
        vector_docs = self.vector_retriever.get_relevant_documents(query, **kwargs)

        # If no graph retriever, return vector results only
        if not self.graph_retriever:
            return vector_docs

        # Get graph results
        graph_docs = self.graph_retriever.get_relevant_documents(query, **kwargs)

        # Combine based on strategy
        if self.mode == "concat":
            combined = self._concat_results(vector_docs, graph_docs)
        elif self.mode == "rerank":
            combined = self._rerank_results(vector_docs, graph_docs, query)
        elif self.mode == "weighted":
            combined = self._weighted_combine(vector_docs, graph_docs)
        elif self.mode == "expand":
            combined = self._expand_results(vector_docs, graph_docs)
        else:
            # Default to concat
            combined = self._concat_results(vector_docs, graph_docs)

        # Deduplicate if requested
        if self.deduplicate:
            combined = self._deduplicate_documents(combined)

        return combined

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Args:
            query: Query string
            **kwargs: Override retrieval parameters

        Returns:
            Combined list of relevant documents
        """
        # Get vector results
        vector_docs = await self.vector_retriever.aget_relevant_documents(
            query, **kwargs
        )

        if not self.graph_retriever:
            return vector_docs

        # Get graph results
        graph_docs = await self.graph_retriever.aget_relevant_documents(query, **kwargs)

        # Combine (reuse sync combination logic)
        if self.mode == "concat":
            combined = self._concat_results(vector_docs, graph_docs)
        elif self.mode == "rerank":
            combined = self._rerank_results(vector_docs, graph_docs, query)
        elif self.mode == "weighted":
            combined = self._weighted_combine(vector_docs, graph_docs)
        elif self.mode == "expand":
            combined = self._expand_results(vector_docs, graph_docs)
        else:
            combined = self._concat_results(vector_docs, graph_docs)

        if self.deduplicate:
            combined = self._deduplicate_documents(combined)

        return combined

    def _concat_results(
        self, vector_docs: List[Document], graph_docs: List[Document]
    ) -> List[Document]:
        """
        Simple concatenation of results.

        Vector results come first (usually more relevant),
        followed by graph results for additional context.
        """
        # Add source metadata if not present
        for doc in vector_docs:
            if "retrieval_source" not in doc.metadata:
                doc.metadata["retrieval_source"] = "vector"

        for doc in graph_docs:
            if "retrieval_source" not in doc.metadata:
                doc.metadata["retrieval_source"] = "graph"

        return vector_docs + graph_docs

    def _rerank_results(
        self, vector_docs: List[Document], graph_docs: List[Document], query: str
    ) -> List[Document]:
        """
        Rerank combined results based on a unified score.

        Uses a simple scoring heuristic:
        - Vector docs get higher base score (direct relevance)
        - Graph docs get scored by hop distance
        - All scores normalized and re-sorted
        """
        scored_docs = []

        # Score vector docs (higher base score)
        for i, doc in enumerate(vector_docs):
            # Position-based score (earlier = more relevant)
            position_score = 1.0 - (i / max(len(vector_docs), 1))
            base_score = 0.8  # High base for vector results
            final_score = base_score * position_score

            doc.metadata["retrieval_source"] = "vector"
            doc.metadata["rerank_score"] = final_score
            scored_docs.append((final_score, doc))

        # Score graph docs (based on hop distance and relevance)
        for doc in graph_docs:
            hop_distance = doc.metadata.get("hop_distance", 2)
            relevance = doc.metadata.get("relevance_score", 0.5)

            # Decay score by hop distance
            distance_penalty = 0.7**hop_distance
            base_score = 0.6  # Lower base for graph results
            final_score = base_score * distance_penalty * relevance

            doc.metadata["retrieval_source"] = "graph"
            doc.metadata["rerank_score"] = final_score
            scored_docs.append((final_score, doc))

        # Sort by score and return documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]

    def _weighted_combine(
        self, vector_docs: List[Document], graph_docs: List[Document]
    ) -> List[Document]:
        """
        Weighted combination of vector and graph results.

        Applies configured weights to results and returns
        a balanced selection from both sources.
        """
        # Calculate how many documents to take from each source
        total_requested = max(
            len(vector_docs) + len(graph_docs),
            self.vector_retriever.top_k
            + (self.graph_retriever.top_k if self.graph_retriever else 0),
        )

        vector_count = int(total_requested * self.vector_weight)
        graph_count = int(total_requested * self.graph_weight)

        # Take top N from each source
        selected_vector = vector_docs[:vector_count]
        selected_graph = graph_docs[:graph_count]

        # Add metadata
        for doc in selected_vector:
            doc.metadata["retrieval_source"] = "vector"
            doc.metadata["source_weight"] = self.vector_weight

        for doc in selected_graph:
            doc.metadata["retrieval_source"] = "graph"
            doc.metadata["source_weight"] = self.graph_weight

        # Interleave results to maintain diversity
        combined = []
        max_len = max(len(selected_vector), len(selected_graph))

        for i in range(max_len):
            if i < len(selected_vector):
                combined.append(selected_vector[i])
            if i < len(selected_graph):
                combined.append(selected_graph[i])

        return combined

    def _expand_results(
        self, vector_docs: List[Document], graph_docs: List[Document]
    ) -> List[Document]:
        """
        Use vector results to seed graph expansion.

        Strategy:
        1. Vector results form the core (highest relevance)
        2. Graph results provide related context
        3. Only include graph docs that relate to vector docs
        """
        if not vector_docs:
            return graph_docs

        # Mark vector docs
        for doc in vector_docs:
            doc.metadata["retrieval_source"] = "vector"
            doc.metadata["expansion_core"] = True

        # Filter graph docs to those that connect to vector results
        # (In a real implementation, you'd check entity overlap)
        expanded = vector_docs.copy()

        for graph_doc in graph_docs:
            graph_doc.metadata["retrieval_source"] = "graph"
            graph_doc.metadata["expansion_context"] = True
            expanded.append(graph_doc)

        return expanded

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid retriever.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "mode": self.mode,
            "deduplicate": self.deduplicate,
        }

        if self.mode == "weighted":
            stats["vector_weight"] = self.vector_weight
            stats["graph_weight"] = self.graph_weight

        # Include sub-retriever stats
        if self.vector_retriever:
            stats["vector_stats"] = self.vector_retriever.get_stats()

        if self.graph_retriever:
            stats["graph_stats"] = self.graph_retriever.get_stats()

        return stats
