"""
Tests for HybridRetriever.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from langchain_core.documents import Document

from ragdoll.retrieval import HybridRetriever, VectorRetriever, GraphRetriever


class TestHybridRetriever:
    """Test suite for HybridRetriever."""

    @pytest.fixture
    def mock_vector_retriever(self):
        """Create a mock vector retriever."""
        retriever = Mock(spec=VectorRetriever)
        retriever.get_relevant_documents = Mock(
            return_value=[
                Document(page_content="vector doc 1", metadata={"id": 1}),
                Document(page_content="vector doc 2", metadata={"id": 2}),
            ]
        )
        retriever.aget_relevant_documents = AsyncMock(
            return_value=[
                Document(page_content="vector doc 1", metadata={"id": 1}),
                Document(page_content="vector doc 2", metadata={"id": 2}),
            ]
        )
        retriever.get_stats = Mock(return_value={"top_k": 3})
        retriever.top_k = 3
        return retriever

    @pytest.fixture
    def mock_graph_retriever(self):
        """Create a mock graph retriever."""
        retriever = Mock(spec=GraphRetriever)
        retriever.get_relevant_documents = Mock(
            return_value=[
                Document(
                    page_content="graph doc 1",
                    metadata={"id": 3, "hop_distance": 0, "relevance_score": 1.0},
                ),
                Document(
                    page_content="graph doc 2",
                    metadata={"id": 4, "hop_distance": 1, "relevance_score": 0.7},
                ),
            ]
        )
        retriever.aget_relevant_documents = AsyncMock(
            return_value=[
                Document(
                    page_content="graph doc 1",
                    metadata={"id": 3, "hop_distance": 0, "relevance_score": 1.0},
                ),
            ]
        )
        retriever.get_stats = Mock(return_value={"top_k": 5})
        retriever.top_k = 5
        return retriever

    def test_init_default_params(self, mock_vector_retriever):
        """Test initialization with default parameters."""
        retriever = HybridRetriever(vector_retriever=mock_vector_retriever)

        assert retriever.vector_retriever == mock_vector_retriever
        assert retriever.graph_retriever is None
        assert retriever.mode == "concat"
        assert retriever.vector_weight == 0.6
        assert retriever.graph_weight == 0.4
        assert retriever.deduplicate is True

    def test_init_custom_params(self, mock_vector_retriever, mock_graph_retriever):
        """Test initialization with custom parameters."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="rerank",
            vector_weight=0.7,
            graph_weight=0.3,
            deduplicate=False,
        )

        assert retriever.mode == "rerank"
        assert retriever.vector_weight == 0.7
        assert retriever.graph_weight == 0.3
        assert retriever.deduplicate is False

    def test_init_weight_normalization(self, mock_vector_retriever):
        """Test automatic weight normalization."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            mode="weighted",
            vector_weight=2.0,
            graph_weight=1.0,
        )

        # Weights should be normalized to sum to 1.0
        assert abs(retriever.vector_weight + retriever.graph_weight - 1.0) < 0.01

    def test_get_relevant_documents_vector_only(self, mock_vector_retriever):
        """Test retrieval with vector-only (no graph retriever)."""
        retriever = HybridRetriever(vector_retriever=mock_vector_retriever)

        docs = retriever.get_relevant_documents("test query")

        assert len(docs) == 2
        assert docs[0].page_content == "vector doc 1"
        mock_vector_retriever.get_relevant_documents.assert_called_once()

    def test_get_relevant_documents_concat(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test concat combination mode."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="concat",
        )

        docs = retriever.get_relevant_documents("test query")

        # Should have both vector and graph results
        assert len(docs) == 4
        assert docs[0].page_content == "vector doc 1"
        assert docs[2].page_content == "graph doc 1"
        # Check metadata added
        assert docs[0].metadata["retrieval_source"] == "vector"
        assert docs[2].metadata["retrieval_source"] == "graph"

    def test_get_relevant_documents_rerank(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test rerank combination mode."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="rerank",
        )

        docs = retriever.get_relevant_documents("test query")

        # Should have all results, re-sorted by score
        assert len(docs) == 4
        # Check that rerank_score was added
        assert "rerank_score" in docs[0].metadata
        # Results should be sorted by rerank_score
        scores = [d.metadata["rerank_score"] for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_get_relevant_documents_weighted(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test weighted combination mode."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="weighted",
            vector_weight=0.7,
            graph_weight=0.3,
        )

        docs = retriever.get_relevant_documents("test query")

        # Should have weighted selection from both sources
        assert len(docs) > 0
        # Check that source_weight was added
        vector_docs = [
            d for d in docs if d.metadata.get("retrieval_source") == "vector"
        ]
        graph_docs = [d for d in docs if d.metadata.get("retrieval_source") == "graph"]

        if vector_docs:
            assert vector_docs[0].metadata["source_weight"] == 0.7
        if graph_docs:
            assert graph_docs[0].metadata["source_weight"] == 0.3

    def test_get_relevant_documents_expand(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test expand combination mode."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="expand",
        )

        docs = retriever.get_relevant_documents("test query")

        # Should have vector core + graph context
        assert len(docs) == 4
        # Vector docs should be marked as core
        vector_docs = [d for d in docs if d.metadata.get("expansion_core")]
        assert len(vector_docs) == 2
        # Graph docs should be marked as context
        graph_docs = [d for d in docs if d.metadata.get("expansion_context")]
        assert len(graph_docs) == 2

    def test_get_relevant_documents_deduplicate(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test deduplication."""
        # Make graph return duplicate of vector doc
        mock_graph_retriever.get_relevant_documents = Mock(
            return_value=[
                Document(page_content="vector doc 1", metadata={"id": 1}),  # Duplicate
                Document(page_content="unique graph doc", metadata={"id": 5}),
            ]
        )

        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="concat",
            deduplicate=True,
        )

        docs = retriever.get_relevant_documents("test query")

        # Should have deduplicated results
        assert len(docs) == 3  # 2 vector + 1 unique graph
        contents = [d.page_content for d in docs]
        assert contents.count("vector doc 1") == 1

    def test_get_relevant_documents_no_deduplicate(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test without deduplication."""
        mock_graph_retriever.get_relevant_documents = Mock(
            return_value=[
                Document(page_content="vector doc 1", metadata={"id": 1}),  # Duplicate
            ]
        )

        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="concat",
            deduplicate=False,
        )

        docs = retriever.get_relevant_documents("test query")

        # Should keep duplicates
        contents = [d.page_content for d in docs]
        assert contents.count("vector doc 1") == 2

    @pytest.mark.asyncio
    async def test_aget_relevant_documents(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test async retrieval."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="concat",
        )

        docs = await retriever.aget_relevant_documents("test query")

        assert len(docs) == 3  # 2 vector + 1 graph (from async mocks)
        mock_vector_retriever.aget_relevant_documents.assert_called_once()
        mock_graph_retriever.aget_relevant_documents.assert_called_once()

    def test_get_stats(self, mock_vector_retriever, mock_graph_retriever):
        """Test retriever statistics."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            mode="weighted",
            vector_weight=0.6,
            graph_weight=0.4,
        )

        stats = retriever.get_stats()

        assert stats["mode"] == "weighted"
        assert stats["deduplicate"] is True
        assert stats["vector_weight"] == 0.6
        assert stats["graph_weight"] == 0.4
        assert "vector_stats" in stats
        assert "graph_stats" in stats

    def test_concat_results_metadata(self, mock_vector_retriever, mock_graph_retriever):
        """Test concat adds proper metadata."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever, graph_retriever=mock_graph_retriever
        )

        vector_docs = [Document(page_content="v1", metadata={})]
        graph_docs = [Document(page_content="g1", metadata={})]

        result = retriever._concat_results(vector_docs, graph_docs)

        assert result[0].metadata["retrieval_source"] == "vector"
        assert result[1].metadata["retrieval_source"] == "graph"

    def test_runtime_parameter_override(
        self, mock_vector_retriever, mock_graph_retriever
    ):
        """Test runtime parameter override."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever, graph_retriever=mock_graph_retriever
        )

        docs = retriever.get_relevant_documents("test", top_k=10)

        # Should pass kwargs to underlying retrievers
        mock_vector_retriever.get_relevant_documents.assert_called_with(
            "test", top_k=10
        )
