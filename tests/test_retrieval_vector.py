"""
Tests for VectorRetriever.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from langchain_core.documents import Document

from ragdoll.retrieval import VectorRetriever


class TestVectorRetriever:
    """Test suite for VectorRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock()
        store.similarity_search = Mock(
            return_value=[
                Document(page_content="doc1", metadata={"id": 1}),
                Document(page_content="doc2", metadata={"id": 2}),
            ]
        )
        store.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content="doc1", metadata={"id": 1}),
                Document(page_content="doc2", metadata={"id": 2}),
            ]
        )
        store.max_marginal_relevance_search = Mock(
            return_value=[
                Document(page_content="doc3", metadata={"id": 3}),
            ]
        )
        store.similarity_search_with_relevance_scores = Mock(
            return_value=[
                (Document(page_content="doc4", metadata={"id": 4}), 0.9),
                (Document(page_content="doc5", metadata={"id": 5}), 0.8),
            ]
        )
        return store

    def test_init_default_params(self, mock_vector_store):
        """Test initialization with default parameters."""
        retriever = VectorRetriever(vector_store=mock_vector_store)

        assert retriever.vector_store == mock_vector_store
        assert retriever.top_k == 3
        assert retriever.search_type == "similarity"
        assert retriever.search_kwargs == {}

    def test_init_custom_params(self, mock_vector_store):
        """Test initialization with custom parameters."""
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            top_k=10,
            search_type="mmr",
            search_kwargs={"fetch_k": 20},
        )

        assert retriever.top_k == 10
        assert retriever.search_type == "mmr"
        assert retriever.search_kwargs == {"fetch_k": 20}

    def test_get_relevant_documents_similarity(self, mock_vector_store):
        """Test retrieval with similarity search."""
        retriever = VectorRetriever(
            vector_store=mock_vector_store, top_k=2, search_type="similarity"
        )

        docs = retriever.get_relevant_documents("test query")

        assert len(docs) == 2
        assert docs[0].page_content == "doc1"
        assert docs[1].page_content == "doc2"
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=2)

    def test_get_relevant_documents_mmr(self, mock_vector_store):
        """Test retrieval with MMR search."""
        retriever = VectorRetriever(
            vector_store=mock_vector_store, top_k=5, search_type="mmr"
        )

        docs = retriever.get_relevant_documents("test query")

        assert len(docs) == 1
        assert docs[0].page_content == "doc3"
        mock_vector_store.max_marginal_relevance_search.assert_called_once()

    def test_get_relevant_documents_threshold(self, mock_vector_store):
        """Test retrieval with score threshold."""
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            top_k=5,
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7},
        )

        docs = retriever.get_relevant_documents("test query")

        assert len(docs) == 2
        assert docs[0].page_content == "doc4"
        assert docs[1].page_content == "doc5"

    def test_get_relevant_documents_runtime_override(self, mock_vector_store):
        """Test runtime parameter override."""
        retriever = VectorRetriever(vector_store=mock_vector_store, top_k=3)

        docs = retriever.get_relevant_documents("test query", top_k=10)

        mock_vector_store.similarity_search.assert_called_once_with("test query", k=10)

    def test_get_relevant_documents_no_store(self):
        """Test retrieval with no vector store."""
        retriever = VectorRetriever(vector_store=None)

        docs = retriever.get_relevant_documents("test query")

        assert docs == []

    def test_get_relevant_documents_error_handling(self, mock_vector_store):
        """Test error handling during retrieval."""
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")

        retriever = VectorRetriever(vector_store=mock_vector_store)
        docs = retriever.get_relevant_documents("test query")

        assert docs == []

    @pytest.mark.asyncio
    async def test_aget_relevant_documents(self, mock_vector_store):
        """Test async retrieval."""
        retriever = VectorRetriever(vector_store=mock_vector_store, top_k=2)

        docs = await retriever.aget_relevant_documents("test query")

        assert len(docs) == 2
        assert docs[0].page_content == "doc1"
        mock_vector_store.asimilarity_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_error(self, mock_vector_store):
        """Test async error handling."""
        mock_vector_store.asimilarity_search.side_effect = Exception(
            "Async search failed"
        )

        retriever = VectorRetriever(vector_store=mock_vector_store)
        docs = await retriever.aget_relevant_documents("test query")

        assert docs == []

    def test_get_stats(self, mock_vector_store):
        """Test retriever statistics."""
        retriever = VectorRetriever(
            vector_store=mock_vector_store, top_k=5, search_type="mmr"
        )

        stats = retriever.get_stats()

        assert stats["top_k"] == 5
        assert stats["search_type"] == "mmr"

    def test_get_stats_with_document_count(self, mock_vector_store):
        """Test stats with document count."""
        mock_vector_store.__len__ = Mock(return_value=100)

        retriever = VectorRetriever(vector_store=mock_vector_store)
        stats = retriever.get_stats()

        assert stats["document_count"] == 100

    def test_get_stats_faiss_specific(self, mock_vector_store):
        """Test stats with FAISS-specific attributes."""
        mock_vector_store.index = Mock()
        mock_vector_store.index.ntotal = 250

        retriever = VectorRetriever(vector_store=mock_vector_store)
        stats = retriever.get_stats()

        assert stats["document_count"] == 250
