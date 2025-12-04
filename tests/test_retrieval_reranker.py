"""
Tests for RerankerRetriever

Test reranking functionality with LLM, Cohere, and cross-encoder strategies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document

from ragdoll.retrieval.reranker import RerankerRetriever
from ragdoll.retrieval.base import BaseRetriever


@pytest.fixture
def mock_base_retriever():
    """Create a mock base retriever that returns test documents."""
    retriever = Mock(spec=BaseRetriever)
    retriever.get_relevant_documents.return_value = [
        Document(page_content="Very relevant content", metadata={"id": "1"}),
        Document(page_content="Somewhat relevant content", metadata={"id": "2"}),
        Document(page_content="Barely relevant content", metadata={"id": "3"}),
        Document(page_content="Irrelevant content", metadata={"id": "4"}),
        Document(page_content="Completely off-topic", metadata={"id": "5"}),
    ]
    return retriever


@pytest.fixture
def mock_llm_caller():
    """Create a mock LLM caller that returns scores."""

    def _create_caller():
        caller = Mock()
        # Return decreasing scores for documents
        caller.invoke.side_effect = ["9", "7", "5", "3", "1"]
        return caller

    return _create_caller()


class TestRerankerRetriever:
    """Test suite for RerankerRetriever."""

    def test_initialization_llm_provider(self, mock_base_retriever):
        """Test initialization with LLM provider."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=3,
        )

        assert reranker.provider == "llm"
        assert reranker.top_k == 3
        assert reranker.base_retriever == mock_base_retriever

    def test_initialization_cohere_provider(self, mock_base_retriever):
        """Test initialization with Cohere provider."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="cohere",
            top_k=5,
        )

        assert reranker.provider == "cohere"
        assert reranker.top_k == 5

    def test_initialization_cross_encoder_provider(self, mock_base_retriever):
        """Test initialization with cross-encoder provider."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="cross-encoder",
            top_k=4,
        )

        assert reranker.provider == "cross-encoder"
        assert reranker.top_k == 4

    def test_initialization_invalid_provider(self, mock_base_retriever):
        """Test initialization with invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            RerankerRetriever(
                base_retriever=mock_base_retriever,
                provider="invalid",
            )

    def test_get_relevant_documents_over_retrieves(self, mock_base_retriever):
        """Test that retriever over-retrieves before reranking."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=3,
            over_retrieve_multiplier=2,
        )

        # Mock LLM to avoid actual calls
        reranker._reranker_llm = Mock()
        reranker._reranker_llm.invoke.return_value = "5"

        reranker.get_relevant_documents("test query")

        # Should request 3 * 2 = 6 documents
        mock_base_retriever.get_relevant_documents.assert_called_once_with(
            "test query", top_k=6
        )

    def test_llm_reranking_sorts_by_score(self, mock_base_retriever, mock_llm_caller):
        """Test LLM reranking sorts documents by relevance score."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            reranker_llm=mock_llm_caller,
            provider="llm",
            top_k=3,
            over_retrieve_multiplier=1,  # Don't over-retrieve for this test
        )

        docs = reranker.get_relevant_documents("test query")

        # Should return top 3 highest scoring documents
        assert len(docs) == 3
        assert docs[0].metadata["id"] == "1"  # Score 9
        assert docs[1].metadata["id"] == "2"  # Score 7
        assert docs[2].metadata["id"] == "3"  # Score 5

        # Check that rerank scores were added to metadata
        assert "rerank_score" in docs[0].metadata
        assert docs[0].metadata["rerank_score"] == 0.9  # Normalized from 9

    def test_llm_reranking_filters_by_threshold(self, mock_base_retriever):
        """Test LLM reranking filters documents below threshold."""
        # Create fresh mock LLM with scores for all 5 documents
        mock_llm = Mock()
        mock_llm.invoke.side_effect = ["9", "7", "5", "3", "1"]

        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            reranker_llm=mock_llm,
            provider="llm",
            top_k=5,
            score_threshold=0.6,  # Only keep scores >= 0.6
            over_retrieve_multiplier=1,
        )

        docs = reranker.get_relevant_documents("test query")

        # Should only return docs with score >= 6 (0.6 normalized)
        assert len(docs) == 2
        assert docs[0].metadata["id"] == "1"  # Score 9
        assert docs[1].metadata["id"] == "2"  # Score 7

    def test_returns_all_if_fewer_than_top_k(self, mock_base_retriever):
        """Test that all documents are returned if fewer than top_k."""
        # Mock base retriever to return only 2 documents
        mock_base_retriever.get_relevant_documents.return_value = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]

        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=5,
        )

        docs = reranker.get_relevant_documents("test query")

        # Should return all 2 documents without reranking
        assert len(docs) == 2

    def test_returns_empty_if_no_documents(self, mock_base_retriever):
        """Test that empty list is returned if base retriever returns nothing."""
        mock_base_retriever.get_relevant_documents.return_value = []

        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=5,
        )

        docs = reranker.get_relevant_documents("test query")

        assert len(docs) == 0

    def test_fallback_on_reranking_error(self, mock_base_retriever):
        """Test that original documents are returned on reranking error."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=3,
            over_retrieve_multiplier=1,
        )

        # Mock LLM to raise exception
        reranker._reranker_llm = Mock()
        reranker._reranker_llm.invoke.side_effect = Exception("LLM error")

        docs = reranker.get_relevant_documents("test query")

        # Should return first 3 documents without reranking
        assert len(docs) == 3
        assert docs[0].metadata["id"] == "1"

    def test_runtime_top_k_override(self, mock_base_retriever, mock_llm_caller):
        """Test that top_k can be overridden at runtime."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            reranker_llm=mock_llm_caller,
            provider="llm",
            top_k=3,
            over_retrieve_multiplier=2,
        )

        docs = reranker.get_relevant_documents("test query", top_k=2)

        # Should return only 2 documents
        assert len(docs) == 2

    def test_cohere_reranking_without_client(self, mock_base_retriever):
        """Test Cohere reranking falls back gracefully without client."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="cohere",
            top_k=3,
            over_retrieve_multiplier=1,
        )

        # Ensure Cohere client is None (not installed)
        reranker._cohere_client = None

        docs = reranker.get_relevant_documents("test query")

        # Should return documents with default score
        assert len(docs) == 3

    def test_cross_encoder_reranking_without_model(self, mock_base_retriever):
        """Test cross-encoder reranking falls back gracefully without model."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="cross-encoder",
            top_k=3,
            over_retrieve_multiplier=1,
        )

        # Ensure cross-encoder is None (not installed)
        reranker._cross_encoder = None

        docs = reranker.get_relevant_documents("test query")

        # Should return documents with default score
        assert len(docs) == 3

    def test_get_stats(self, mock_base_retriever):
        """Test get_stats returns retriever statistics."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=5,
            over_retrieve_multiplier=2,
            score_threshold=0.3,
        )

        stats = reranker.get_stats()

        assert stats["provider"] == "llm"
        assert stats["top_k"] == 5
        assert stats["over_retrieve_multiplier"] == 2
        assert stats["score_threshold"] == 0.3

    def test_async_get_relevant_documents(self, mock_base_retriever, mock_llm_caller):
        """Test async version delegates to sync version."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            reranker_llm=mock_llm_caller,
            provider="llm",
            top_k=3,
        )

        # Call async version
        import asyncio

        docs = asyncio.run(reranker.aget_relevant_documents("test query"))

        # Should return documents
        assert len(docs) == 3

    def test_log_scores_enabled(self, mock_base_retriever, mock_llm_caller, caplog):
        """Test that scores are logged when log_scores is True."""
        import logging

        caplog.set_level(logging.DEBUG)

        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            reranker_llm=mock_llm_caller,
            provider="llm",
            top_k=3,
            log_scores=True,
            over_retrieve_multiplier=1,
        )

        reranker.get_relevant_documents("test query")

        # Check that debug logs contain score information
        assert any("Rerank score" in record.message for record in caplog.records)

    def test_extracts_numeric_score_from_text(self, mock_base_retriever):
        """Test that numeric scores are extracted from various LLM response formats."""
        reranker = RerankerRetriever(
            base_retriever=mock_base_retriever,
            provider="llm",
            top_k=3,
            over_retrieve_multiplier=1,
        )

        # Mock LLM with various response formats
        mock_llm = Mock()
        reranker._reranker_llm = mock_llm

        # Test various response formats
        test_cases = [
            ("8", 0.8),
            ("Score: 7", 0.7),
            ("The relevance is 6/10", 0.6),
            ("9.5", 0.95),
            ("Relevance: 5.5 out of 10", 0.55),
        ]

        for response, expected_score in test_cases:
            mock_llm.invoke.return_value = response
            docs = reranker.get_relevant_documents("test query")

            # Check first document got the expected score
            assert docs[0].metadata["rerank_score"] == pytest.approx(
                expected_score, abs=0.01
            )
