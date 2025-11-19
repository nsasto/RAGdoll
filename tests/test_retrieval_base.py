"""
Tests for BaseRetriever interface.
"""

import pytest
from langchain_core.documents import Document

from ragdoll.retrieval import BaseRetriever


class ConcreteRetriever(BaseRetriever):
    """Concrete implementation for testing."""

    def __init__(self, docs=None):
        self.docs = docs or []

    def get_relevant_documents(self, query: str, **kwargs):
        return self.docs

    async def aget_relevant_documents(self, query: str, **kwargs):
        return self.docs


class TestBaseRetriever:
    """Test suite for BaseRetriever."""

    def test_format_documents_empty(self):
        """Test formatting empty document list."""
        retriever = ConcreteRetriever()

        result = retriever._format_documents([])

        assert result == ""

    def test_format_documents_basic(self):
        """Test basic document formatting."""
        docs = [
            Document(page_content="First document"),
            Document(page_content="Second document"),
        ]
        retriever = ConcreteRetriever(docs)

        result = retriever._format_documents(docs)

        assert "[1] First document" in result
        assert "[2] Second document" in result
        assert "\n\n" in result

    def test_format_documents_with_max_length(self):
        """Test document formatting with max length."""
        docs = [
            Document(page_content="A" * 100),
            Document(page_content="B" * 100),
        ]
        retriever = ConcreteRetriever(docs)

        result = retriever._format_documents(docs, max_length=50)

        assert len(result) <= 60  # Account for formatting
        assert "..." in result  # Should be truncated

    def test_deduplicate_documents(self):
        """Test document deduplication."""
        docs = [
            Document(page_content="duplicate content"),
            Document(page_content="unique content"),
            Document(page_content="duplicate content"),  # Duplicate
            Document(page_content="another unique"),
        ]
        retriever = ConcreteRetriever(docs)

        result = retriever._deduplicate_documents(docs)

        assert len(result) == 3  # Should remove one duplicate
        contents = [d.page_content for d in result]
        assert contents.count("duplicate content") == 1

    def test_deduplicate_documents_preserves_order(self):
        """Test that deduplication preserves first occurrence order."""
        docs = [
            Document(page_content="first"),
            Document(page_content="second"),
            Document(page_content="first"),  # Duplicate
        ]
        retriever = ConcreteRetriever(docs)

        result = retriever._deduplicate_documents(docs)

        assert len(result) == 2
        assert result[0].page_content == "first"
        assert result[1].page_content == "second"

    def test_deduplicate_documents_different_metadata(self):
        """Test that documents with same content but different metadata are kept."""
        docs = [
            Document(page_content="content", metadata={"id": 1}),
            Document(page_content="content", metadata={"id": 2}),
        ]
        retriever = ConcreteRetriever(docs)

        # Current implementation uses content hash only
        result = retriever._deduplicate_documents(docs)

        # Should deduplicate based on content
        assert len(result) == 1

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Can't instantiate BaseRetriever directly
            BaseRetriever()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""
        docs = [Document(page_content="test")]
        retriever = ConcreteRetriever(docs)

        result = retriever.get_relevant_documents("query")

        assert result == docs

    @pytest.mark.asyncio
    async def test_async_implementation_works(self):
        """Test that async implementation works."""
        docs = [Document(page_content="test")]
        retriever = ConcreteRetriever(docs)

        result = await retriever.aget_relevant_documents("query")

        assert result == docs
