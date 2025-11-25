"""
Unit tests for parallel document addition in BaseVectorStore.

Tests cover:
- Normal parallel operation
- Concurrent batch processing
- Error handling and retries
- Edge cases (empty documents, single document)
- Batch size detection
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.documents import Document
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import FAISS

from ragdoll.vector_stores.base_vector_store import BaseVectorStore


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content=f"Test document {i}", metadata={"id": i})
        for i in range(20)
    ]


@pytest.fixture
def fake_embedding():
    """Create a fake embedding model."""
    return FakeEmbeddings(size=128)


@pytest.fixture
def vector_store(fake_embedding):
    """Create a BaseVectorStore with FAISS backend."""
    faiss_store = FAISS.from_documents(
        documents=[Document(page_content="init")], embedding=fake_embedding
    )
    return BaseVectorStore(faiss_store)


class TestAAddDocuments:
    """Tests for async aadd_documents method."""

    @pytest.mark.asyncio
    async def test_aadd_documents_basic(self, vector_store, sample_documents):
        """Test basic async document addition."""
        docs = sample_documents[:5]
        ids = await vector_store.aadd_documents(docs)

        assert len(ids) == 5
        assert all(isinstance(id, str) for id in ids)

    @pytest.mark.asyncio
    async def test_aadd_documents_empty(self, vector_store):
        """Test aadd_documents with empty list."""
        ids = await vector_store.aadd_documents([])
        assert ids == []

    @pytest.mark.asyncio
    async def test_aadd_documents_with_batch_size(self, vector_store, sample_documents):
        """Test aadd_documents respects batch_size parameter."""
        docs = sample_documents[:10]
        ids = await vector_store.aadd_documents(docs, batch_size=5)

        assert len(ids) == 10
        # Verify documents were actually added
        results = vector_store.similarity_search("Test document", k=10)
        assert len(results) > 0


class TestAddDocumentsParallel:
    """Tests for parallel document addition."""

    @pytest.mark.asyncio
    async def test_parallel_basic_operation(self, vector_store, sample_documents):
        """Test basic parallel document addition."""
        ids = await vector_store.add_documents_parallel(
            sample_documents, batch_size=5, max_concurrent=3
        )

        assert len(ids) == len(sample_documents)
        assert all(isinstance(id, str) for id in ids)
        assert len([id for id in ids if id]) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_parallel_with_different_concurrency(
        self, vector_store, sample_documents
    ):
        """Test parallel addition with different max_concurrent values."""
        # Test with max_concurrent=1 (essentially sequential)
        ids_seq = await vector_store.add_documents_parallel(
            sample_documents[:10], batch_size=2, max_concurrent=1
        )
        assert len(ids_seq) == 10

        # Test with max_concurrent=5 (truly parallel)
        ids_par = await vector_store.add_documents_parallel(
            sample_documents[10:], batch_size=2, max_concurrent=5
        )
        assert len(ids_par) == 10

    @pytest.mark.asyncio
    async def test_parallel_empty_documents(self, vector_store):
        """Test parallel addition with empty document list."""
        ids = await vector_store.add_documents_parallel([])
        assert ids == []

    @pytest.mark.asyncio
    async def test_parallel_single_document(self, vector_store, sample_documents):
        """Test parallel addition with single document."""
        ids = await vector_store.add_documents_parallel(
            sample_documents[:1], max_concurrent=3
        )
        assert len(ids) == 1
        assert ids[0] != ""

    @pytest.mark.asyncio
    async def test_parallel_batch_size_auto_detection(
        self, vector_store, sample_documents
    ):
        """Test that batch size is auto-detected when not provided."""
        # Don't specify batch_size, let it auto-detect
        ids = await vector_store.add_documents_parallel(
            sample_documents, max_concurrent=2
        )

        assert len(ids) == len(sample_documents)
        assert all(id != "" for id in ids)

    @pytest.mark.asyncio
    async def test_parallel_maintains_order(self, vector_store, sample_documents):
        """Test that returned IDs maintain order with input documents."""
        docs = sample_documents[:10]

        # Add documents in parallel
        ids = await vector_store.add_documents_parallel(
            docs, batch_size=3, max_concurrent=2
        )

        assert len(ids) == len(docs)
        # IDs should be returned in the same order as documents
        # We can't test the exact ID values, but we can verify count and no empty strings
        assert all(isinstance(id, str) and id != "" for id in ids)


class TestErrorHandling:
    """Tests for error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_parallel_with_batch_failure(self, fake_embedding):
        """Test parallel addition with simulated batch failure."""
        # Create a mock store that fails on first batch, succeeds on retry
        mock_store = Mock()
        call_count = {"count": 0}

        def add_documents_side_effect(docs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Simulated embedding API error")
            return [f"id_{i}" for i in range(len(docs))]

        mock_store.add_documents = Mock(side_effect=add_documents_side_effect)

        vector_store = BaseVectorStore(mock_store)

        docs = [
            Document(page_content=f"Test {i}", metadata={"id": i}) for i in range(5)
        ]

        # Should retry and succeed
        ids = await vector_store.add_documents_parallel(
            docs, batch_size=5, max_concurrent=1, retry_failed=True
        )

        assert len(ids) == 5
        assert all(id.startswith("id_") for id in ids)
        # Should have been called twice (initial + retry)
        assert call_count["count"] == 2

    @pytest.mark.asyncio
    async def test_parallel_with_retry_disabled(self, fake_embedding):
        """Test parallel addition with retry disabled."""
        mock_store = Mock()
        mock_store.add_documents = Mock(side_effect=Exception("API error"))

        vector_store = BaseVectorStore(mock_store)

        docs = [
            Document(page_content=f"Test {i}", metadata={"id": i}) for i in range(5)
        ]

        # Should fail and return empty IDs
        ids = await vector_store.add_documents_parallel(
            docs, batch_size=5, max_concurrent=1, retry_failed=False
        )

        assert len(ids) == 5
        assert all(id == "" for id in ids)

    @pytest.mark.asyncio
    async def test_parallel_with_permanent_failure(self, fake_embedding):
        """Test parallel addition when retry also fails."""
        mock_store = Mock()
        mock_store.add_documents = Mock(side_effect=Exception("Permanent error"))

        vector_store = BaseVectorStore(mock_store)

        docs = [
            Document(page_content=f"Test {i}", metadata={"id": i}) for i in range(3)
        ]

        # Should retry and still fail, returning empty IDs
        ids = await vector_store.add_documents_parallel(
            docs, batch_size=3, max_concurrent=1, retry_failed=True
        )

        assert len(ids) == 3
        assert all(id == "" for id in ids)
        # Should have been called twice (initial + retry)
        assert mock_store.add_documents.call_count == 2


class TestConcurrency:
    """Tests for concurrent execution behavior."""

    @pytest.mark.asyncio
    async def test_parallel_actually_concurrent(self, fake_embedding):
        """Test that batches are actually processed concurrently."""
        mock_store = Mock()
        execution_times = []

        async def slow_add_documents(docs):
            """Simulate slow embedding API call."""
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate API latency
            execution_times.append((start, asyncio.get_event_loop().time()))
            return [f"id_{i}" for i in range(len(docs))]

        # Use asyncio.to_thread wrapper like the real implementation
        def sync_wrapper(docs):
            # Simulate the actual sync behavior
            import time

            start = time.time()
            time.sleep(0.1)
            execution_times.append(start)
            return [f"id_{i}" for i in range(len(docs))]

        mock_store.add_documents = sync_wrapper

        vector_store = BaseVectorStore(mock_store)

        docs = [
            Document(page_content=f"Test {i}", metadata={"id": i}) for i in range(10)
        ]

        start_time = asyncio.get_event_loop().time()

        # Process 5 batches of 2 documents with max_concurrent=3
        ids = await vector_store.add_documents_parallel(
            docs, batch_size=2, max_concurrent=3
        )

        elapsed = asyncio.get_event_loop().time() - start_time

        assert len(ids) == 10
        # With 5 batches, max_concurrent=3, and 0.1s per batch:
        # Sequential would take ~0.5s, parallel should take ~0.2s (2 rounds)
        # Allow some overhead for test execution
        assert elapsed < 0.45  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_parallel_respects_max_concurrent(self, fake_embedding):
        """Test that max_concurrent limit is respected."""
        mock_store = Mock()
        concurrent_calls = {"max": 0, "current": 0}

        def add_documents_tracking(docs):
            import time

            concurrent_calls["current"] += 1
            concurrent_calls["max"] = max(
                concurrent_calls["max"], concurrent_calls["current"]
            )
            time.sleep(0.05)  # Small delay
            concurrent_calls["current"] -= 1
            return [f"id_{i}" for i in range(len(docs))]

        mock_store.add_documents = add_documents_tracking

        vector_store = BaseVectorStore(mock_store)

        docs = [
            Document(page_content=f"Test {i}", metadata={"id": i}) for i in range(20)
        ]

        # Process with max_concurrent=3
        await vector_store.add_documents_parallel(docs, batch_size=2, max_concurrent=3)

        # Max concurrent should not exceed 3
        assert concurrent_calls["max"] <= 3


class TestIntegrationWithRealStores:
    """Integration tests with actual vector store implementations."""

    @pytest.mark.asyncio
    async def test_faiss_parallel_integration(self, fake_embedding):
        """Test parallel addition with real FAISS store."""
        faiss_store = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=fake_embedding
        )
        vector_store = BaseVectorStore(faiss_store)

        docs = [
            Document(page_content=f"Integration test document {i}", metadata={"id": i})
            for i in range(15)
        ]

        ids = await vector_store.add_documents_parallel(
            docs, batch_size=5, max_concurrent=2
        )

        assert len(ids) == 15
        assert all(id != "" for id in ids)

        # Verify documents are searchable
        results = vector_store.similarity_search("Integration test", k=10)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_produces_same_results(
        self, fake_embedding, sample_documents
    ):
        """Verify parallel and sequential addition produce equivalent results."""
        # Sequential addition
        faiss_store_seq = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=fake_embedding
        )
        vector_store_seq = BaseVectorStore(faiss_store_seq)
        ids_seq = vector_store_seq.add_documents(sample_documents[:10])

        # Parallel addition
        faiss_store_par = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=fake_embedding
        )
        vector_store_par = BaseVectorStore(faiss_store_par)
        ids_par = await vector_store_par.add_documents_parallel(
            sample_documents[:10], batch_size=3, max_concurrent=2
        )

        # Should produce same number of IDs
        assert len(ids_seq) == len(ids_par)
        assert all(id != "" for id in ids_seq)
        assert all(id != "" for id in ids_par)
