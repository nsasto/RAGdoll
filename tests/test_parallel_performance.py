"""
Performance demonstration test for parallel document addition.

This test demonstrates the performance improvement from parallel execution
and provides metrics to show the speedup achieved.

Run with: pytest tests/test_parallel_performance.py -v -s
"""

import time
import asyncio
import pytest
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings.fake import FakeEmbeddings

from ragdoll.vector_stores.base_vector_store import BaseVectorStore
from ragdoll.metrics.metrics_manager import MetricsManager


class SlowEmbeddings(FakeEmbeddings):
    """
    Fake embeddings with artificial latency to simulate remote API calls.

    This simulates real-world embedding services like OpenAI, Cohere, etc.
    where network latency is a significant factor.
    """

    model_config = {"extra": "allow"}  # Allow extra attributes

    def __init__(self, size: int = 1536, latency_ms: int = 100, **kwargs):
        super().__init__(size=size, **kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "call_count", 0)
        object.__setattr__(self, "concurrent_calls", {"max": 0, "current": 0})

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Simulate slow embedding generation with network latency."""
        self.call_count += 1
        self.concurrent_calls["current"] += 1
        self.concurrent_calls["max"] = max(
            self.concurrent_calls["max"], self.concurrent_calls["current"]
        )

        # Simulate network latency
        time.sleep(self.latency_ms / 1000.0)

        self.concurrent_calls["current"] -= 1
        return super().embed_documents(texts)


@pytest.fixture
def performance_documents():
    """Create a larger set of documents for performance testing."""
    return [
        Document(
            page_content=f"This is test document number {i}. "
            f"It contains some sample text to be embedded. "
            f"Document ID: {i}",
            metadata={"id": i, "type": "test"},
        )
        for i in range(50)  # 50 documents for meaningful performance comparison
    ]


class TestPerformanceComparison:
    """Performance comparison tests between sequential and parallel execution."""

    def test_sequential_baseline(self, performance_documents):
        """Establish baseline performance with sequential processing."""
        from langchain_community.vectorstores import FAISS

        print("\n" + "=" * 70)
        print("SEQUENTIAL PROCESSING BASELINE")
        print("=" * 70)

        # Create embeddings with simulated latency
        embeddings = SlowEmbeddings(size=128, latency_ms=50)

        # Create vector store
        faiss_store = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=embeddings
        )
        vector_store = BaseVectorStore(faiss_store)

        # Process documents sequentially
        start_time = time.time()
        ids = vector_store.add_documents(performance_documents, batch_size=10)
        duration = time.time() - start_time

        print(f"\nDocuments processed: {len(performance_documents)}")
        print(f"Batch size: 10")
        print(f"Total batches: {len(performance_documents) // 10}")
        print(f"Embedding API calls: {embeddings.call_count}")
        print(f"Total time: {duration:.2f}s")
        print(f"Time per document: {duration / len(performance_documents):.3f}s")
        print(f"Max concurrent calls: {embeddings.concurrent_calls['max']}")
        print(f"Documents added: {len(ids)}")

        assert len(ids) == len(performance_documents)
        assert embeddings.concurrent_calls["max"] == 1  # Sequential = no concurrency

    @pytest.mark.asyncio
    async def test_parallel_performance(self, performance_documents):
        """Demonstrate performance improvement with parallel processing."""
        from langchain_community.vectorstores import FAISS

        print("\n" + "=" * 70)
        print("PARALLEL PROCESSING WITH CONCURRENCY")
        print("=" * 70)

        # Create embeddings with simulated latency
        embeddings = SlowEmbeddings(size=128, latency_ms=50)

        # Create vector store
        faiss_store = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=embeddings
        )
        vector_store = BaseVectorStore(faiss_store)

        # Process documents in parallel
        start_time = time.time()
        ids = await vector_store.add_documents_parallel(
            performance_documents, batch_size=10, max_concurrent=5
        )
        duration = time.time() - start_time

        print(f"\nDocuments processed: {len(performance_documents)}")
        print(f"Batch size: 10")
        print(f"Max concurrent batches: 5")
        print(f"Total batches: {len(performance_documents) // 10}")
        print(f"Embedding API calls: {embeddings.call_count}")
        print(f"Total time: {duration:.2f}s")
        print(f"Time per document: {duration / len(performance_documents):.3f}s")
        print(f"Max concurrent calls observed: {embeddings.concurrent_calls['max']}")
        print(f"Documents added: {len(ids)}")

        assert len(ids) == len(performance_documents)
        # Should have processed multiple batches concurrently
        assert embeddings.concurrent_calls["max"] > 1

    @pytest.mark.asyncio
    async def test_side_by_side_comparison(self, performance_documents):
        """Side-by-side comparison of sequential vs parallel performance."""
        from langchain_community.vectorstores import FAISS

        print("\n" + "=" * 70)
        print("SIDE-BY-SIDE PERFORMANCE COMPARISON")
        print("=" * 70)

        # Test configuration
        batch_size = 10
        max_concurrent = 5
        latency_ms = 50

        # Sequential test
        print("\n--- Sequential Processing ---")
        embeddings_seq = SlowEmbeddings(size=128, latency_ms=latency_ms)
        faiss_store_seq = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=embeddings_seq
        )
        vector_store_seq = BaseVectorStore(faiss_store_seq)

        start_seq = time.time()
        ids_seq = vector_store_seq.add_documents(
            performance_documents, batch_size=batch_size
        )
        duration_seq = time.time() - start_seq

        print(f"Time: {duration_seq:.2f}s")
        print(f"API calls: {embeddings_seq.call_count}")
        print(f"Max concurrent: {embeddings_seq.concurrent_calls['max']}")

        # Parallel test
        print("\n--- Parallel Processing ---")
        embeddings_par = SlowEmbeddings(size=128, latency_ms=latency_ms)
        faiss_store_par = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=embeddings_par
        )
        vector_store_par = BaseVectorStore(faiss_store_par)

        start_par = time.time()
        ids_par = await vector_store_par.add_documents_parallel(
            performance_documents, batch_size=batch_size, max_concurrent=max_concurrent
        )
        duration_par = time.time() - start_par

        print(f"Time: {duration_par:.2f}s")
        print(f"API calls: {embeddings_par.call_count}")
        print(f"Max concurrent: {embeddings_par.concurrent_calls['max']}")

        # Calculate improvement
        speedup = duration_seq / duration_par
        time_saved = duration_seq - duration_par
        improvement_pct = ((duration_seq - duration_par) / duration_seq) * 100

        print("\n" + "=" * 70)
        print("PERFORMANCE IMPROVEMENT SUMMARY")
        print("=" * 70)
        print(f"Sequential time:     {duration_seq:.2f}s")
        print(f"Parallel time:       {duration_par:.2f}s")
        print(f"Time saved:          {time_saved:.2f}s")
        print(f"Speedup factor:      {speedup:.2f}x")
        print(f"Improvement:         {improvement_pct:.1f}%")
        print(f"\n✓ Parallel processing is {speedup:.2f}x faster!")
        print("=" * 70)

        # Assertions
        assert len(ids_seq) == len(ids_par) == len(performance_documents)
        assert duration_par < duration_seq  # Parallel should be faster
        assert speedup > 1.5  # Should see at least 1.5x improvement

    @pytest.mark.asyncio
    async def test_varying_concurrency_levels(self, performance_documents):
        """Test performance with different concurrency levels."""
        from langchain_community.vectorstores import FAISS

        print("\n" + "=" * 70)
        print("CONCURRENCY LEVEL COMPARISON")
        print("=" * 70)

        results = []
        concurrency_levels = [1, 2, 3, 5, 10]

        for max_concurrent in concurrency_levels:
            embeddings = SlowEmbeddings(size=128, latency_ms=50)
            faiss_store = FAISS.from_documents(
                documents=[Document(page_content="init")], embedding=embeddings
            )
            vector_store = BaseVectorStore(faiss_store)

            start_time = time.time()
            ids = await vector_store.add_documents_parallel(
                performance_documents, batch_size=10, max_concurrent=max_concurrent
            )
            duration = time.time() - start_time

            results.append(
                {
                    "max_concurrent": max_concurrent,
                    "duration": duration,
                    "api_calls": embeddings.call_count,
                    "max_observed_concurrent": embeddings.concurrent_calls["max"],
                }
            )

            assert len(ids) == len(performance_documents)

        # Print results table
        print(
            f"\n{'Concurrency':<15} {'Time (s)':<12} {'Speedup':<12} {'Max Concurrent'}"
        )
        print("-" * 60)

        baseline = results[0]["duration"]
        for r in results:
            speedup = baseline / r["duration"]
            print(
                f"{r['max_concurrent']:<15} {r['duration']:<12.2f} {speedup:<12.2f}x {r['max_observed_concurrent']}"
            )

        print("\n" + "=" * 70)
        print("Key Findings:")
        print(
            f"  • Higher concurrency = faster processing (up to {baseline/results[-1]['duration']:.1f}x speedup)"
        )
        print("  • Diminishing returns at very high concurrency (thread pool overhead)")
        print("  • Optimal concurrency depends on API rate limits and latency")
        print("=" * 70)


class TestMetricsIntegration:
    """Test metrics collection during parallel processing."""

    @pytest.mark.asyncio
    async def test_metrics_collection(self, performance_documents):
        """Demonstrate metrics collection during parallel processing."""
        from langchain_community.vectorstores import FAISS

        print("\n" + "=" * 70)
        print("PERFORMANCE TRACKING DURING PARALLEL PROCESSING")
        print("=" * 70)

        # Initialize metrics manager
        metrics = MetricsManager()

        # Start a session to track metrics
        session_info = metrics.start_session(input_count=1)
        print(f"Started metrics session: {session_info['session_id']}")

        # Create vector store
        embeddings = SlowEmbeddings(size=128, latency_ms=30)
        faiss_store = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=embeddings
        )
        vector_store = BaseVectorStore(faiss_store)

        # Track operation with custom timing
        start_time = time.time()
        ids = await vector_store.add_documents_parallel(
            performance_documents[:30], batch_size=10, max_concurrent=3
        )
        duration = time.time() - start_time

        # Calculate metrics manually
        doc_count = len(performance_documents[:30])
        throughput = doc_count / duration if duration > 0 else 0

        print(f"\nOperation Metrics:")
        print(f"  Documents embedded:     {doc_count}")
        print(f"  Duration:               {duration:.2f}s")
        print(f"  Throughput:             {throughput:.1f} docs/s")
        print(f"  Batch size:             10")
        print(f"  Max concurrent:         3")

        # End the session
        session_result = metrics.end_session(document_count=doc_count)
        print(f"\nSession completed:")
        print(f"  Success count:          {session_result.get('success_count', 0)}")
        print(f"  Document count:         {session_result.get('document_count', 0)}")

        print("\n✓ Performance tracking completed successfully")
        print("=" * 70)

        assert len(ids) == 30
        assert session_result["document_count"] == doc_count


class TestRealWorldScenarios:
    """Test scenarios that mimic real-world usage patterns."""

    @pytest.mark.asyncio
    async def test_document_ingestion_pipeline(self):
        """Simulate a complete document ingestion pipeline."""
        from langchain_community.vectorstores import FAISS

        print("\n" + "=" * 70)
        print("SIMULATED DOCUMENT INGESTION PIPELINE")
        print("=" * 70)

        # Simulate loading documents
        print("\n1. Loading documents...")
        documents = [
            Document(
                page_content=f"Research paper section {i}. "
                f"This section discusses important findings. "
                f"The methodology employed rigorous testing protocols. "
                f"Results show significant improvements in performance metrics.",
                metadata={
                    "source": f"paper_{i // 10}.pdf",
                    "section": i % 10,
                    "page": i // 5,
                },
            )
            for i in range(100)
        ]
        print(f"   Loaded {len(documents)} document chunks")

        # Simulate chunking (already done above)
        print(f"\n2. Chunking complete: {len(documents)} chunks")

        # Simulate embedding with parallel processing
        print(
            f"\n3. Generating embeddings with parallel processing (max_concurrent=5)..."
        )
        embeddings = SlowEmbeddings(size=384, latency_ms=40)  # Simulate API latency
        faiss_store = FAISS.from_documents(
            documents=[Document(page_content="init")], embedding=embeddings
        )
        vector_store = BaseVectorStore(faiss_store)

        start_embed = time.time()
        vector_ids = await vector_store.add_documents_parallel(
            documents, batch_size=10, max_concurrent=5
        )
        embed_duration = time.time() - start_embed

        print(f"   Embeddings generated in {embed_duration:.2f}s")
        print(f"   Throughput: {len(documents) / embed_duration:.1f} documents/second")
        print(f"   Embedding API calls: {embeddings.call_count}")

        # Simulate retrieval
        print(f"\n4. Testing retrieval...")
        results = vector_store.similarity_search("research findings methodology", k=5)
        print(f"   Retrieved {len(results)} relevant documents")

        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Total documents:        {len(documents)}")
        print(f"Vector IDs generated:   {len(vector_ids)}")
        print(f"Embedding time:         {embed_duration:.2f}s")
        print(f"Average time/doc:       {embed_duration / len(documents) * 1000:.1f}ms")
        print(f"Throughput:             {len(documents) / embed_duration:.1f} docs/s")
        print("\n✓ Pipeline completed successfully with parallel processing!")
        print("=" * 70)

        assert len(vector_ids) == len(documents)
        assert all(vid != "" for vid in vector_ids)
        assert len(results) == 5


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAGDOLL PARALLEL PROCESSING PERFORMANCE TESTS")
    print("=" * 70)
    print("\nRun with: pytest tests/test_parallel_performance.py -v -s\n")
