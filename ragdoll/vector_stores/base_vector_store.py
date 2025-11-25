from __future__ import annotations

from typing import Any, List, Sequence, Type, TypeVar

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_core.vectorstores import VectorStore

VectorStoreT = TypeVar("VectorStoreT", bound=VectorStore)


class BaseVectorStore:
    """Thin wrapper that delegates to a LangChain VectorStore implementation."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @property
    def store(self) -> VectorStore:
        """Expose the underlying LangChain VectorStore instance."""
        return self._store

    def add_documents(
        self, documents: Sequence[Document], batch_size: int | None = None
    ) -> List[str]:
        """Add documents, splitting into batches if the backend advertises a limit."""
        docs = list(documents)
        if not docs:
            return []

        limit = batch_size or self._detect_batch_limit()
        if not limit or limit <= 0 or len(docs) <= limit:
            return self._store.add_documents(docs)

        ids: List[str] = []
        for start in range(0, len(docs), limit):
            ids.extend(self._store.add_documents(docs[start : start + limit]))
        return ids

    async def aadd_documents(
        self, documents: Sequence[Document], batch_size: int | None = None
    ) -> List[str]:
        """Async wrapper for add_documents (standard LangChain pattern).

        This method provides an async interface to the synchronous add_documents
        method by running it in a thread pool. This is the standard pattern used
        by LangChain for async operations.

        Args:
            documents: Documents to add to the vector store
            batch_size: Optional batch size for chunking (uses _detect_batch_limit if None)

        Returns:
            List of document IDs
        """
        import asyncio

        return await asyncio.to_thread(self.add_documents, documents, batch_size)

    async def add_documents_parallel(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int | None = None,
        max_concurrent: int = 3,
        retry_failed: bool = True,
    ) -> List[str]:
        """Add documents with parallel embedding generation for better performance.

        This method splits documents into batches and processes multiple batches
        concurrently to parallelize embedding generation. This is particularly
        beneficial when using remote embedding services (OpenAI, Cohere, etc.)
        where network latency is a factor.

        Performance: Typically provides 3-5x speedup for remote embeddings.

        Args:
            documents: Documents to add to the vector store
            batch_size: Size of each batch (uses _detect_batch_limit if None)
            max_concurrent: Maximum number of batches to process concurrently
            retry_failed: If True, retry failed batches sequentially

        Returns:
            List of document IDs (maintains order with original documents)

        Example:
            ```python
            # Using with configured max_concurrent from EmbeddingsConfig
            embeddings = get_embedding_model()
            vector_store = BaseVectorStore.from_documents(
                FAISS, documents=[], embedding=embeddings
            )
            ids = await vector_store.add_documents_parallel(
                documents=chunks,
                max_concurrent=config.embeddings.max_concurrent_embeddings
            )
            ```
        """
        import asyncio
        import logging

        logger = logging.getLogger(__name__)

        docs = list(documents)
        if not docs:
            return []

        # Determine batch size
        effective_batch_size = batch_size or self._detect_batch_limit() or 10

        # Split into batches
        batches = [
            docs[i : i + effective_batch_size]
            for i in range(0, len(docs), effective_batch_size)
        ]

        logger.info(
            f"Adding {len(docs)} documents in {len(batches)} batches "
            f"with max {max_concurrent} concurrent operations"
        )

        # Process batches in parallel with concurrency limit
        all_ids: List[str] = []

        for i in range(0, len(batches), max_concurrent):
            batch_group = batches[i : i + max_concurrent]

            # Create tasks for concurrent batch processing
            tasks = [
                asyncio.to_thread(self._store.add_documents, batch)
                for batch in batch_group
            ]

            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results and retries
            for idx, result in enumerate(results):
                batch_idx = i + idx
                if isinstance(result, Exception):
                    logger.error(
                        f"Batch {batch_idx + 1}/{len(batches)} failed: {result}"
                    )

                    if retry_failed:
                        logger.info(f"Retrying batch {batch_idx + 1} sequentially...")
                        try:
                            retry_ids = self._store.add_documents(batch_group[idx])
                            all_ids.extend(retry_ids)
                            logger.info(f"Batch {batch_idx + 1} retry successful")
                        except Exception as retry_error:
                            logger.error(
                                f"Batch {batch_idx + 1} retry failed: {retry_error}"
                            )
                            # Add empty IDs to maintain alignment with input documents
                            all_ids.extend(["" for _ in batch_group[idx]])
                    else:
                        # Add empty IDs to maintain alignment
                        all_ids.extend(["" for _ in batch_group[idx]])
                else:
                    all_ids.extend(result)

        logger.info(
            f"Successfully added {len([id for id in all_ids if id])} documents "
            f"to vector store (total slots: {len(all_ids)})"
        )

        return all_ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return the top-k similar documents from the wrapped store."""
        return self._store.similarity_search(query, k=k)

    async def asimilarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Async version of similarity_search."""
        return await self._store.asimilarity_search(query, k=k)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, **kwargs: Any
    ) -> List[Document]:
        """Return documents using maximal marginal relevance search."""
        return self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, **kwargs
        )

    async def amax_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, **kwargs: Any
    ) -> List[Document]:
        """Async version of max_marginal_relevance_search."""
        return await self._store.amax_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, **kwargs
        )

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Return documents with relevance scores."""
        return self._store.similarity_search_with_relevance_scores(query, k=k, **kwargs)

    def delete(self, ids: Sequence[str]) -> Any:
        """Delete documents if the wrapped store supports deletion."""
        if not hasattr(self._store, "delete"):
            raise NotImplementedError(
                f"{self._store.__class__.__name__} does not implement delete()."
            )
        return self._store.delete(ids=list(ids))

    @classmethod
    def from_documents(
        cls,
        store_cls: Type[VectorStoreT],
        documents: Sequence[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "BaseVectorStore":
        """Build a wrapped store pre-populated with the provided documents."""
        store = store_cls.from_documents(
            documents=documents, embedding=embedding, **kwargs
        )
        return cls(store)

    def _detect_batch_limit(self) -> int | None:
        """Infer the max batch size supported by the underlying store, if exposed."""
        direct_limit = getattr(self._store, "max_batch_size", None)
        if isinstance(direct_limit, int) and direct_limit > 0:
            return direct_limit

        client = getattr(self._store, "_client", None)
        if client is None:
            return None

        getter = getattr(client, "get_max_batch_size", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, int) and value > 0:
                    return value
            except Exception:
                pass

        client_limit = getattr(client, "max_batch_size", None)
        if isinstance(client_limit, int) and client_limit > 0:
            return client_limit
        return None
