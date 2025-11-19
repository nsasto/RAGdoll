"""
Base Retriever Interface

Provides LangChain-compatible base class for all RAGdoll retrievers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.

    Follows LangChain's retriever interface to ensure compatibility
    with LangChain pipelines and tooling.
    """

    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: The query string to retrieve documents for
            **kwargs: Additional retrieval parameters

        Returns:
            List of relevant Document objects
        """
        pass

    @abstractmethod
    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Args:
            query: The query string to retrieve documents for
            **kwargs: Additional retrieval parameters

        Returns:
            List of relevant Document objects
        """
        pass

    def _format_documents(
        self, documents: List[Document], max_length: Optional[int] = None
    ) -> str:
        """
        Format documents into a context string.

        Args:
            documents: List of documents to format
            max_length: Optional maximum length for context

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            content = doc.page_content
            if max_length and total_length + len(content) > max_length:
                # Truncate if we exceed max_length
                remaining = max_length - total_length
                if remaining > 0:
                    content = content[:remaining] + "..."
                    context_parts.append(f"[{i+1}] {content}")
                break

            context_parts.append(f"[{i+1}] {content}")
            total_length += len(content)

        return "\n\n".join(context_parts)

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on content.

        Args:
            documents: List of documents to deduplicate

        Returns:
            Deduplicated list of documents
        """
        seen = set()
        unique_docs = []

        for doc in documents:
            # Use content hash as deduplication key
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        return unique_docs
