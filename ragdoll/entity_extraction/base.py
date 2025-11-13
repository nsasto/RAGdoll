from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from langchain_core.documents import Document
from .models import Graph


class BaseEntityExtractor(ABC):
    """
    Base class for entity extraction implementations.

    This serves as the abstract base for all entity extractors in RAGdoll,
    similar to how LangChain defines base classes like BaseEmbeddings or how
    our LLM stack standardizes on the BaseLLMCaller protocol.
    """

    @abstractmethod
    async def extract(self, documents: List[Document], **kwargs) -> Graph:
        """
        Extract entities and relationships from a set of documents.

        Args:
            documents: List of Document objects
            **kwargs: Additional implementation-specific parameters

        Returns:
            Graph object with nodes and edges
        """
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about this entity extractor."""
        return {"name": self.__class__.__name__}
