from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search the vector store for similar documents."""
        pass

    @abstractmethod
    def from_documents(self, documents: List[Document], embedding: Embeddings):
        """Create the vector store from a list of documents."""
        pass