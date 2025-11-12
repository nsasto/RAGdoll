import abc
from typing import List, Dict, Any


class BaseIngestionService(abc.ABC):
    @abc.abstractmethod
    def ingest_documents(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ingests documents from various sources concurrently.

        Args:
            sources: A list of dictionaries, where each dictionary represents a source
                     and contains keys like "type" (e.g., "website", "pdf") and "identifier"
                     (e.g., a URL, a file path).

        Returns:
            A list of documents with metadata.
        """
        pass
