import concurrent.futures
from typing import List, Dict, Any, Optional

from ragdoll.loaders import WebLoader
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, PDFLoader

from ragdoll.ingestion.base_ingestion_service import BaseIngestionService

LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PDFLoader,
    ".html": WebLoader,
}

class IngestionService(BaseIngestionService):
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

        def _load_source(source: Dict[str, Any]) -> List[Dict[str, Any]]:
            source_type = source.get("type")
            identifier = source["identifier"]
            file_extension = source.get("extension", "")

            if source_type == "website":
                loader_class = WebLoader
            elif file_extension and file_extension in self.custom_loader_overrides:
                loader_class = self.custom_loader_overrides[file_extension]
            elif file_extension and file_extension in LOADER_MAPPING:
                loader_class = LOADER_MAPPING[file_extension]
            elif source_type and source_type in LOADER_MAPPING:
                loader_class = LOADER_MAPPING[source_type]
            else:   
                loader_class = UnstructuredFileLoader

            loader = loader_class(identifier)
            return loader.load()

        with concurrent.futures.ThreadPoolExecutor() as executor:            
            documents_list = list(executor.map(_load_source, sources))
        
        documents = [item for sublist in documents_list for item in sublist]

    def __init__(self, custom_loader_overrides: Optional[Dict[str, Any]] = None):
        self.custom_loader_overrides = custom_loader_overrides or {}