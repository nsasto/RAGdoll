import concurrent.futures
import glob
from typing import List, Dict, Any, Optional

from ragdoll.loaders.web_loader import WebLoader
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader

from ragdoll.ingestion.base_ingestion_service import BaseIngestionService

LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": WebLoader,
}

class IngestionService(BaseIngestionService):
    def ingest_documents(self, inputs: List[str]) -> List[Dict[str, Any]]:
        def _build_sources(inputs: List[str]) -> List[Dict[str, Any]]:
            sources = []
            for input_str in inputs:
                if input_str.startswith("http://") or input_str.startswith("https://"):                    
                    if "arxiv.org" in input_str:
                        source_type = "arxiv"
                        identifier = input_str.split("/")[-1]                        
                    elif input_str.endswith(".pdf"):
                        source_type = "pdf"
                    else:                    
                        source_type = "website"                    
                    sources.append({"type": source_type, "identifier": input_str})
                

                else:
                    # Treat as file path or pattern
                    expanded_files = glob.glob(input_str)
                    for file_path in expanded_files:
                        parts = file_path.split(".")
                        if len(parts) > 1:
                            extension = "." + parts[-1]
                            if extension in LOADER_MAPPING:
                                source_type = extension
                            else:
                                source_type = "file"
                        else:
                            source_type = "file"
                            extension = ""
                        
                        source = {"type": source_type, "identifier": file_path}                        
                        if source_type == "file" and extension:
                            source["extension"] = extension
                        sources.append(source)
            return sources

        sources = _build_sources(inputs)

        """
        Ingests documents from various sources concurrently.

        Args:
            inputs: A list of strings, which can be file paths, URLs, or glob patterns.

        """

        def _load_source(source: Dict[str, Any]) -> List[Dict[str, Any]]:
            source_type = source.get("type")
            identifier = source["identifier"]
            file_extension = source.get("extension", "")

            if source_type == "arxiv":
                from langchain_community.retrievers import ArxivRetriever
                return ArxivRetriever().get_relevant_documents(query=identifier)
            if source_type == "website":                
                loader_class = WebLoader
                loader = loader_class()
                return loader.load(identifier)          
            elif file_extension and file_extension in self.custom_loader_overrides:                
                loader_class = self.custom_loader_overrides[file_extension]            
            elif file_extension and file_extension in LOADER_MAPPING:                
                loader_class = LOADER_MAPPING[file_extension]            
            elif source_type and source_type in LOADER_MAPPING:                
                loader_class = LOADER_MAPPING[source_type]            
            else:
                loader_class = UnstructuredLoader

            loader = loader_class(identifier)
            return loader.load()

        with concurrent.futures.ThreadPoolExecutor() as executor:            
            documents_list = list(executor.map(_load_source, sources))
        
        documents = [item for sublist in documents_list for item in sublist]
        return documents

    def __init__(self, custom_loader_overrides: Optional[Dict[str, Any]] = None):
        self.custom_loader_overrides = custom_loader_overrides or {}