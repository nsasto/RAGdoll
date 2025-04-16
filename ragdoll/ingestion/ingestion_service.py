import concurrent.futures
import glob
import os
from typing import List, Dict, Any, Optional
import logging
from langchain_community.document_loaders import (
    CSVLoader,
    ArxivLoader,
    PyMuPDFLoader,
    TextLoader,
    JSONLoader
)
from langchain_markitdown import (
    DocxLoader,
    PptxLoader,   
    ImageLoader,
    EpubLoader,
    XlsxLoader,
    HtmlLoader,
    RssLoader
)
from langchain_community.retrievers import ArxivRetriever
from ragdoll.loaders.web_loader import WebLoader
from ragdoll.ingestion.base_ingestion_service import BaseIngestionService

#default loaders
LOADER_MAPPING = {
    ".json": JSONLoader,
    ".jsonl": JSONLoader,
    ".yaml": JSONLoader,
    ".csv": CSVLoader,
    ".epub": EpubLoader,
    ".xlsx": XlsxLoader,
    #".xls": UnstructuredExcelLoader,
    ".html": HtmlLoader,
    ".bmp": ImageLoader,
    ".jpeg": ImageLoader,
    ".jpg": ImageLoader,
    ".png": ImageLoader,
    ".tiff": ImageLoader,
    ".md": TextLoader, 
    ".pdf": PyMuPDFLoader,
    ".pptx": PptxLoader,
    #".ppt": UnstructuredPowerPointLoader,
    #".rtf": RtfLoader, #todo
    #".tsv": UnstructuredTSVLoader,
    ".docx": DocxLoader,
    #".doc": UnstructuredWordDocumentLoader,
    ".xml": RssLoader,
    ".txt": TextLoader,
}


class IngestionService(BaseIngestionService):
    logger = logging.getLogger(__name__)

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
                        self.logger.info(f"Processing file: {os.path.abspath(file_path)}")
                        file_extension = ""
                        file_extension = os.path.splitext(file_path)[1].lower()
                        if file_extension in LOADER_MAPPING:
                            source_type = file_extension                          
                        else:
                            
                            source_type = "file"


                        if source_type == "file" and file_extension == "":
                            self.logger.warning(f"no extension found for file: {file_path}")

                        extension = file_extension
                        
                        source = {"type": source_type.replace(".",""), "identifier": file_path}                        
                        source["extension"] = extension
                        sources.append(source)

            return sources

        self.logger.info(f"Starting ingestion of {len(inputs)} sources")

        sources = _build_sources(inputs)

        """
        Ingests documents from various sources concurrently.

        Args:
            inputs: A list of strings, which can be file paths, URLs, or glob patterns.

        """

        def _load_source(source: Dict[str, Any]) -> List[Dict[str, Any]]:
            self.logger.info(f"Loading source: {source}")
            source_type = source.get("type")
            identifier = source["identifier"]
            file_extension = source.get("extension", "")
            
            self.logger.info(f"  Source dictionary: {source}")
            self.logger.info(f"  Source type: {source_type}")
            self.logger.info(f"  File extension: {file_extension}")

            if source_type == "arxiv":
                loader = ArxivRetriever()                
                return loader.get_relevant_documents(query=identifier)                
            elif source_type == "website":
                loader = WebLoader()
                return loader.load(identifier)
            elif file_extension in LOADER_MAPPING:
                loader_class = LOADER_MAPPING[file_extension]
                loader = loader_class(file_path=identifier)
                return loader.load()
            
            else:
                 raise ValueError(f"Unsupported source type or file extension: {source_type} {file_extension}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            documents_list = list(executor.map(_load_source, sources))
        
        documents = [item for sublist in documents_list for item in sublist]        
        self.logger.info(f"Finished ingestion")
        return documents

    def __init__(self, custom_loader_overrides: Optional[Dict[str, Any]] = None):
        logging.basicConfig(level=logging.INFO)
        self.custom_loader_overrides = custom_loader_overrides or {}
