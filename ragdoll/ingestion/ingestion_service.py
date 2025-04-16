import concurrent.futures
import glob
import os
from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from retry import retry

# Keep imports for loaders used directly in the code
from langchain_community.retrievers import ArxivRetriever
from ragdoll.loaders.web_loader import WebLoader
from ragdoll.ingestion.base_ingestion_service import BaseIngestionService
from ragdoll.loaders.base_loader import BaseLoader, ensure_loader_compatibility
from ragdoll.config.config_manager import ConfigManager

class SourceType(Enum):
    """Enum for supported source types."""
    ARXIV = "arxiv"
    WEBSITE = "website"
    FILE = "file"

@dataclass
class Source:
    """Represents a document source."""
    type: SourceType
    identifier: str
    extension: Optional[str] = None

class IngestionService(BaseIngestionService):
    """Service for ingesting documents from various sources concurrently."""
    
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        config_path: Optional[str] = None,
        custom_loaders: Optional[Dict[str, Any]] = None,
        max_threads: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the ingestion service.

        Args:
            config_path: Optional path to configuration file.
            custom_loaders: Optional dictionary of custom file extension to loader mappings.
            max_threads: Maximum number of threads for concurrent processing. Overrides config if provided.
            batch_size: Number of sources to process in a single batch. Overrides config if provided.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        ingestion_config = config_manager.ingestion_config
        
        # Set parameters, prioritizing constructor arguments over config values
        self.max_threads = max_threads if max_threads is not None else ingestion_config.max_threads
        self.batch_size = batch_size if batch_size is not None else ingestion_config.batch_size
        
        # Get loader mappings from config
        self.loaders = config_manager.get_loader_mapping()
        
        # Add custom loaders with validation
        if custom_loaders:
            for ext, loader_class in custom_loaders.items():
                try:
                    # Check if the loader is compatible
                    if not issubclass(loader_class, BaseLoader) and not hasattr(loader_class, 'load'):
                        self.logger.warning(
                            f"Custom loader for extension {ext} doesn't inherit from BaseLoader "
                            f"and doesn't have a load method: {loader_class.__name__}"
                        )
                    self.loaders[ext] = loader_class
                except TypeError as e:
                    self.logger.error(f"Invalid custom loader for extension {ext}: {e}")
        
        self.logger.info(f"Initialized with {len(self.loaders)} loaders, max_threads={self.max_threads}")

    def _build_sources(self, inputs: List[str]) -> List[Source]:
        """
        Build a list of Source objects from input strings.

        Args:
            inputs: List of file paths, URLs, or glob patterns.

        Returns:
            List of Source objects.
        """
        sources = []
        for input_str in inputs:
            try:
                if input_str.startswith(("http://", "https://")):
                    if "arxiv.org" in input_str:
                        sources.append(Source(type=SourceType.ARXIV, identifier=input_str.split("/")[-1]))
                    elif input_str.endswith(".pdf"):
                        sources.append(Source(type=SourceType.FILE, identifier=input_str, extension=".pdf"))
                    else:
                        sources.append(Source(type=SourceType.WEBSITE, identifier=input_str))
                else:
                    # Handle file paths or glob patterns
                    for path in Path().glob(input_str) if "*" in input_str else [Path(input_str)]:
                        if not path.is_file():
                            self.logger.warning(f"Skipping non-file: {path}")
                            continue
                        ext = path.suffix.lower()
                        source_type = SourceType.FILE if ext in self.loaders else SourceType.FILE
                        sources.append(Source(
                            type=source_type,
                            identifier=str(path.absolute()),
                            extension=ext or None,
                        ))
                        self.logger.debug(f"Added file source: {path} (ext={ext})")
            except Exception as e:
                self.logger.error(f"Error processing input {input_str}: {str(e)}", exc_info=True)
        return sources

    @retry(tries=3, delay=1, backoff=2, exceptions=(ConnectionError, TimeoutError))
    def _load_source(self, source: Source) -> List[Dict[str, Any]]:
        """
        Load documents from a single source with retry logic for network sources.

        Args:
            source: Source object to load.

        Returns:
            List of loaded document dictionaries.

        Raises:
            ValueError: If the source type or extension is unsupported.
            Exception: For other loading errors, logged with details.
        """
        self.logger.debug(f"Loading source: {source}")
        try:
            if source.type == SourceType.ARXIV:
                loader = ArxivRetriever()
                return loader.get_relevant_documents(query=source.identifier)
            elif source.type == SourceType.WEBSITE:
                loader = WebLoader()
                return loader.load(source.identifier)
            elif source.extension and source.extension in self.loaders:
                loader_class = self.loaders[source.extension]
                loader = loader_class(file_path=source.identifier)
                return loader.load()
            else:
                raise ValueError(f"Unsupported source: type={source.type}, ext={source.extension}")
        except Exception as e:
            self.logger.error(f"Failed to load {source}: {str(e)}", exc_info=True)
            return []

    def ingest_documents(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest documents from various sources concurrently.

        Args:
            inputs: List of file paths, URLs, or glob patterns.

        Returns:
            List of document dictionaries.

        Raises:
            ValueError: If no valid sources are provided.
        """
        self.logger.info(f"Starting ingestion of {len(inputs)} inputs")
        sources = self._build_sources(inputs)
        if not sources:
            raise ValueError("No valid sources found")

        documents = []
        # Process sources in batches to manage memory
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1} with {len(batch)} sources")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                batch_docs = executor.map(self._load_source, batch)
                for docs in batch_docs:
                    documents.extend(docs)

        self.logger.info(f"Finished ingestion, loaded {len(documents)} documents")
        return documents