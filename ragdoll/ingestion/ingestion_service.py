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
from ragdoll.cache.cache_manager import CacheManager
from ragdoll.metrics.metrics_manager import MetricsManager

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
        cache_dir: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        use_cache: bool = True,
        metrics_dir: Optional[str] = None,
        collect_metrics: bool = True,
    ):
        """
        Initialize the ingestion service.

        Args:
            config_path: Optional path to configuration file.
            custom_loaders: Optional dictionary of custom file extension to loader mappings.
            max_threads: Maximum number of threads for concurrent processing. Overrides config if provided.
            batch_size: Number of sources to process in a single batch. Overrides config if provided.
            cache_dir: Directory for caching documents.
            cache_ttl: Time-to-live for cache entries in seconds.
            use_cache: Whether to use caching for network sources.
            metrics_dir: Directory for storing metrics data.
            collect_metrics: Whether to collect metrics during ingestion.
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
        
        # Initialize cache if enabled
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_manager = CacheManager(
                cache_dir=cache_dir,
                ttl_seconds=cache_ttl or 86400  # Default 24 hours
            )
        else:
            self.cache_manager = None
        
        # Initialize metrics collection
        self.collect_metrics = collect_metrics
        if self.collect_metrics:
            self.metrics_manager = MetricsManager(metrics_dir=metrics_dir)
        else:
            self.metrics_manager = None

        self.logger.info(f"Initialized with {len(self.loaders)} loaders, max_threads={self.max_threads}, use_cache={use_cache}, collect_metrics={collect_metrics}")

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
    def _load_source(self, source: Source, batch_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load documents from a single source with retry logic for network sources.

        Args:
            source: Source object to load.
            batch_id: Optional batch ID for metrics tracking.

        Returns:
            List of loaded document dictionaries.
        """
        self.logger.debug(f"Loading source: {source}")
        
        # Start metrics tracking if enabled
        metrics_info = None
        source_size_bytes = 0
        if self.collect_metrics and batch_id is not None and self.metrics_manager is not None:
            metrics_info = self.metrics_manager.start_source(
                batch_id=batch_id,
                source_id=source.identifier,
                source_type=source.type.value
            )
        
        # Check cache for network sources
        if self.use_cache and source.type in [SourceType.ARXIV, SourceType.WEBSITE]:
            cached_docs = self.cache_manager.get_from_cache(
                source_type=source.type.value,
                identifier=source.identifier
            )
            if cached_docs:
                self.logger.info(f"Using cached version of {source.type.value}:{source.identifier}")
                
                # Update metrics if tracking
                if self.collect_metrics and metrics_info and self.metrics_manager is not None:
                    self.metrics_manager.end_source(
                        batch_id=batch_id,
                        source_id=source.identifier,
                        success=True,
                        document_count=len(cached_docs),
                        bytes_count=source_size_bytes
                    )
                
                return cached_docs
        
        try:
            docs = []
            
            # Try to get file size for local files
            if source.type == SourceType.FILE and os.path.exists(source.identifier):
                try:
                    source_size_bytes = os.path.getsize(source.identifier)
                except (OSError, IOError):
                    source_size_bytes = 0
            
            # Load documents based on source type
            if source.type == SourceType.ARXIV:
                loader = ArxivRetriever()
                docs = loader.get_relevant_documents(query=source.identifier)
                
                # Cache the results
                if self.use_cache:
                    self.cache_manager.save_to_cache(
                        source_type=source.type.value,
                        identifier=source.identifier,
                        documents=docs
                    )
                
            elif source.type == SourceType.WEBSITE:
                loader = WebLoader()
                docs = loader.load(source.identifier)
                
                # Estimate size based on content length
                source_size_bytes = sum(len(doc.get("page_content", "")) for doc in docs)
                
                # Cache the results
                if self.use_cache:
                    self.cache_manager.save_to_cache(
                        source_type=source.type.value,
                        identifier=source.identifier,
                        documents=docs
                    )
                
            elif source.extension and source.extension in self.loaders:
                loader_class = self.loaders[source.extension]
                loader = loader_class(file_path=source.identifier)
                docs = loader.load()
            else:
                raise ValueError(f"Unsupported source: type={source.type}, ext={source.extension}")
            
            # Update metrics if tracking
            if self.collect_metrics and metrics_info and self.metrics_manager is not None:
                self.metrics_manager.end_source(
                    batch_id=batch_id,
                    source_id=source.identifier,
                    success=True,
                    document_count=len(docs),
                    bytes_count=source_size_bytes
                )
            
            return docs
        except Exception as e:
            self.logger.error(f"Failed to load {source}: {str(e)}", exc_info=True)
            
            # Update metrics if tracking
            if self.collect_metrics and metrics_info and self.metrics_manager is not None:
                self.metrics_manager.end_source(
                    batch_id=batch_id,
                    source_id=source.identifier,
                    success=False,
                    document_count=0,
                    bytes_count=0,
                    error_message=str(e)
                )
            
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
        
        # Start metrics session if enabled
        if self.collect_metrics and self.metrics_manager is not None:
            self.metrics_manager.start_session()
        
        sources = self._build_sources(inputs)
        if not sources:
            raise ValueError("No valid sources found")

        documents = []
        # Process sources in batches to manage memory
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1} with {len(batch)} sources")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                batch_docs = executor.map(lambda src: self._load_source(src, batch_id=i // self.batch_size + 1), batch)
                for docs in batch_docs:
                    documents.extend(docs)
        
        # End metrics session if enabled
        if self.collect_metrics and self.metrics_manager is not None:
            self.metrics_manager.end_session(document_count=len(documents))
        
        self.logger.info(f"Finished ingestion, loaded {len(documents)} documents")
        return documents

    def clear_cache(self, source_type: Optional[str] = None, identifier: Optional[str] = None) -> int:
        """
        Clear the cache.
        
        Args:
            source_type: Optional source type to clear
            identifier: Optional identifier to clear
            
        Returns:
            Number of cache entries cleared.
        """
        if not self.use_cache:
            return 0
        
        return self.cache_manager.clear_cache(source_type, identifier)

    def get_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get ingestion metrics.
        
        Args:
            days: Number of days to include in metrics report
            
        Returns:
            Dictionary of metrics data
        """
        if not self.collect_metrics or self.metrics_manager is None:
            return {"enabled": False, "message": "Metrics collection is disabled"}
        
        recent = self.metrics_manager.get_recent_sessions(limit=5)
        aggregate = self.metrics_manager.get_aggregate_metrics(days=days)
        
        return {
            "enabled": True,
            "recent_sessions": recent,
            "aggregate": aggregate
        }