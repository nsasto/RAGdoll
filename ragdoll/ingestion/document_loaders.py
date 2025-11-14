import concurrent.futures
import logging
import inspect
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from retry import retry

from ragdoll import settings
from ragdoll.app_config import AppConfig, bootstrap_app
from ragdoll.config import Config
from ragdoll.ingestion.base import BaseIngestionService
from ragdoll.cache.cache_manager import CacheManager
from ragdoll.metrics.metrics_manager import MetricsManager


@dataclass
class Source:
    identifier: str
    extension: Optional[str] = None
    is_file: bool = False


class DocumentLoaderService(BaseIngestionService):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        config_path: Optional[str] = None,
        custom_loaders: Optional[Dict[str, Any]] = None,
        max_threads: Optional[int] = None,
        batch_size: Optional[int] = None,
        cache_manager: Optional[CacheManager] = None,
        metrics_manager: Optional[MetricsManager] = None,
        use_cache: bool = True,
        collect_metrics: Optional[bool] = None,
        config_manager: Optional[Config] = None,
        app_config: Optional[AppConfig] = None,
    ):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        if sum(1 for item in (app_config, config_manager, config_path) if item) > 1:
            raise ValueError(
                "Provide only one of app_config, config_manager, or config_path."
            )

        self.app_config = app_config
        if self.app_config is not None:
            self.config_manager = self.app_config.config
        elif config_manager is not None:
            self.config_manager = config_manager
        elif config_path is not None:
            self.app_config = bootstrap_app(config_path)
            self.config_manager = self.app_config.config
        else:
            self.app_config = settings.get_app()
            self.config_manager = self.app_config.config

        config = self.config_manager.ingestion_config
        monitor_config = self.config_manager.monitor_config

        self.max_threads = (
            max_threads if max_threads is not None else config.max_threads
        )
        self.batch_size = batch_size if batch_size is not None else config.batch_size

        self.use_cache = use_cache
        if cache_manager is not None:
            self.cache_manager = cache_manager
        elif self.app_config is not None:
            self.cache_manager = self.app_config.get_cache_manager()
        else:
            self.cache_manager = CacheManager(
                ttl_seconds=self.config_manager.cache_config.cache_ttl
            )

        self.collect_metrics = (
            collect_metrics if collect_metrics is not None else monitor_config.enabled
        )
        if metrics_manager is not None:
            self.metrics_manager = metrics_manager
        elif self.app_config is not None:
            self.metrics_manager = self.app_config.get_metrics_manager()
        else:
            self.metrics_manager = MetricsManager()

        self.loaders = self.config_manager.get_loader_mapping()
        self.logger.debug(f"Available loaders: {list(self.loaders.keys())}")

        if custom_loaders:
            for ext, loader_class in custom_loaders.items():
                if hasattr(loader_class, "load"):
                    self.loaders[ext] = loader_class
                else:
                    self.logger.warning(f"Invalid custom loader for {ext}")

        # Note: you can also register loader classes globally using the
        # `ragdoll.ingestion.register_loader("name")` decorator. If a short
        # name is used in the configuration (instead of a full module path),
        # the ConfigManager will prefer the registered loader class.

        self.logger.info(
            f"Service initialized: loaders={len(self.loaders)}, max_threads={self.max_threads}"
        )

    def _is_arxiv_url(self, url: str) -> bool:
        return "arxiv.org" in url

    def _parse_url_sources(self, url: str) -> Source:
        if self._is_arxiv_url(url):
            return Source(identifier=url, extension="arxiv", is_file=False)

        extension = "website" if not Path(url).suffix else Path(url).suffix.lower()
        return Source(identifier=url, extension=extension, is_file=False)

    def _parse_file_sources(self, pattern: str) -> List[Source]:
        sources: List[Source] = []
        has_wildcard = any(token in pattern for token in ("*", "?", "["))
        matched_paths = (
            [Path(p) for p in glob(pattern, recursive=True)]
            if has_wildcard
            else [Path(pattern)]
        )

        for path in matched_paths:
            if path.exists() and path.is_file():
                sources.append(
                    Source(
                        identifier=str(path.absolute()),
                        extension=path.suffix.lower(),
                        is_file=True,
                    )
                )
        return sources

    def _build_sources(self, inputs: List[str]) -> List[Source]:
        sources = []
        for input_str in inputs:
            if input_str.startswith(("http://", "https://")):
                sources.append(self._parse_url_sources(input_str))
            else:
                sources.extend(self._parse_file_sources(input_str))
        return sources

    @retry(tries=3, delay=1, backoff=2, exceptions=(ConnectionError, TimeoutError))
    def _load_source(
        self, source: Source, batch_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        path = Path(source.identifier)
        source_size_bytes = (
            path.stat().st_size if source.is_file and path.exists() else 0
        )

        metrics_info = None
        if self.collect_metrics and batch_id is not None:
            _source_type = (
                source.extension if source.extension is not None else "unknown"
            )
            metrics_info = self.metrics_manager.start_source(
                batch_id, source.identifier, _source_type
            )

        try:
            if not source.extension:
                if self.use_cache and not source.is_file:
                    cached = self.cache_manager.get_from_cache(
                        "website", source.identifier
                    )
                    if cached:
                        self.logger.info(
                            f"Using cached content for {source.identifier}"
                        )
                        self._record_metrics(
                            metrics_info,
                            batch_id,
                            source,
                            len(cached),
                            source_size_bytes,
                            success=True,
                        )
                        return cached

            if source.extension in self.loaders:
                loader_class = self.loaders[source.extension]

                # Log which loader is being used for which file
                loader_name = getattr(loader_class, "__name__", repr(loader_class))
                loader_module = getattr(
                    loader_class, "__module__", type(loader_class).__module__
                )
                self.logger.info(
                    f"Using {loader_name} from {loader_module} for {source.identifier} (extension: {source.extension})"
                )

                constructor_params = inspect.signature(loader_class.__init__).parameters

                if (
                    "file_path" in constructor_params
                    or "path" in constructor_params
                    or "web_path" in constructor_params
                    or len(constructor_params) == 1
                ):
                    if source.extension == "website":
                        self.logger.info(
                            f"Initializing website loader for {source.identifier}"
                        )
                        loader = loader_class(source.identifier)
                        docs = loader.load()
                    else:
                        if "file_path" in constructor_params:
                            self.logger.info(
                                f"Initializing loader with file_path={source.identifier}"
                            )
                            loader = loader_class(file_path=source.identifier)
                        elif "path" in constructor_params:
                            self.logger.info(
                                f"Initializing loader with path={source.identifier}"
                            )
                            loader = loader_class(path=source.identifier)
                        elif "web_path" in constructor_params:
                            self.logger.info(
                                f"Initializing loader with web_path={source.identifier}"
                            )
                            loader = loader_class(web_path=source.identifier)
                        else:
                            self.logger.info(
                                f"Initializing loader with positional argument: {source.identifier}"
                            )
                            loader = loader_class(source.identifier)
                        docs = loader.load()
                else:
                    self.logger.info(f"Initializing loader with no arguments")
                    loader = loader_class()
                    docs = loader.load()

                self.logger.info(
                    f"Loader {loader_name} returned {len(docs)} documents from {source.identifier}"
                )
                self._record_metrics(
                    metrics_info,
                    batch_id,
                    source,
                    len(docs),
                    source_size_bytes,
                    success=True,
                )
                return docs
            else:
                self.logger.error(
                    f"Unsupported source: No loader found for extension {source.extension}"
                )
                raise ValueError(f"Unsupported source: ext={source.extension}")

        except ValueError:
            # Re-raise ValueError for unsupported sources
            raise
        except Exception as e:
            self.logger.error(f"Failed to load {source.identifier}: {str(e)}")
            self._record_metrics(
                metrics_info, batch_id, source, 0, 0, success=False, error=str(e)
            )
            return []

    def _record_metrics(
        self,
        metrics_info,
        batch_id,
        source,
        doc_count,
        byte_size,
        success=True,
        error=None,
    ):
        if self.collect_metrics and metrics_info:
            self.metrics_manager.end_source(
                batch_id=batch_id,
                source_id=source.identifier,
                success=success,
                document_count=doc_count,
                bytes_count=byte_size,
                error_message=error,
            )

    def ingest_documents(self, inputs: List[str]) -> List[Dict[str, Any]]:
        self.logger.info(f"Starting ingestion of {len(inputs)} inputs")

        if self.collect_metrics:
            self.metrics_manager.start_session(input_count=len(inputs))

        sources = self._build_sources(inputs)
        if not sources:
            if self.collect_metrics:
                self.metrics_manager.end_session(document_count=0)
            raise ValueError("No valid sources found")

        documents = []
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i : i + self.batch_size]
            batch_id = i // self.batch_size + 1

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_threads
            ) as executor:
                results = list(
                    executor.map(
                        lambda s: self._load_source(s, batch_id=batch_id), batch
                    )
                )
                for docs in results:
                    documents.extend(docs)

        if self.collect_metrics:
            self.metrics_manager.end_session(document_count=len(documents))

        self.logger.info(f"Finished ingestion: {len(documents)} documents")
        return documents

    def clear_cache(
        self, source_type: Optional[str] = None, identifier: Optional[str] = None
    ) -> int:
        if not self.use_cache:
            return 0
        return self.cache_manager.clear_cache(source_type, identifier)

    def get_metrics(self, days: int = 30) -> Dict[str, Any]:
        if not self.collect_metrics:
            return {"enabled": False, "message": "Metrics collection is disabled"}
        return {
            "enabled": True,
            "recent_sessions": self.metrics_manager.get_recent_sessions(limit=5),
            "aggregate": self.metrics_manager.get_aggregate_metrics(days=days),
        }



