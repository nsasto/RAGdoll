import concurrent.futures
import os
import logging
import inspect
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from retry import retry

from ragdoll import settings
from ragdoll.config import Config
from ragdoll.ingestion.base import BaseIngestionService
from ragdoll.cache.cache_manager import CacheManager
from ragdoll.metrics.metrics_manager import MetricsManager


@dataclass
class Source:
    identifier: str
    extension: Optional[str] = None
    is_file: bool = False


class IngestionService(BaseIngestionService):
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
        collect_metrics: bool = False,
        config_manager: Optional[Config] = None,
    ):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        if config_manager is not None and config_path is not None:
            raise ValueError("Provide either config_manager or config_path, not both.")

        if config_manager is not None:
            self.config_manager = config_manager
        elif config_path is not None:
            self.config_manager = Config(config_path)
        else:
            self.config_manager = settings.get_config_manager()
        config = self.config_manager.ingestion_config
        monitor_config = self.config_manager.monitor_config

        self.max_threads = (
            max_threads if max_threads is not None else config.max_threads
        )
        self.collect_metrics = (
            collect_metrics if collect_metrics is not None else monitor_config.enabled
        )
        self.batch_size = batch_size if batch_size is not None else config.batch_size

        self.use_cache = use_cache
        ttl = self.config_manager.cache_config.cache_ttl
        self.cache_manager = cache_manager or CacheManager(ttl_seconds=ttl)
        self.metrics_manager = metrics_manager
        self.collect_metrics = metrics_manager is not None

        self.loaders = self.config_manager.get_loader_mapping()
        self.logger.debug(f"Available loaders: {list(self.loaders.keys())}")

        if custom_loaders:
            for ext, loader_class in custom_loaders.items():
                if hasattr(loader_class, "load"):
                    self.loaders[ext] = loader_class
                else:
                    self.logger.warning(f"Invalid custom loader for {ext}")

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
            metrics_info = self.metrics_manager.start_source(
                batch_id, source.identifier
            )

        try:
            if not source.extension:
                self.logger.warning(
                    "Skipping source %s without extension", source.identifier
                )
                self._record_metrics(
                    metrics_info,
                    batch_id,
                    source,
                    0,
                    source_size_bytes,
                    success=False,
                    error="Missing extension",
                )
                return []

            loader_class = self.loaders.get(source.extension)
            if loader_class is None:
                raise ValueError(f"Unsupported source: ext={source.extension}")

            docs = self._load_with_class(loader_class, source)
            normalized_docs = self._normalize_documents(docs)

            self._record_metrics(
                metrics_info,
                batch_id,
                source,
                len(normalized_docs),
                source_size_bytes,
                success=True,
            )
            return normalized_docs

        except Exception as e:
            error_message = str(e)
            self._record_metrics(
                metrics_info,
                batch_id,
                source,
                0,
                source_size_bytes,
                success=False,
                error=error_message,
            )
            raise ValueError(f"Failed to load {source.identifier}: {error_message}")

    def _load_with_class(self, loader_class, source: Source):
        loader = self._instantiate_loader(loader_class, source)
        load_callable = getattr(loader, "load", None)

        if callable(load_callable):
            return load_callable()

        retrieve_callable = getattr(loader, "get_relevant_documents", None)
        if callable(retrieve_callable):
            return retrieve_callable(source.identifier)

        raise ValueError(f"Loader {loader_class} does not provide a load method")

    def _instantiate_loader(self, loader_class, source: Source):
        try:
            constructor_params = inspect.signature(loader_class.__init__).parameters
        except (TypeError, ValueError):
            constructor_params = {}

        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in constructor_params.values()
        )

        if source.is_file:
            if "file_path" in constructor_params or accepts_kwargs:
                return loader_class(file_path=source.identifier)
            if "path" in constructor_params:
                return loader_class(path=source.identifier)

        if source.extension == "website":
            if "web_path" in constructor_params or accepts_kwargs:
                return loader_class(web_path=source.identifier)
            if "url" in constructor_params:
                return loader_class(url=source.identifier)
            return loader_class(source.identifier)

        if len(constructor_params) <= 1:
            return loader_class()

        return loader_class(source.identifier)

    def _normalize_documents(self, docs: Any) -> List[Dict[str, Any]]:
        if not docs:
            return []

        normalized: List[Dict[str, Any]] = []
        for doc in docs:
            if isinstance(doc, dict):
                page_content = doc.get("page_content", "")
                metadata = doc.get("metadata", {}) or {}
            elif hasattr(doc, "page_content"):
                page_content = getattr(doc, "page_content", "")
                metadata = getattr(doc, "metadata", {}) or {}
            else:
                page_content = str(doc)
                metadata = {}

            if not isinstance(metadata, dict):
                try:
                    metadata = dict(metadata)
                except (TypeError, ValueError):
                    metadata = {"value": metadata}

            normalized.append({"page_content": page_content, "metadata": metadata})

        return normalized

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
