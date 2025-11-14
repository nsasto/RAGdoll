import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging
from ragdoll import settings

if TYPE_CHECKING:
    from ragdoll.app_config import AppConfig


def canonical_cache_key(input_str: Optional[str]) -> str:
    """Return canonical cache key: MD5 hex of the UTF-8 string (32 lowercase hex chars).

    Use this helper anywhere a logical identifier (e.g. "website:https://...")
    needs to be converted into a stable filesystem-friendly key.
    """
    if input_str is None:
        input_str = ""
    if not isinstance(input_str, str):
        input_str = str(input_str)
    return hashlib.md5(input_str.encode("utf-8")).hexdigest()


class CacheManager:
    """Manages caching for network-based document sources."""

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        app_config: Optional["AppConfig"] = None,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store the cache. If None, uses ~/.ragdoll/cache/
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".ragdoll", "cache")

        # Use provided ttl_seconds or fall back to config driven defaults
        if ttl_seconds is None:
            if app_config is not None:
                ttl_seconds = app_config.config.cache_config.cache_ttl
            else:
                cache_config = settings.get_cache_config()
                ttl_seconds = cache_config.cache_ttl

        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Add in-memory cache for frequently accessed documents
        self.memory_cache = {}
        self.max_memory_cache_items = 100  # Limit to avoid memory issues

        self.logger.info(
            f"Cache initialized at {self.cache_dir} with TTL={ttl_seconds}s"
        )

    def _get_cache_key(self, source_type: str, identifier: str) -> str:
        """Generate a unique cache key for a source."""
        # Use the canonical helper so the same logical id maps to the same
        # MD5 hex digest everywhere in the codebase.
        return canonical_cache_key(f"{source_type}:{identifier}")

    def _get_cache_path(self, source_type: str, identifier: str) -> Path:
        """Get the file path for a cache entry."""
        key = self._get_cache_key(source_type, identifier)
        return self.cache_dir / f"{key}.json"

    def _get_iso_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def get_from_cache(self, source_type: str, identifier: str) -> Optional[List]:
        """
        Retrieve documents from cache with optimized performance.
        """
        # Use canonical hashed key for both memory and file lookup
        cache_key = self._get_cache_key(source_type, identifier)

        # Check memory cache first (hashed key)
        if cache_key in self.memory_cache:
            self.logger.debug(f"Retrieved {cache_key} from memory cache")
            return self.memory_cache[cache_key]

        try:
            cache_key = self._get_cache_key(source_type, identifier)
            cache_path = self.cache_dir / f"{cache_key}.json"

            if not cache_path.exists():
                return None

            # Quick check for cache expiry
            if self.ttl_seconds > 0:
                mod_time = os.path.getmtime(cache_path)
                if time.time() - mod_time > self.ttl_seconds:
                    self.logger.debug(f"Cache expired for {source_type}:{identifier}")
                    return None

            # Load and reconstruct document objects
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            documents = cache_data.get("documents", [])

            # Convert dictionaries back to Document objects if needed
            from langchain_core.documents import Document

            result_docs = []
            for doc_dict in documents:
                if isinstance(doc_dict, dict) and "page_content" in doc_dict:
                    result_docs.append(
                        Document(
                            page_content=doc_dict["page_content"],
                            metadata=doc_dict.get("metadata", {}),
                        )
                    )
                else:
                    result_docs.append(doc_dict)

            self.logger.debug(
                f"Retrieved {len(result_docs)} documents from cache for {source_type}:{identifier}"
            )

            # Store in memory cache for next time
            if result_docs and len(self.memory_cache) < self.max_memory_cache_items:
                # store under hashed key for consistent hits
                self.memory_cache[cache_key] = result_docs

            return result_docs

        except Exception as e:
            self.logger.error(
                f"Error retrieving from cache {source_type}:{identifier}: {str(e)}"
            )
            return None

    def save_to_cache(self, source_type: str, identifier: str, documents: List) -> None:
        """
        Save documents to cache with optimized serialization.
        """
        try:
            # Convert documents to a serializable format
            serializable_docs = []
            for doc in documents:
                if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    # Convert Document object to dict without deep copying
                    serializable_docs.append(
                        {
                            "page_content": doc.page_content,
                            "metadata": dict(doc.metadata),
                        }
                    )
                elif isinstance(doc, dict) and "page_content" in doc:
                    # Already serializable
                    serializable_docs.append(doc)
                else:
                    self.logger.warning(
                        f"Unknown document format for cache: {type(doc)}"
                    )
                    serializable_docs.append(
                        {
                            "page_content": str(doc),
                            "metadata": {
                                "source_type": source_type,
                                "source": identifier,
                            },
                        }
                    )

            # Store cache data with minimal indentation
            cache_data = {
                "source_type": source_type,
                "identifier": identifier,
                "timestamp": self._get_iso_timestamp(),
                "documents": serializable_docs,
            }

            # Create cache key (hashed) and file path
            cache_key = self._get_cache_key(source_type, identifier)
            cache_path = self.cache_dir / f"{cache_key}.json"

            # Write to cache file with minimal formatting
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=None)

            self.logger.debug(
                f"Cached {len(documents)} documents for {source_type}:{identifier}"
            )

            # Also populate memory cache using the canonical hashed key
            # Convert to Document objects similarly to get_from_cache's reconstruction
            try:
                from langchain_core.documents import Document

                reconstructed = []
                for d in serializable_docs:
                    if isinstance(d, dict) and "page_content" in d:
                        reconstructed.append(
                            Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                        )
                    else:
                        reconstructed.append(d)

                if reconstructed and len(self.memory_cache) < self.max_memory_cache_items:
                    self.memory_cache[cache_key] = reconstructed
            except Exception:
                # If Document class not available, store serializable docs
                if serializable_docs and len(self.memory_cache) < self.max_memory_cache_items:
                    self.memory_cache[cache_key] = serializable_docs

        except Exception as e:
            self.logger.error(f"Error caching {source_type}:{identifier}: {str(e)}")

    def clear_cache(
        self, source_type: Optional[str] = None, identifier: Optional[str] = None
    ) -> int:
        """
        Clear cache entries.

        Args:
            source_type: If provided, only clear entries of this source type
            identifier: If provided, only clear entries with this identifier

        Returns:
            Number of cache entries cleared.
        """
        if source_type and identifier:
            # Clear specific cache entry
            cache_path = self._get_cache_path(source_type, identifier)
            if cache_path.exists():
                os.remove(cache_path)
                # remove from memory cache as well
                try:
                    cache_key = self._get_cache_key(source_type, identifier)
                    if cache_key in self.memory_cache:
                        del self.memory_cache[cache_key]
                except Exception:
                    pass
                return 1
            return 0

        # Clear all cache entries or by source type
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                if source_type is None or cache_data.get("source_type") == source_type:
                    os.remove(cache_file)
                    count += 1
            except:
                # If we can't read the file, remove it anyway
                os.remove(cache_file)
                count += 1

        # For broad clears (by source_type or full), clear memory cache to avoid stale entries
        try:
            if source_type is None:
                # full clear
                self.memory_cache.clear()
            else:
                # best-effort: clear entire memory cache when clearing by source_type
                self.memory_cache.clear()
        except Exception:
            pass

        return count
