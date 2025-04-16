import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

class CacheManager:
    """Manages caching for network-based document sources."""
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, cache_dir: str = None, ttl_seconds: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store the cache. If None, uses ~/.ragdoll/cache/
            ttl_seconds: Time-to-live for cache entries in seconds. Default is 24 hours.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".ragdoll", "cache")
        
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger.info(f"Cache initialized at {self.cache_dir} with TTL={ttl_seconds}s")
    
    def _get_cache_key(self, source_type: str, identifier: str) -> str:
        """Generate a unique cache key for a source."""
        key = f"{source_type}:{identifier}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, source_type: str, identifier: str) -> Path:
        """Get the file path for a cache entry."""
        key = self._get_cache_key(source_type, identifier)
        return self.cache_dir / f"{key}.json"
    
    def get_from_cache(self, source_type: str, identifier: str) -> Optional[List[Dict[str, Any]]]:
        """
        Try to get documents from cache.
        
        Args:
            source_type: Type of the source (arxiv, website, etc.)
            identifier: Unique identifier for the source (URL, ID, etc.)
            
        Returns:
            Cached documents if found and not expired, None otherwise.
        """
        cache_path = self._get_cache_path(source_type, identifier)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            if time.time() - cache_data["timestamp"] > self.ttl_seconds:
                self.logger.debug(f"Cache expired for {source_type}:{identifier}")
                return None
            
            self.logger.info(f"Cache hit for {source_type}:{identifier}")
            return cache_data["documents"]
            
        except Exception as e:
            self.logger.warning(f"Error reading cache for {source_type}:{identifier}: {e}")
            return None
    
    def save_to_cache(self, source_type: str, identifier: str, documents: List[Dict[str, Any]]) -> bool:
        """
        Save documents to cache.
        
        Args:
            source_type: Type of the source
            identifier: Unique identifier for the source
            documents: Documents to cache
            
        Returns:
            True if saved successfully, False otherwise.
        """
        cache_path = self._get_cache_path(source_type, identifier)
        
        try:
            cache_data = {
                "timestamp": time.time(),
                "source_type": source_type,
                "identifier": identifier,
                "documents": documents
            }
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Cached {len(documents)} documents for {source_type}:{identifier}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching {source_type}:{identifier}: {e}", exc_info=True)
            return False
    
    def clear_cache(self, source_type: Optional[str] = None, identifier: Optional[str] = None) -> int:
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
        
        return count