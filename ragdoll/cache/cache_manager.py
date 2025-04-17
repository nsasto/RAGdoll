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
    
    def get_from_cache(self, source_type: str, identifier: str) -> Optional[List]:
        """
        Retrieve documents from cache if available and not expired.
        
        Args:
            source_type: Source type
            identifier: Source identifier
            
        Returns:
            List of cached documents or None if not found/expired
        """
        try:
            cache_key = self._get_cache_key(source_type, identifier)
            cache_path = self.cache_dir / f"{cache_key}.json"
            
            if not cache_path.exists():
                return None
            
            # Check if cache is expired
            if self.ttl_seconds > 0:
                mod_time = os.path.getmtime(cache_path)
                age_seconds = time.time() - mod_time
                if age_seconds > self.ttl_seconds:
                    self.logger.debug(f"Cache expired for {source_type}:{identifier}")
                    return None
            
            # Load cache data
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            documents = cache_data.get("documents", [])
            self.logger.debug(f"Retrieved {len(documents)} documents from cache for {source_type}:{identifier}")
            
            return documents
        
        except Exception as e:
            self.logger.error(f"Error retrieving from cache {source_type}:{identifier}: {str(e)}")
            return None
    
    def save_to_cache(self, source_type: str, identifier: str, documents: List) -> None:
        """
        Save documents to cache.
        
        Args:
            source_type: Source type
            identifier: Source identifier
            documents: List of documents to cache
        """
        try:
            # Convert documents to a serializable format if needed
            serializable_docs = []
            for doc in documents:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    # Convert Document object to dict
                    serializable_docs.append({
                        'page_content': doc.page_content,
                        'metadata': dict(doc.metadata)  # Convert metadata to a regular dict
                    })
                elif isinstance(doc, dict) and 'page_content' in doc:
                    # Already a dictionary
                    serializable_docs.append(doc)
                else:
                    # Unknown format - create minimal representation
                    self.logger.warning(f"Unknown document format for cache: {type(doc)}")
                    serializable_docs.append({
                        'page_content': str(doc),
                        'metadata': {'source_type': source_type, 'source': identifier}
                    })
            
            # Store cache data
            cache_data = {
                "source_type": source_type,
                "identifier": identifier,
                "timestamp": self._get_iso_timestamp(),
                "documents": serializable_docs
            }
            
            # Create cache key
            cache_key = self._get_cache_key(source_type, identifier)
            cache_path = self.cache_dir / f"{cache_key}.json"
            
            # Write to cache file
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"Cached {len(documents)} documents for {source_type}:{identifier}")
            
        except Exception as e:
            self.logger.error(f"Error caching {source_type}:{identifier}: {str(e)}")
    
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