import os
import pytest
import tempfile
import json
from pathlib import Path
from ragdoll.cache.cache_manager import CacheManager


class TestCacheManager:
    def test_cache_manager_init(self):
        """Test CacheManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir, ttl_seconds=3600)
            assert cache_manager.cache_dir == Path(temp_dir)
            assert cache_manager.ttl_seconds == 3600
            assert cache_manager.cache_dir.exists()

    def test_get_cache_key(self):
        """Test cache key generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            key1 = cache_manager._get_cache_key("website", "https://example.com")
            key2 = cache_manager._get_cache_key("website", "https://example.com")
            assert key1 == key2
            assert len(key1) == 32  # MD5 hash length

    def test_save_and_get_from_cache(self):
        """Test saving to and retrieving from cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir, ttl_seconds=3600)

            # Test data
            documents = [
                {"page_content": "Test content 1", "metadata": {"source": "test"}},
                {"page_content": "Test content 2", "metadata": {"source": "test"}},
            ]

            # Save to cache
            cache_manager.save_to_cache("website", "https://example.com", documents)

            # Retrieve from cache
            cached_docs = cache_manager.get_from_cache("website", "https://example.com")

            assert cached_docs is not None
            assert len(cached_docs) == 2
            # Cache returns Document objects, so check attributes
            assert cached_docs[0].page_content == "Test content 1"
            assert cached_docs[1].page_content == "Test content 2"

    def test_cache_miss(self):
        """Test cache miss for non-existent entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            result = cache_manager.get_from_cache("website", "https://nonexistent.com")
            assert result is None

    def test_clear_cache_specific(self):
        """Test clearing a specific cache entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Save to cache
            documents = [{"page_content": "Test", "metadata": {}}]
            cache_manager.save_to_cache("website", "https://example.com", documents)

            # Verify it's cached
            assert (
                cache_manager.get_from_cache("website", "https://example.com")
                is not None
            )

            # Clear specific entry
            cleared = cache_manager.clear_cache("website", "https://example.com")
            assert cleared == 1

            # Verify it's gone
            assert (
                cache_manager.get_from_cache("website", "https://example.com") is None
            )

    def test_clear_cache_all(self):
        """Test clearing all cache entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Save multiple entries
            documents = [{"page_content": "Test", "metadata": {}}]
            cache_manager.save_to_cache("website", "https://example1.com", documents)
            cache_manager.save_to_cache("website", "https://example2.com", documents)

            # Clear all
            cleared = cache_manager.clear_cache()
            assert cleared == 2

    def test_memory_cache(self):
        """Test in-memory cache functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            documents = [{"page_content": "Test", "metadata": {}}]
            cache_manager.save_to_cache("website", "https://example.com", documents)

            # First access should load from file and store in memory
            cached1 = cache_manager.get_from_cache("website", "https://example.com")
            assert cached1 is not None

            # Second access should use memory cache
            cached2 = cache_manager.get_from_cache("website", "https://example.com")
            assert cached2 is not None
            assert cached1 == cached2
