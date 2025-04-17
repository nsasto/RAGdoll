import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from ragdoll.cache.cache_manager import CacheManager

logging.getLogger("ragdoll.config.config_manager").setLevel(logging.ERROR)

@pytest.fixture
def sample_fixture():
    return "sample data"

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"page_content": "Test content 1", "metadata": {"source": "test1"}},
        {"page_content": "Test content 2", "metadata": {"source": "test2"}}
    ]

@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a cache manager with a short TTL for testing."""
    # Set smaller memory cache limit for testing
    cache_mgr = CacheManager(cache_dir=temp_cache_dir, ttl_seconds=5)
    cache_mgr.max_memory_cache_items = 5
    cache_mgr.memory_cache = {}  # Ensure cache starts empty
    return cache_mgr