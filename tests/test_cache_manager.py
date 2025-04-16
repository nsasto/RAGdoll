import os
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from ragdoll.cache.cache_manager import CacheManager
from ragdoll.ingestion.ingestion_service import IngestionService, Source, SourceType

class TestCacheManager:
    @pytest.fixture
    def cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, cache_dir):
        """Create a cache manager with a short TTL for testing."""
        return CacheManager(cache_dir=cache_dir, ttl_seconds=5)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"page_content": "Test content 1", "metadata": {"source": "test1"}},
            {"page_content": "Test content 2", "metadata": {"source": "test2"}}
        ]
    
    def test_cache_key_generation(self, cache_manager):
        """Test that cache keys are generated consistently."""
        key1 = cache_manager._get_cache_key("arxiv", "1234.5678")
        key2 = cache_manager._get_cache_key("arxiv", "1234.5678")
        assert key1 == key2
        
        # Different source types should have different keys
        key3 = cache_manager._get_cache_key("website", "1234.5678")
        assert key1 != key3
    
    def test_save_and_retrieve_from_cache(self, cache_manager, sample_documents):
        """Test saving and retrieving documents from cache."""
        # Save to cache
        result = cache_manager.save_to_cache("arxiv", "1234.5678", sample_documents)
        assert result is True
        
        # Check if file exists
        cache_path = cache_manager._get_cache_path("arxiv", "1234.5678")
        assert cache_path.exists()
        
        # Retrieve from cache
        cached_docs = cache_manager.get_from_cache("arxiv", "1234.5678")
        assert cached_docs == sample_documents
        
        # Check contents of cache file
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        assert cache_data["source_type"] == "arxiv"
        assert cache_data["identifier"] == "1234.5678"
        assert cache_data["documents"] == sample_documents
        assert "timestamp" in cache_data
    
    def test_cache_expiration(self, cache_manager, sample_documents):
        """Test that expired cache entries are not returned."""
        # Save to cache with short TTL
        cache_manager.save_to_cache("arxiv", "1234.5678", sample_documents)
        
        # Verify it's in the cache
        assert cache_manager.get_from_cache("arxiv", "1234.5678") == sample_documents
        
        # Wait for cache to expire
        time.sleep(6)  # TTL is 5 seconds
        
        # Should now return None
        assert cache_manager.get_from_cache("arxiv", "1234.5678") is None
    
    def test_clear_specific_cache(self, cache_manager, sample_documents):
        """Test clearing specific cache entries."""
        # Save multiple entries
        cache_manager.save_to_cache("arxiv", "1234.5678", sample_documents)
        cache_manager.save_to_cache("website", "https://example.com", sample_documents)
        
        # Clear specific entry
        count = cache_manager.clear_cache("arxiv", "1234.5678")
        assert count == 1
        
        # Verify it's gone
        assert cache_manager.get_from_cache("arxiv", "1234.5678") is None
        
        # Other entry should still be there
        assert cache_manager.get_from_cache("website", "https://example.com") == sample_documents
    
    def test_clear_by_source_type(self, cache_manager, sample_documents):
        """Test clearing all cache entries of a specific type."""
        # Save multiple entries
        cache_manager.save_to_cache("arxiv", "1234.5678", sample_documents)
        cache_manager.save_to_cache("arxiv", "8765.4321", sample_documents)
        cache_manager.save_to_cache("website", "https://example.com", sample_documents)
        
        # Clear all arxiv entries
        count = cache_manager.clear_cache("arxiv")
        assert count == 2
        
        # Verify they're gone
        assert cache_manager.get_from_cache("arxiv", "1234.5678") is None
        assert cache_manager.get_from_cache("arxiv", "8765.4321") is None
        
        # Website entry should still be there
        assert cache_manager.get_from_cache("website", "https://example.com") == sample_documents
    
    def test_clear_all_cache(self, cache_manager, sample_documents):
        """Test clearing all cache entries."""
        # Save multiple entries
        cache_manager.save_to_cache("arxiv", "1234.5678", sample_documents)
        cache_manager.save_to_cache("website", "https://example.com", sample_documents)
        
        # Clear all cache
        count = cache_manager.clear_cache()
        assert count == 2
        
        # Verify all are gone
        assert cache_manager.get_from_cache("arxiv", "1234.5678") is None
        assert cache_manager.get_from_cache("website", "https://example.com") is None


class TestIngestionServiceCache:
    @pytest.fixture
    def cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"page_content": "Test content 1", "metadata": {"source": "test1"}},
            {"page_content": "Test content 2", "metadata": {"source": "test2"}}
        ]
    
    @pytest.fixture
    def mock_config_manager(self, monkeypatch):
        """Mock the config manager."""
        mock_manager = MagicMock()
        mock_manager.get_loader_mapping.return_value = {}
        mock_manager.ingestion_config.max_threads = 2
        mock_manager.ingestion_config.batch_size = 5
        
        mock_config = MagicMock()
        mock_config.return_value = mock_manager
        monkeypatch.setattr("ragdoll.ingestion.ingestion_service.ConfigManager", mock_config)
        
        return mock_config
    
    def test_cache_integration(self, cache_dir, sample_documents, mock_config_manager):
        """Test that the ingestion service uses cache correctly."""
        # Set up ingestion service with cache
        service = IngestionService(cache_dir=cache_dir, cache_ttl=3600, use_cache=True)
        
        # Mock the ArxivRetriever
        with patch("ragdoll.ingestion.ingestion_service.ArxivRetriever") as mock_arxiv:
            mock_retriever = MagicMock()
            mock_retriever.get_relevant_documents.return_value = sample_documents
            mock_arxiv.return_value = mock_retriever
            
            # First call should use the retriever
            source = Source(type=SourceType.ARXIV, identifier="1234.5678")
            docs = service._load_source(source)
            
            # Verify retriever was called
            mock_retriever.get_relevant_documents.assert_called_once_with(query="1234.5678")
            assert docs == sample_documents
            
            # Reset mock
            mock_retriever.get_relevant_documents.reset_mock()
            
            # Second call should use cache
            docs = service._load_source(source)
            
            # Verify retriever was NOT called
            mock_retriever.get_relevant_documents.assert_not_called()
            assert docs == sample_documents
    
    def test_website_caching(self, cache_dir, sample_documents, mock_config_manager):
        """Test caching for website sources."""
        # Set up ingestion service with cache
        service = IngestionService(cache_dir=cache_dir, cache_ttl=3600, use_cache=True)
        
        # Mock the WebLoader
        with patch("ragdoll.ingestion.ingestion_service.WebLoader") as mock_web:
            mock_loader = MagicMock()
            mock_loader.load.return_value = sample_documents
            mock_web.return_value = mock_loader
            
            # First call should use the loader
            source = Source(type=SourceType.WEBSITE, identifier="https://example.com")
            docs = service._load_source(source)
            
            # Verify loader was called
            mock_loader.load.assert_called_once_with("https://example.com")
            assert docs == sample_documents
            
            # Reset mock
            mock_loader.load.reset_mock()
            
            # Second call should use cache
            docs = service._load_source(source)
            
            # Verify loader was NOT called
            mock_loader.load.assert_not_called()
            assert docs == sample_documents
    
    def test_cache_disabled(self, cache_dir, sample_documents, mock_config_manager):
        """Test that caching can be disabled."""
        # Set up ingestion service with cache disabled
        service = IngestionService(cache_dir=cache_dir, use_cache=False)
        
        # Mock the ArxivRetriever
        with patch("ragdoll.ingestion.ingestion_service.ArxivRetriever") as mock_arxiv:
            mock_retriever = MagicMock()
            mock_retriever.get_relevant_documents.return_value = sample_documents
            mock_arxiv.return_value = mock_retriever
            
            # First call should use the retriever
            source = Source(type=SourceType.ARXIV, identifier="1234.5678")
            service._load_source(source)
            
            # Reset mock
            mock_retriever.get_relevant_documents.reset_mock()
            
            # Second call should also use retriever (no caching)
            service._load_source(source)
            
            # Verify retriever was called again
            mock_retriever.get_relevant_documents.assert_called_once_with(query="1234.5678")
    
    def test_clear_cache_through_service(self, cache_dir, sample_documents, mock_config_manager):
        """Test clearing cache via the ingestion service."""
        # Set up ingestion service with cache
        service = IngestionService(cache_dir=cache_dir, cache_ttl=3600, use_cache=True)
        
        # Create some cached content
        with patch("ragdoll.ingestion.ingestion_service.ArxivRetriever") as mock_arxiv:
            mock_retriever = MagicMock()
            mock_retriever.get_relevant_documents.return_value = sample_documents
            mock_arxiv.return_value = mock_retriever
            
            # Cache some content
            source1 = Source(type=SourceType.ARXIV, identifier="1234.5678")
            source2 = Source(type=SourceType.ARXIV, identifier="8765.4321")
            
            service._load_source(source1)
            service._load_source(source2)
            
            # Clear specific cache entry
            count = service.clear_cache("arxiv", "1234.5678")
            assert count == 1
            
            # First source should require retrieval again
            mock_retriever.get_relevant_documents.reset_mock()
            service._load_source(source1)
            mock_retriever.get_relevant_documents.assert_called_once()
            
            # Second source should still be cached
            mock_retriever.get_relevant_documents.reset_mock()
            service._load_source(source2)
            mock_retriever.get_relevant_documents.assert_not_called()
            
            # Clear all cache
            service.clear_cache()
            
            # Both should require retrieval again
            mock_retriever.get_relevant_documents.reset_mock()
            service._load_source(source2)
            mock_retriever.get_relevant_documents.assert_called_once()