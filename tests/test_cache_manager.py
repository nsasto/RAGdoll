import pytest
import os
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ragdoll.ingestion import ContentExtractionService, Source
from ragdoll.cache.cache_manager import CacheManager


# Fixture for a ContentExtractionService with mocked cache manager
@pytest.fixture
def content_extraction_service_with_mock_cache():
    mock_cache_manager = MagicMock(spec=CacheManager)
    service = ContentExtractionService(cache_manager=mock_cache_manager, use_cache=True)
    return service, mock_cache_manager

# Fixture for a ContentExtractionService with caching disabled
@pytest.fixture
def content_extraction_service_without_cache():
    service = ContentExtractionService(use_cache=False)
    return service

# Test clear_cache method
class TestClearCache:
    def test_clear_cache_with_cache_enabled(self, content_extraction_service_with_mock_cache):
        service, mock_cache_manager = content_extraction_service_with_mock_cache
        service.clear_cache("test_type", "test_identifier")
        mock_cache_manager.clear_cache.assert_called_once_with("test_type", "test_identifier")

    def test_clear_cache_with_cache_disabled(self, content_extraction_service_without_cache):
        service = content_extraction_service_without_cache
        # clear_cache should do nothing if cache is disabled
        # We can assert that no calls were made to a cache manager method
        with patch.object(service, 'cache_manager') as mock_cache_manager:
             service.clear_cache("test_type", "test_identifier")
             mock_cache_manager.assert_not_called()


# Test get_metrics method (assuming metrics are handled by MetricsManager)
class TestGetMetrics:
    # This test assumes MetricsManager is integrated into ContentExtractionService
    # and has methods like get_recent_sessions and get_aggregate_metrics
    # Adjust mock setup and assertions based on actual implementation

    # Update the test method to use a more direct approach
    def test_get_metrics_with_metrics_enabled(self):
        # Create a mock metrics manager directly
        mock_metrics_manager = MagicMock()
        mock_metrics_manager.get_recent_sessions.return_value = ["session1", "session2"]
        mock_metrics_manager.get_aggregate_metrics.return_value = {"total_docs": 100}
        
        # Use the mock directly AND explicitly set collect_metrics to True
        service = ContentExtractionService(
            metrics_manager=mock_metrics_manager,
            collect_metrics=True  # This is the key fix
        )
        
        metrics = service.get_metrics()
        
        # First check what we're actually getting to debug
        print(f"Metrics returned: {metrics}")
        
        # Then run assertions
        assert metrics["enabled"] is True
        assert metrics["recent_sessions"] == ["session1", "session2"]
        assert metrics["aggregate"] == {"total_docs": 100}
        mock_metrics_manager.get_recent_sessions.assert_called_once_with(limit=5)
        mock_metrics_manager.get_aggregate_metrics.assert_called_once_with(days=30)

    def test_get_metrics_with_metrics_disabled(self):
        service = ContentExtractionService(metrics_manager=None)
        metrics = service.get_metrics()
        assert metrics["enabled"] is False
        assert "message" in metrics


# Add more tests for other ContentExtractionService methods as needed
