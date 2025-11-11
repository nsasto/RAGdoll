import os
import time
import json
import tempfile
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
from ragdoll.metrics.metrics_manager import MetricsManager

class TestMetricsManager:
    @pytest.fixture
    def metrics_dir(self):
        """Create a temporary directory for metrics testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def metrics_manager(self, metrics_dir):
        """Create a metrics manager for testing."""
        return MetricsManager(metrics_dir=metrics_dir)
    
    def test_full_session_workflow(self, metrics_manager):
        """Test the full metrics workflow from session start to end."""
        # Start session
        session = metrics_manager.start_session(input_count=3)
        assert session["input_count"] == 3
        assert session["timestamp_start"] is not None
        
        # Start tracking first source
        source1 = metrics_manager.start_source(batch_id=1, source_id="test1.txt", source_type="file")
        assert source1["batch_id"] == 1
        assert source1["source_id"] == "test1.txt"
        
        # End tracking first source as success
        time.sleep(0.01)  # Small delay to ensure timing metrics
        source1 = metrics_manager.end_source(
            batch_id=1, 
            source_id="test1.txt", 
            success=True, 
            document_count=5, 
            bytes_count=1024
        )
        assert source1["success"] is True
        assert source1["document_count"] == 5
        assert source1["bytes"] == 1024
        assert source1["processing_time_ms"] > 0
        
        # Start and end second source as failure
        source2 = metrics_manager.start_source(batch_id=1, source_id="test2.txt", source_type="file")
        time.sleep(0.01)
        source2 = metrics_manager.end_source(
            batch_id=1, 
            source_id="test2.txt", 
            success=False, 
            error_message="File not found"
        )
        assert source2["success"] is False
        assert source2["error"] == "File not found"
        
        # End session
        final_metrics = metrics_manager.end_session(document_count=5)
        assert final_metrics["document_count"] == 5
        assert final_metrics["success_count"] == 1
        assert final_metrics["failure_count"] == 1
        assert final_metrics["duration_seconds"] > 0
        assert 0 < final_metrics["success_rate"] < 1
        
        # Verify file was saved
        metrics_files = list(Path(metrics_manager.metrics_dir).glob("session_*.json"))
        assert len(metrics_files) == 1
        
        # Check file contents
        with open(metrics_files[0], "r") as f:
            saved_data = json.load(f)
        assert saved_data["document_count"] == 5
        assert saved_data["success_rate"] == 0.5  # 1 success, 1 failure
    
    def test_get_recent_sessions(self, metrics_manager, metrics_dir):
        """Test retrieving recent sessions."""
        # Create test session files
        for i in range(3):
            session_data = {
                "session_id": f"test{i}",
                "timestamp_start": "2023-01-01T00:00:00",
                "document_count": i * 10,
                "success_rate": 0.8
            }
            
            with open(os.path.join(metrics_dir, f"session_test{i}.json"), "w") as f:
                json.dump(session_data, f)
                
            # Add small delay so file modification times are different
            time.sleep(0.1)
        
        # Get recent sessions
        recent = metrics_manager.get_recent_sessions(limit=2)
        assert len(recent) == 2
        # Should return the most recent ones (highest indices)
        assert recent[0]["session_id"] == "test2"
        assert recent[1]["session_id"] == "test1"
