import os
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

class MetricsManager:
    """Collects and manages metrics for document ingestion process."""
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, metrics_dir: Optional[str] = None):
        """
        Initialize the metrics manager.
        
        Args:
            metrics_dir: Directory to store metrics data. If None, uses ~/.ragdoll/metrics/
        """
        if metrics_dir is None:
            metrics_dir = os.path.join(os.path.expanduser("~"), ".ragdoll", "metrics")
        
        self.metrics_dir = Path(metrics_dir)
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Current session data
        self.current_session = None
        self.sources_metrics = {}
        
        self.logger.info(f"Metrics system initialized with storage at {self.metrics_dir}")
    
    def start_session(self, input_count: int) -> Dict[str, Any]:
        """
        Start a new metrics collection session.
        
        Args:
            input_count: Number of input sources being processed
            
        Returns:
            Session information dictionary
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        self.current_session = {
            "session_id": session_id,
            "timestamp_start": timestamp,
            "timestamp_end": None,
            "input_count": input_count,
            "success_count": 0,
            "failure_count": 0,
            "document_count": 0,
            "total_bytes": 0,
            "total_processing_time_ms": 0,
            "sources": {}
        }
        
        self.sources_metrics = {}
        self.logger.info(f"Started metrics session {session_id} with {input_count} inputs")
        return self.current_session
    
    def start_source(self, batch_id: int, source_id: str, source_type: str) -> Dict[str, Any]:
        """
        Start tracking metrics for a source.
        
        Args:
            batch_id: ID of the batch containing this source
            source_id: Identifier for the source
            source_type: Type of the source (file, website, arxiv, etc.)
            
        Returns:
            Source metrics information dictionary
        """
        if self.current_session is None:
            self.logger.warning("Attempted to start source metrics without an active session")
            return {}
        
        timestamp = datetime.now()
        
        # Use a consistent key format
        source_key = f"{source_id}"
        
        source_metrics = {
            "batch_id": batch_id,
            "source_id": source_id,
            "source_type": source_type,
            "timestamp_start": timestamp.isoformat(),
            "timestamp_end": None,
            "processing_time_ms": 0,
            "success": None,
            "document_count": 0,
            "bytes": 0,
            "error": None
        }
        
        self.sources_metrics[source_key] = {
            "metrics": source_metrics,
            "start_time": time.time()
        }
        
        return source_metrics
    
    def end_source(self, batch_id: int, source_id: str, success: bool, 
                   document_count: int = 0, bytes_count: int = 0,
                   error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        End tracking metrics for a source.
        
        Args:
            batch_id: ID of the batch containing this source
            source_id: Identifier for the source
            success: Whether the source was processed successfully
            document_count: Number of documents extracted
            bytes_count: Number of bytes processed
            error_message: Error message if processing failed
            
        Returns:
            Updated source metrics
        """
        if self.current_session is None:
            self.logger.warning("Attempted to end source metrics without an active session")
            return {}
        
        # Fix: Use the same key format as in start_source
        source_key = f"{source_id}"
        if source_key not in self.sources_metrics:
            self.logger.warning(f"No start metrics found for source {source_key}")
            return {}
        
        source_data = self.sources_metrics[source_key]
        source_metrics = source_data["metrics"]
        
        # Calculate processing time
        end_time = time.time()
        processing_time_ms = int((end_time - source_data["start_time"]) * 1000)
        
        # Update metrics
        source_metrics["timestamp_end"] = datetime.now().isoformat()
        source_metrics["processing_time_ms"] = processing_time_ms
        source_metrics["success"] = success
        source_metrics["document_count"] = document_count
        source_metrics["bytes"] = bytes_count
        source_metrics["error"] = error_message
        
        # Update session aggregates
        if self.current_session:
            if success:
                self.current_session["success_count"] += 1
            else:
                self.current_session["failure_count"] += 1
            
            self.current_session["document_count"] += document_count
            self.current_session["total_bytes"] += bytes_count
            self.current_session["total_processing_time_ms"] += processing_time_ms
            self.current_session["sources"][source_key] = source_metrics
        
        return source_metrics
    
    def end_session(self, document_count: Optional[int] = None) -> Dict[str, Any]:
        """
        End the current metrics session and save results.
        
        Args:
            document_count: Total number of documents processed (if known)
            
        Returns:
            Complete session metrics
        """
        if self.current_session is None:
            self.logger.warning("Attempted to end metrics session that wasn't started")
            return {}
        
        self.current_session["timestamp_end"] = datetime.now().isoformat()
        
        # If document count is provided, use it (it might be more accurate than our running count)
        if document_count is not None:
            self.current_session["document_count"] = document_count
        
        # Calculate session duration
        start_time = datetime.fromisoformat(self.current_session["timestamp_start"])
        end_time = datetime.fromisoformat(self.current_session["timestamp_end"])
        duration_seconds = (end_time - start_time).total_seconds()
        self.current_session["duration_seconds"] = duration_seconds
        
        # Calculate success rate
        total_sources = self.current_session["success_count"] + self.current_session["failure_count"]
        if total_sources > 0:
            self.current_session["success_rate"] = self.current_session["success_count"] / total_sources
        else:
            self.current_session["success_rate"] = 0
        
        # Save to file
        session_filename = f"session_{self.current_session['session_id']}.json"
        session_path = self.metrics_dir / session_filename
        
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(self.current_session, f, indent=2)
            
            self.logger.info(f"Metrics session completed and saved to {session_path}")
            self.logger.info(f"Processed {self.current_session['document_count']} documents with "
                           f"{self.current_session['success_rate']:.1%} success rate")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
        
        session_data = self.current_session
        self.current_session = None
        self.sources_metrics = {}
        
        return session_data
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent metrics sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metrics dictionaries
        """
        sessions = []
        
        try:
            json_files = list(self.metrics_dir.glob("session_*.json"))
            # Sort by modification time (most recent first)
            json_files.sort(key=os.path.getmtime, reverse=True)
            
            # Load the most recent files
            for file_path in json_files[:limit]:
                with open(file_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
                    
        except Exception as e:
            self.logger.error(f"Error retrieving session metrics: {e}")
            
        return sessions
    
    def get_aggregate_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get aggregate metrics over a time period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Aggregate metrics
        """
        # Fix: Use timedelta for proper date calculation
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date - timedelta(days=days)  # Use timedelta instead of day replacement

        aggregate = {
            "total_sessions": 0,
            "total_documents": 0,
            "total_sources": 0,
            "successful_sources": 0,
            "failed_sources": 0,
            "avg_success_rate": 0,
            "avg_documents_per_source": 0,
            "avg_processing_time_ms": 0,
            "by_source_type": {}
        }
        
        try:
            json_files = list(self.metrics_dir.glob("session_*.json"))
            
            # Process each session file
            for file_path in json_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    session = json.load(f)
                
                # Skip if older than cutoff
                session_date = datetime.fromisoformat(session.get("timestamp_start", ""))
                if session_date < cutoff_date:
                    continue
                
                # Update aggregate metrics
                aggregate["total_sessions"] += 1
                aggregate["total_documents"] += session.get("document_count", 0)
                aggregate["total_sources"] += session.get("success_count", 0) + session.get("failure_count", 0)
                aggregate["successful_sources"] += session.get("success_count", 0)
                aggregate["failed_sources"] += session.get("failure_count", 0)
                
                # Process by source type
                for source_key, source_metrics in session.get("sources", {}).items():
                    source_type = source_metrics.get("source_type", "unknown")
                    
                    if source_type not in aggregate["by_source_type"]:
                        aggregate["by_source_type"][source_type] = {
                            "count": 0,
                            "success_count": 0,
                            "document_count": 0,
                            "total_processing_time_ms": 0
                        }
                    
                    type_metrics = aggregate["by_source_type"][source_type]
                    type_metrics["count"] += 1
                    
                    if source_metrics.get("success", False):
                        type_metrics["success_count"] += 1
                    
                    type_metrics["document_count"] += source_metrics.get("document_count", 0)
                    type_metrics["total_processing_time_ms"] += source_metrics.get("processing_time_ms", 0)
            
            # Calculate averages
            if aggregate["total_sources"] > 0:
                aggregate["avg_success_rate"] = aggregate["successful_sources"] / aggregate["total_sources"]
                aggregate["avg_documents_per_source"] = aggregate["total_documents"] / aggregate["total_sources"]
                
                total_time = 0
                total_items = 0
                for source_type, metrics in aggregate["by_source_type"].items():
                    total_time += metrics["total_processing_time_ms"]
                    total_items += metrics["count"]
                    
                    # Calculate source type specific metrics
                    if metrics["count"] > 0:
                        metrics["avg_processing_time_ms"] = metrics["total_processing_time_ms"] / metrics["count"]
                        metrics["avg_documents"] = metrics["document_count"] / metrics["count"]
                        metrics["success_rate"] = metrics["success_count"] / metrics["count"]
                
                if total_items > 0:
                    aggregate["avg_processing_time_ms"] = total_time / total_items
            
        except Exception as e:
            self.logger.error(f"Error calculating aggregate metrics: {e}")
            
        return aggregate