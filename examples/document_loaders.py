# # Ingestion Service
# ## Basic Usage

import os
import glob
from pathlib import Path
from ragdoll.ingestion import DocumentLoaderService

# Get absolute path to the test_data directory
current_file = Path(os.path.abspath(""))  # Current notebook directory
test_data_dir = (current_file.parent / "tests" / "test_data").resolve()

# Find all files using glob
file_paths = glob.glob(str(test_data_dir / "*"))
print(f"Found {len(file_paths)} files")


# Create ingestion service with default settings

service = DocumentLoaderService()
# Process all documents
documents = service.ingest_documents(file_paths)

# Show how many documents were extracted
print(f"Processed {len(documents)} documents")


import json

# Access the first document
if documents:
    doc = documents[0]
    print(f"First document content (preview): {doc.page_content[:100]}...")
    print(f"Metadata:\n {json.dumps(doc.metadata, indent=2)}")

# ## Working with different file types

from ragdoll.ingestion import DocumentLoaderService

# Initialize service
service = DocumentLoaderService()
# Process files of different types
pdf_docs = service.ingest_documents(["../tests/test_data/test_pdf.pdf"])
text_docs = service.ingest_documents(["../tests/test_data/test_txt.txt", "../tests/test_data/test_txt.txt"])
docx_docs = service.ingest_documents(["../tests/test_data/test_docx.docx"])

# Process HTML from URLs
web_docs = service.ingest_documents(["https://github.com/nsasto/langchain-markitdown"])

# Combine all documents
all_docs = pdf_docs + text_docs + docx_docs + web_docs

print(f"Total documents: {len(all_docs)}")
print(f"Documents by type:")
print(f"  - PDF: {len(pdf_docs)}")
print(f"  - Text: {len(text_docs)}")
print(f"  - DOCX: {len(docx_docs)}")
print(f"  - Web: {len(web_docs)}")

# ## Customizing Ingestion Settings

# Modified initialization with supported parameters
from ragdoll.ingestion import DocumentLoaderService

# Initialize with only the supported parameters
service = DocumentLoaderService(
    max_threads=4,                # Limit concurrency
    batch_size=10,                # Process files in batches of 10
    use_cache=True,               # Enable caching
    collect_metrics=True          # Enable metrics collection
)

# Process documents - pass file_paths directly, not [file_paths]
documents = service.ingest_documents(file_paths)

print(f"Processed {len(documents)} document chunks")

# Document properties can be accessed differently depending on type
if documents:
    doc = documents[0]
    if hasattr(doc, 'page_content'):
        content_length = len(doc.page_content)
    elif isinstance(doc, dict) and 'page_content' in doc:
        content_length = len(doc['page_content'])
    else:
        content_length = 0
    print(f"First document size: {content_length} characters")

# ## Working with Caching

# Complete caching performance test
from ragdoll.ingestion import DocumentLoaderService
import time
import statistics

def measure_processing_time(use_cache: bool, file_path: str, runs: int = 3) -> dict:
    """Measure document processing time with or without cache."""
    times = []
    
    service = DocumentLoaderService(use_cache=use_cache)
    
    # Run multiple times to get average performance
    for i in range(runs):
        start = time.time()
        docs = service.ingest_documents([file_path])
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Don't sleep on the last run
        if i < runs-1:
            time.sleep(0.5)  # Short pause between runs
    
    return {
        "avg_time": statistics.mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "doc_count": len(docs),
        "runs": runs
    }

# Clear any existing cache first
service_clear = DocumentLoaderService(use_cache=True)
service_clear.clear_cache()
print("Cache cleared")

# Test with no cache
no_cache_results = measure_processing_time(False, "../tests/test_data/test_pdf.pdf", runs=3)
print("\nWithout cache:")
print(f"  Processed {no_cache_results['doc_count']} documents")
print(f"  Average time: {no_cache_results['avg_time']:.3f} seconds")
print(f"  Min time: {no_cache_results['min_time']:.3f}, Max time: {no_cache_results['max_time']:.3f}")

# Test with cache (first run populates, subsequent runs use cache)
cache_results = measure_processing_time(True, "../tests/test_data/test_pdf.pdf", runs=3)
print("\nWith cache:")
print(f"  Processed {cache_results['doc_count']} documents")
print(f"  Average time: {cache_results['avg_time']:.3f} seconds")
print(f"  Min time: {cache_results['min_time']:.3f}, Max time: {cache_results['max_time']:.3f}")

# Speed improvement calculation
if no_cache_results['avg_time'] > 0:
    improvement = (no_cache_results['avg_time'] - cache_results['avg_time']) / no_cache_results['avg_time'] * 100
    print(f"\nCache performance improvement: {improvement:.1f}%")

# ## Handling Errors

from ragdoll.ingestion import DocumentLoaderService
import logging

# Configure logging to see warnings and errors
logging.basicConfig(level=logging.INFO)

# Create service

service = DocumentLoaderService()
# Mix of valid and invalid files
files = [
    "documents/valid.pdf",
    "documents/corrupted.pdf",
    "documents/nonexistent.txt",
    "documents/valid.txt"
]

try:
    # Service will skip files it can't process
    documents = service.ingest_documents(files)
    print(f"Successfully processed {len(documents)} documents")
    
    # Check how many files were actually processed
    sources = set([doc['metadata'].get('source') for doc in documents if 'source' in doc['metadata']])
    print(f"Documents came from {len(sources)} source files")
    print(f"Source files: {sources}")
    
except Exception as e:
    print(f"Error during ingestion: {e}")

# ## Logging metrics

# Replace your current loading code with this
import os
import glob
from pathlib import Path
from ragdoll.ingestion import DocumentLoaderService

# Get absolute path to the test_data directory
current_file = Path(os.path.abspath(""))  # Current notebook directory
test_data_dir = (current_file.parent / "tests" / "test_data").resolve()

# Instead of using Path.glob(), use the glob module which handles absolute paths
file_paths = glob.glob(str(test_data_dir / "*"))
print(f"Found {len(file_paths)} files")

# ### Basic Usage

# Create service
service = DocumentLoaderService(collect_metrics=True)
metrics = service.get_metrics(days=30) 
metrics

# Create service
service = DocumentLoaderService(collect_metrics=True)

# Pass the actual file paths, not the glob pattern
service.ingest_documents(file_paths)


# Get metrics after running
metrics = service.get_metrics(days=30)  # Get metrics from the last 30 days

# Use the metrics data
print(f"Total documents processed: {metrics['aggregate']['total_documents']}")
print(f"Average success rate: {metrics['aggregate']['avg_success_rate']:.2%}")

# Print metrics for each source type
for source_type, type_metrics in metrics['aggregate']['by_source_type'].items():
    print(f"\nMetrics for {source_type} sources:")
    print(f"  Count: {type_metrics['count']}")
    print(f"  Success rate: {type_metrics['success_rate']:.2%}")
    print(f"  Average documents: {type_metrics['avg_documents']:.1f}")
    print(f"  Average processing time: {type_metrics['avg_processing_time_ms']:.1f}ms")

# Get the most recent session details
if metrics['recent_sessions']:
    latest = metrics['recent_sessions'][0]
    print(f"\nLatest session ({latest['session_id']}):")
    print(f"  Time: {latest['timestamp_start']}")
    print(f"  Documents: {latest['document_count']}")
    print(f"  Duration: {latest['duration_seconds']:.2f} seconds")

# ### Direct Access
# 
# You can also access metrics directly from the metrics directory

import json
from pathlib import Path
import os

# Default metrics location
metrics_dir = Path.home() / ".ragdoll" / "metrics"
# Or custom location if you specified one
# metrics_dir = Path("/path/to/your/metrics")

# List all session files
session_files = list(metrics_dir.glob("session_*.json"))
# Sort by modification time (most recent first)
session_files.sort(key=os.path.getmtime, reverse=True)

# Read the most recent session
if session_files:
    with open(session_files[0], "r", encoding="utf-8") as f:
        latest_session = json.load(f)
        
    print(f"Session ID: {latest_session['session_id']}")
    print(f"Date: {latest_session['timestamp_start']}")
    print(f"Documents processed: {latest_session['document_count']}")
    print(f"Success rate: {latest_session['success_rate']:.2%}")
    
    # Print details about each source
    for source_id, source_data in latest_session["sources"].items():
        print(f"\nSource: {source_id}")
        print(f"  Type: {source_data['source_type']}")
        print(f"  Success: {source_data['success']}")
        print(f"  Documents: {source_data['document_count']}")
        print(f"  Processing time: {source_data['processing_time_ms']}ms")

# ### Displaying Outputs
# 
# Simple Metrics dashboard

# Notebook-friendly version of the dashboard script
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from ragdoll.metrics.metrics_manager import MetricsManager

def print_section(title: str):
    """Print a section title."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def print_session_summary(session: Dict[str, Any]):
    """Print a summary of a session."""
    start_time = datetime.fromisoformat(session["timestamp_start"])
    
    print(f"Session: {session['session_id']}")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {session.get('duration_seconds', 0):.2f} seconds")
    print(f"  Documents: {session['document_count']}")
    print(f"  Sources: {session['success_count'] + session['failure_count']} "
          f"({session['success_count']} successful, {session['failure_count']} failed)")
    print(f"  Success rate: {session.get('success_rate', 0):.2%}")

def format_bytes(bytes_count: int) -> str:
    """Format bytes as human-readable size."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024**2:
        return f"{bytes_count / 1024:.1f} KB"
    elif bytes_count < 1024**3:
        return f"{bytes_count / (1024**2):.1f} MB"
    else:
        return f"{bytes_count / (1024**3):.2f} GB"

# Initialize metrics manager with the path to your metrics directory
metrics_dir = Path.home() / ".ragdoll" / "metrics"
metrics_manager = MetricsManager(metrics_dir=metrics_dir)

# Show aggregate metrics and recent sessions
print_section("RAGdoll Metrics Dashboard")

# Get aggregate metrics for the last 30 days
days = 30
try:
    # Fix the date handling issue by using timedelta
    from datetime import datetime, timedelta
    
    # Monkey patch the get_aggregate_metrics method to avoid date issues
    def fixed_get_aggregate_metrics(self, days=30):
        cutoff_date = datetime.now() - timedelta(days=days)
        
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
            print(f"Error calculating aggregate metrics: {e}")
            
        return aggregate
    
    # Apply the monkey patch
    from types import MethodType
    metrics_manager.get_aggregate_metrics = MethodType(fixed_get_aggregate_metrics, metrics_manager)
    
    aggregate = metrics_manager.get_aggregate_metrics(days=days)
    
    print(f"Showing metrics for the past {days} days:")
    print(f"  Total sessions: {aggregate['total_sessions']}")
    print(f"  Total documents: {aggregate['total_documents']}")
    print(f"  Total sources: {aggregate['total_sources']}")
    print(f"  Success rate: {aggregate['avg_success_rate']:.2%}")
    print(f"  Avg processing time: {aggregate['avg_processing_time_ms']:.1f}ms per source")
    
    # Show metrics by source type
    print_section("Metrics by Source Type")
    for source_type, metrics in aggregate["by_source_type"].items():
        print(f"\n{source_type.upper()} Sources:")
        print(f"  Count: {metrics['count']}")
        print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
        print(f"  Avg documents: {metrics.get('avg_documents', 0):.1f} per source")
        print(f"  Avg processing time: {metrics.get('avg_processing_time_ms', 0):.1f}ms")
    
    # Show recent sessions
    recent_sessions = metrics_manager.get_recent_sessions(limit=5)
    
    print_section("Recent Sessions")
    for session in recent_sessions:
        print("")
        print_session_summary(session)
        
    # Pick a specific session to view in detail
    if recent_sessions:
        session_id = recent_sessions[0]["session_id"]
        print_section(f"Detailed Session Report: {session_id}")
        
        # Get the session data
        session_path = Path(metrics_manager.metrics_dir) / f"session_{session_id}.json"
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        
        print_session_summary(session)
        
        print("\nSource Details:")
        for source_id, source_data in session["sources"].items():
            success = "✅" if source_data["success"] else "❌"
            error = f" - Error: {source_data['error']}" if source_data["error"] else ""
            
            print(f"\n{success} {source_id} ({source_data['source_type']}){error}")
            print(f"  Documents: {source_data['document_count']}")
            print(f"  Size: {format_bytes(source_data['bytes'])}")
            print(f"  Processing time: {source_data['processing_time_ms']}ms")
            
except Exception as e:
    print(f"Error running dashboard: {e}")

# ### Export to file
# 
# Export to other formats

import csv

# Export to CSV
def export_sessions_to_csv(metrics_manager, output_path):
    sessions = metrics_manager.get_recent_sessions(limit=100)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['session_id', 'timestamp', 'documents', 'sources', 
                      'success_rate', 'duration_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for session in sessions:
            writer.writerow({
                'session_id': session['session_id'],
                'timestamp': session['timestamp_start'],
                'documents': session['document_count'],
                'sources': session['success_count'] + session['failure_count'],
                'success_rate': session['success_rate'],
                'duration_seconds': session['duration_seconds']
            })






