"""
Legacy import path for the ingestion pipeline.

The pipeline modules now live under ``ragdoll.pipeline``.
This file re-exports the public API for backward compatibility.
"""

from ragdoll.pipeline import IngestionPipeline, IngestionOptions, ingest_documents

__all__ = ["IngestionPipeline", "IngestionOptions", "ingest_documents"]
