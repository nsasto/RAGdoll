"""
LLM ingestion service for RAGdoll

This module provides utilities for extracting data from various file formats 
and building a structured data pipeline for data ingestion into graph and vector db.
"""

import logging

logger = logging.getLogger("ragdoll.ingestion")

from ragdoll.ingestion.content_extraction import ContentExtractionService, Source

__all__ = [
    "ContentExtractionService",
    "Source",
]
