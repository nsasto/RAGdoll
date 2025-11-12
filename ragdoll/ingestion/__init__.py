"""
LLM ingestion service for RAGdoll

This module provides utilities for extracting data from various file formats 
and building a structured data pipeline for data ingestion into graph and vector db.
"""

import logging

logger = logging.getLogger("ragdoll.ingestion")

from ragdoll.ingestion.document_loaders import DocumentLoaderService, Source

__all__ = [
    "DocumentLoaderService",
    "Source",
]
