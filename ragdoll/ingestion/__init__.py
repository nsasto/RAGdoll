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

# Simple registry for loader classes. Loaders can register themselves
# with a short name which can be referenced in configuration.
from typing import Dict, Type, Optional, List

_loader_registry: Dict[str, Type] = {}


def register_loader(name: str):
    """Decorator to register a loader class under a short name."""

    def _decorator(cls: Type):
        _loader_registry[_normalize_name(name)] = cls
        return cls

    return _decorator


def get_loader(name: str) -> Optional[Type]:
    """Return a registered loader class for the short name or None."""
    return _loader_registry.get(_normalize_name(name))


def register_loader_class(name: str, cls: Type) -> None:
    """Programmatically register a loader class under a given name."""
    _loader_registry[_normalize_name(name)] = cls


def _normalize_name(name: str) -> str:
    """Normalize registry keys: strip leading '.' and lowercase the name."""
    if not isinstance(name, str):
        return name
    return name.lstrip(".").lower()


def list_loaders() -> List[str]:
    return sorted(_loader_registry.keys())


def clear_loader_registry() -> None:
    """Clear registry (useful for tests)."""

    _loader_registry.clear()

__all__.extend(["register_loader", "get_loader", "list_loaders", "clear_loader_registry"])
