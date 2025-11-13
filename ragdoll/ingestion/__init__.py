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
from pathlib import Path
from importlib import import_module

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


# --- bootstrap: register default-config loaders if possible -------------------
def _bootstrap_register_default_loaders() -> None:
    """
    Load ragdoll/config/default_config.yaml and register any file_mappings
    found. This reads the YAML file directly (no ragdoll.config import) and
    performs a best-effort import+registration of loader classes under
    normalized short names (strip leading '.' and lowercase).
    """
    try:
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"
        if not cfg_path.exists():
            # try one level up if layout differs
            cfg_path = Path(__file__).resolve().parents[2] / "ragdoll" / "config" / "default_config.yaml"
        if not cfg_path.exists():
            logger.debug("No default_config.yaml found at %s", cfg_path)
            return

        text = cfg_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug("Default config not found for loader bootstrap: %s", e)
        return

    try:
        import yaml

        cfg = yaml.safe_load(text) or {}
    except Exception as e:
        logger.debug("Could not parse default_config.yaml for loader bootstrap: %s", e)
        return

    # file_mappings live under ingestion.loaders.file_mappings in config
    ingestion_cfg = cfg.get("ingestion", {}) or {}
    loaders_block = ingestion_cfg.get("loaders", {}) or {}
    file_mappings = loaders_block.get("file_mappings", {}) or {}

    for ext, loader_ref in file_mappings.items():
        name = _normalize_name(ext)
        if not loader_ref or not isinstance(loader_ref, str):
            continue
        try:
            # support "pkg.module:Class" and "pkg.module.Class"
            ref = loader_ref.replace(":", ".")
            module_path, _, class_name = ref.rpartition('.')
            if not module_path or not class_name:
                raise ImportError(f"Malformed loader import string '{loader_ref}'")
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            register_loader_class(name, cls)
            logger.debug("Registered loader '%s' -> %s", name, loader_ref)
        except Exception as e:
            logger.debug("Skipping registration for loader '%s' (%s): %s", name, loader_ref, e)


try:
    _bootstrap_register_default_loaders()
except Exception as _e:
    logger.debug("Loader bootstrap failed: %s", _e)
# ---------------------------------------------------------------------------
