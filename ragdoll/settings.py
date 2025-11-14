"""
Shared configuration access helpers.

This module ensures the configuration YAML is only loaded once per process and
provides convenient accessors for typed sections.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from ragdoll.app_config import AppConfig, CONFIG_ENV_VAR, bootstrap_app
from ragdoll.config import (
    CacheConfig,
    Config,
    EmbeddingsConfig,
    EntityExtractionConfig,
    GraphDatabaseConfig,
    IngestionConfig,
    LLMPromptsConfig,
    LoadersConfig,
    MonitorConfig,
)


@lru_cache(maxsize=1)
def get_app(config_path: Optional[str] = None) -> AppConfig:
    """
    Return the shared :class:`AppConfig` instance.

    Order of precedence for config path:
      1. Explicit ``config_path`` argument.
      2. ``RAGDOLL_CONFIG_PATH`` environment variable.
      3. Default path baked into ConfigManager.
    """

    resolved_path = config_path or os.environ.get(CONFIG_ENV_VAR)
    return bootstrap_app(resolved_path)


def get_config_manager(config_path: Optional[str] = None) -> Config:
    """Compatibility shim returning the ConfigManager from the shared app."""

    return get_app(config_path).config


def get_cache_config() -> CacheConfig:
    return get_config_manager().cache_config


def get_embeddings_config() -> EmbeddingsConfig:
    return get_config_manager().embeddings_config


def get_ingestion_config() -> IngestionConfig:
    return get_config_manager().ingestion_config


def get_loaders_config() -> LoadersConfig:
    return get_config_manager().ingestion_config.loaders


def get_monitor_config() -> MonitorConfig:
    return get_config_manager().monitor_config


def get_llm_prompts_config() -> LLMPromptsConfig:
    return get_config_manager().llm_prompts_config


def get_entity_extraction_config() -> EntityExtractionConfig:
    return get_config_manager().entity_extraction_config


def get_graph_database_config() -> GraphDatabaseConfig:
    return get_config_manager().entity_extraction_config.graph_database_config
