"""
Configuration management module for RAGdoll.

This module provides configuration loading, validation, and access
for all components of the RAGdoll system. It includes predefined
configuration schemas for various system components and a ConfigManager
class for handling configuration files.
"""

from ragdoll.config.config_manager import ConfigManager as Config
from ragdoll.config.base_config import (
    # Base configs
    BaseConfig,
    # Component configs
    LoaderConfig,
    ChunkerConfig,
    # System configs
    LoadersConfig,
    IngestionConfig,
    EmbeddingsConfig,
    CacheConfig,
    MonitorConfig,
    LLMPromptsConfig,
    GraphDatabaseConfig,
    EntityExtractionConfig,
)

__all__ = [
    # Main config manager
    "Config",
    # Base config classes
    "BaseConfig",
    # Component config classes
    "LoaderConfig",
    "ChunkerConfig",
    # System config classes
    "LoadersConfig",
    "IngestionConfig",
    "EmbeddingsConfig",
    "CacheConfig",
    "MonitorConfig",
    "LLMPromptsConfig",
    "GraphDatabaseConfig",
    "EntityExtractionConfig",
]
