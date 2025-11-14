from __future__ import annotations

from typing import Dict, Any, Optional
import logging

from ragdoll import settings
from ragdoll.app_config import AppConfig
from .models import (
    Entity,
    Relationship,
    EntityList,
    RelationshipList,
    GraphNode,
    GraphEdge,
    Graph,
)
from .base import BaseEntityExtractor
from .entity_extraction_service import EntityExtractionService
from .graph_persistence import GraphPersistenceService

logger = logging.getLogger("ragdoll.entity_extraction")

def get_entity_extractor(
    extractor_type: str | None = None,
    config_manager=None,
    app_config: Optional[AppConfig] = None,
    config: Dict[str, Any] | None = None,
    **kwargs,
) -> BaseEntityExtractor:
    """
    Factory function for getting an entity extractor.
    
    Args:
        extractor_type: Type of extractor (defaults to 'entity_extraction_service')
        config_manager: Optional ConfigManager instance
        config: Optional configuration dictionary
        **kwargs: Additional parameters to override config settings
        
    Returns:
        A BaseEntityExtractor instance
    """
    # Initialize config
    extraction_config: Dict[str, Any] = {}
    if app_config is not None:
        extraction_config = app_config.config._config.get("entity_extraction", {})
    elif config_manager is not None:
        extraction_config = config_manager._config.get("entity_extraction", {})
    elif config is not None:
        if isinstance(config, dict):
            if "entity_extraction" in config:
                extraction_config = config["entity_extraction"]
            else:
                extraction_config = config
    else:
        extraction_config = (
            settings.get_app().config._config.get("entity_extraction", {})
        )
    
    # Merge kwargs into extraction_config (kwargs take priority)
    merged_config = {**extraction_config, **kwargs}
    
    # Determine extractor type (priority: parameter > config > default)
    actual_extractor_type = extractor_type or extraction_config.get(
        "extractor_type", "entity_extraction_service"
    )

    if actual_extractor_type == "graph_creation_service":
        logger.warning(
            "Extractor type 'graph_creation_service' is deprecated; "
            "use 'entity_extraction_service' instead."
        )
    elif actual_extractor_type != "entity_extraction_service":
        logger.warning(
            "Unknown extractor type '%s'; defaulting to EntityExtractionService",
            actual_extractor_type,
        )

    logger.info("Creating entity extractor of type: %s", actual_extractor_type)
    return EntityExtractionService(config=merged_config, app_config=app_config)

__all__ = [
    "BaseEntityExtractor",  # Export the base class
    "Entity",
    "Relationship", 
    "EntityList",
    "RelationshipList",
    "GraphNode",
    "GraphEdge",
    "Graph",
    "EntityExtractionService",
    "GraphPersistenceService",
    "get_entity_extractor",
]
