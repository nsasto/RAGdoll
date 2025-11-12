from typing import Dict, Any, Optional
import logging

from ragdoll import settings
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

logger = logging.getLogger("ragdoll.entity_extraction")

try:
    from .entity_extraction_service import GraphCreationService
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    GraphCreationService = None  # type: ignore[assignment]
    logger.warning(
        "GraphCreationService could not be imported (missing optional dependency): %s",
        exc,
    )

def get_entity_extractor(
    extractor_type: str = None,
    config_manager = None,
    config: Dict[str, Any] = None,
    **kwargs
) -> BaseEntityExtractor:
    """
    Factory function for getting an entity extractor.
    
    Args:
        extractor_type: Type of extractor (defaults to 'graph_creation_service')
        config_manager: Optional ConfigManager instance
        config: Optional configuration dictionary
        **kwargs: Additional parameters to override config settings
        
    Returns:
        A BaseEntityExtractor instance
    """
    # Initialize config
    extraction_config = {}
    if config_manager is not None:
        extraction_config = config_manager._config.get("entity_extraction", {})
    elif config is not None:
        if isinstance(config, dict):
            if "entity_extraction" in config:
                extraction_config = config["entity_extraction"]
            else:
                extraction_config = config
    else:
        extraction_config = settings.get_config_manager()._config.get("entity_extraction", {})
    
    # Merge kwargs into extraction_config (kwargs take priority)
    merged_config = {**extraction_config, **kwargs}
    
    # Determine extractor type (priority: parameter > config > default)
    actual_extractor_type = extractor_type or extraction_config.get("extractor_type", "graph_creation_service")
    
    logger.info(f"Creating entity extractor of type: {actual_extractor_type}")
    
    # Create the appropriate extractor instance
    if actual_extractor_type == "graph_creation_service":
        if GraphCreationService is None:
            raise ImportError(
                "GraphCreationService is unavailable. Install optional dependencies like 'spacy' to enable it."
            )
        return GraphCreationService(config=merged_config)
    else:
        logger.warning(
            f"Unknown extractor type: {actual_extractor_type}, defaulting to GraphCreationService"
        )
        if GraphCreationService is None:
            raise ImportError(
                "GraphCreationService is unavailable. Install optional dependencies like 'spacy' to enable it."
            )
        return GraphCreationService(config=merged_config)

__all__ = [
    "BaseEntityExtractor",  # Export the base class
    "Entity",
    "Relationship", 
    "EntityList",
    "RelationshipList",
    "GraphNode",
    "GraphEdge",
    "Graph",
    "GraphCreationService",
    "get_entity_extractor",
]
