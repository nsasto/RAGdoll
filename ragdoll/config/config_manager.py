import os
import yaml
from importlib import import_module
from typing import Dict, Any, Type
from pathlib import Path

from ragdoll.config.base_config import IngestionConfig

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
        if not config_path:
            # Use default config in the package
            config_path = Path(__file__).parent / "default_config.yaml"
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def ingestion_config(self) -> IngestionConfig:
        """Get the ingestion configuration."""
        # Use model_validate instead of parse_obj (Pydantic v2 compatibility)
        return IngestionConfig.model_validate(self._config.get("ingestion", {}))
    
    def get_loader_mapping(self) -> Dict[str, Type]:
        """
        Get the loader mapping with imported classes.
        
        Returns:
            Dictionary mapping file extensions to loader classes.
        """
        config = self.ingestion_config.loader_mappings.mappings
        result = {}
        
        for ext, class_path in config.items():
            try:
                # Split into module path and class name
                module_path, class_name = class_path.rsplit(".", 1)
                
                # Import the module
                module = import_module(module_path)
                
                # Get the class
                loader_class = getattr(module, class_name)
                
                # Add to result
                result[ext] = loader_class
            except (ImportError, AttributeError, ValueError) as e:
                # Log the error but continue with other loaders
                print(f"Error loading loader for extension {ext}: {e}")
        
        return result