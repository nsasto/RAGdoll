import os
import yaml
import logging
from importlib import import_module
from typing import Dict, Any, Type
from pathlib import Path

from ragdoll.config.base_config import IngestionConfig

class ConfigManager:
    """Manages configuration loading and validation"""
    
    logger = logging.getLogger(__name__)
    
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
        """Load configuration from file and environment."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Retrieve OpenAI API key from environment if not in config
        if "embeddings" in config and "openai" in config["embeddings"] and "openai_api_key" not in config["embeddings"]["openai"]:
            config["embeddings"]["openai"]["openai_api_key"] = os.environ.get(
                "OPENAI_API_KEY", "your_default_api_key"
            )
        
        return config
    
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
                if not hasattr(module, class_name):
                    self.logger.warning(
                        f"Module {module_path} does not have attribute {class_name}"
                        f" for extension {ext}. Skipping this loader."
                    )
                    continue
                
                loader_class = getattr(module, class_name)
                
                # Add to result
                result[ext] = loader_class
                self.logger.debug(f"Loaded loader for extension {ext}: {class_path}")
                
            except ImportError:
                # Handle module not found
                self.logger.info(
                    f"Module {module_path} for extension {ext} could not be imported."
                    f" This extension will not be supported."
                )
            except (AttributeError, ValueError) as e:
                # Handle other errors
                self.logger.warning(
                    f"Error loading loader for extension {ext}: {e}"
                )
        
        self.logger.info(f"Loaded {len(result)} file extension loaders")
        return result