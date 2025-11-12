import os
import yaml
import logging
from importlib import import_module
from typing import Dict, Any, Type
from pathlib import Path

from ragdoll.config.base_config import (
    CacheConfig,
    EmbeddingsConfig,
    IngestionConfig,
    LoadersConfig,
    MonitorConfig,
    VectorStoreConfig,
)

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
        self.logger.debug(f"Loaded config: {self._config}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment."""

        if not os.path.exists(self.config_path):
            self.logger.debug(f"Config file not found at {self.config_path}. Using default values.")
        else:
            self.logger.debug(f"Config file loaded from {self.config_path}.")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Retrieve OpenAI API key from environment if not in config
        if "embeddings" in config and "openai" in config["embeddings"] and "openai_api_key" not in config["embeddings"]["openai"]:
            config["embeddings"]["openai"]["openai_api_key"] = os.environ.get(
                "OPENAI_API_KEY"
            )
        if "embeddings" in config and "default_client" not in config["embeddings"]:
             config["embeddings"]["default_client"] = "openai"
              
        if "embeddings" in config and "default_model" not in config["embeddings"]:
             config["embeddings"]["default_model"] = "openai"

        if "embeddings" in config and "clients" not in config["embeddings"]:
            config["embeddings"]["clients"] = {"openai":config["embeddings"]["openai"], "huggingface":config["embeddings"]["huggingface"]}
        
        return config
    
    @property
    def embeddings_config(self) -> EmbeddingsConfig:
        """Get the embeddings configuration."""
        return EmbeddingsConfig.model_validate(self._config.get("embeddings", {})) 

    @property
    def cache_config(self) -> CacheConfig:
        """Get the cache configuration."""
        return CacheConfig.model_validate(self._config.get("cache", {}))
    
    @property
    def monitor_config(self) -> MonitorConfig:
        """Get the monitor configuration."""
        return MonitorConfig.model_validate(self._config.get("monitor", {}))

    @property
    def vector_store_config(self) -> VectorStoreConfig:
        """Get the vector store configuration."""
        return VectorStoreConfig.model_validate(self._config.get("vector_store", {}))



    

    
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
        loaders_config = self.ingestion_config.loaders
        result = {}
        if loaders_config and loaders_config.file_mappings:
            self.logger.debug(f"Loaders {loaders_config.file_mappings}")
            for ext, class_path in loaders_config.file_mappings.items():

                self.logger.debug(f"Loading: {ext}, {class_path}")
                
                try:
                    # Split into module path and class name
                    module_path, class_name = class_path.rsplit(".", 1)

                    self.logger.debug(f"Loading module: {module_path}, Class: {class_name}")


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

    def get_source_loader_mapping(self) -> Dict[str, Type]:
        """
        Get the source loader mapping with imported classes.
        
        Returns:
            Dictionary mapping source types to loader classes.
        """
        source_loaders = {}
        loaders_config = self.ingestion_config.loaders

        # Copy file mappings to source_loaders
        for ext, class_path in loaders_config.file_mappings.items():
            if ext not in source_loaders:
                try:
                    module_path, class_name = class_path.rsplit(".", 1)
                    module = import_module(module_path)
                    loader_class = getattr(module, class_name)
                    source_loaders[ext] = loader_class
                    self.logger.debug(f"Loaded loader for extension {ext} source: {class_path}")
                except (ImportError, AttributeError, ValueError) as e:
                    self.logger.warning(
                        f"Error loading loader for extension {ext} source: {e}. This source type will not be supported."
                    )
        
        # Add arxiv_retriever if present
        if "arxiv_retriever" in loaders_config:
            class_path = loaders_config["arxiv_retriever"]
            if "arxiv" not in source_loaders:
                try:
                    module_path, class_name = class_path.rsplit(".", 1)
                    module = import_module(module_path)
                    loader_class = getattr(module, class_name)
                    source_loaders["arxiv"] = loader_class
                    self.logger.debug(f"Loaded loader for arxiv source: {class_path}")
                except (ImportError, AttributeError, ValueError) as e:
                    self.logger.warning(
                        f"Error loading loader for arxiv source: {e}. This source type will not be supported."
                    )

        self.logger.info(f"Loaded {len(source_loaders)} source loaders")
        return source_loaders
        

