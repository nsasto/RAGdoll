import os
import yaml
import logging
import json
from importlib import import_module
from typing import Dict, Any, Type, List
from pathlib import Path

from pydantic import BaseModel, Field

from ragdoll.config.base_config import IngestionConfig, LoadersConfig, EmbeddingsConfig, CacheConfig, MonitorConfig
from ragdoll.prompts import get_prompt, list_prompts  # Import the prompt functions


class LLMPromptsConfig(BaseModel):
    """Configuration for LLM prompts"""
    entity_extraction: str = Field(default="entity_extraction")
    extract_relationships: str = Field(default="relationship_extraction")
    coreference_resolution: str = Field(default="coreference_resolution")
    entity_relationship_gleaning: str = Field(default="entity_relationship_continue")
    entity_relationship_gleaning_continue: str = Field(default="entity_relationship_gleaning_continue")


class GraphDatabaseConfig(BaseModel):
    """Configuration for graph database output"""
    output_file: str = Field(default="graph_output.json")
    uri: str = Field(default="")
    user: str = Field(default="")
    password: str = Field(default="")


class EntityExtractionConfig(BaseModel):
    """Configuration for entity extraction and graph creation"""
    enabled: bool = Field(default=True)
    spacy_model: str = Field(default="en_core_web_sm")
    chunking_strategy: str = Field(default="fixed")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=50)
    coreference_resolution_method: str = Field(default="llm")
    entity_extraction_methods: List[str] = Field(default=["ner", "llm"])
    relationship_extraction_method: str = Field(default="llm")
    entity_types: List[str] = Field(default=["PERSON", "ORG", "GPE", "DATE", "LOC"])
    relationship_types: List[str] = Field(default=["HAS_ROLE", "WORKS_FOR"])
    gleaning_enabled: bool = Field(default=True)
    max_gleaning_steps: int = Field(default=2)
    entity_linking_enabled: bool = Field(default=True)
    entity_linking_method: str = Field(default="string_similarity")
    entity_linking_threshold: float = Field(default=0.8)
    postprocessing_steps: List[str] = Field(default=["merge_similar_entities", "normalize_relations"])
    output_format: str = Field(default="json")
    graph_database_config: GraphDatabaseConfig = Field(default_factory=GraphDatabaseConfig)


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
        self.logger.debug(f"Loaded config:\n{json.dumps(self._config, indent=2)}")
        
        # Initialize available prompts
        try:
            self.available_prompts = set(list_prompts())
            self.logger.debug(f"Available prompts: {self.available_prompts}")
        except Exception as e:
            self.logger.warning(f"Could not load available prompts: {e}")
            self.available_prompts = set()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment."""

        if not os.path.exists(self.config_path):
            self.logger.debug(f"Config file not found at {self.config_path}. Using default values.")
            return {}
        else:
            self.logger.debug(f"Config file loaded from {self.config_path}.")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Return empty dict if config is None
        if config is None:
            return {}

        # Retrieve OpenAI API key from environment if not in config
        if "embeddings" in config and "openai" in config.get("embeddings", {}):
            if "openai_api_key" not in config["embeddings"]["openai"]:
                config["embeddings"]["openai"]["openai_api_key"] = os.environ.get(
                    "OPENAI_API_KEY"
                )
        if "embeddings" in config and "default_client" not in config["embeddings"]:
            config["embeddings"]["default_client"] = "openai"
              
        if "embeddings" in config and "default_model" not in config["embeddings"]:
            config["embeddings"]["default_model"] = "openai"

        if "embeddings" in config and "clients" not in config["embeddings"]:
            if "openai" in config["embeddings"] and "huggingface" in config["embeddings"]:
                config["embeddings"]["clients"] = {
                    "openai": config["embeddings"]["openai"], 
                    "huggingface": config["embeddings"]["huggingface"]
                }
        
        return config
    
    @property
    def embeddings_config(self) -> EmbeddingsConfig:
        """Get the embeddings configuration."""
        config = self._config.get("embeddings", {})
        return EmbeddingsConfig.model_validate(config)

    @property
    def cache_config(self) -> CacheConfig:
        """Get the cache configuration."""
        config = self._config.get("cache", {})
        return CacheConfig.model_validate(config)
    
    @property
    def monitor_config(self) -> MonitorConfig:
        """Get the monitor configuration."""
        config = self._config.get("monitor", {})
        return MonitorConfig.model_validate(config)
    
    @property
    def entity_extraction_config(self) -> EntityExtractionConfig:
        """Get the entity extraction configuration."""
        config = self._config.get("entity_extraction", {})
        return EntityExtractionConfig.model_validate(config)
    
    @property
    def llm_prompts_config(self) -> LLMPromptsConfig:
        """Get the LLM prompts configuration."""
        config = self._config.get("llm_prompts", {})
        return LLMPromptsConfig.model_validate(config)

    @property
    def ingestion_config(self) -> IngestionConfig:
        """Get the ingestion configuration."""
        config = self._config.get("ingestion", {})
        return IngestionConfig.model_validate(config)

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

        if loaders_config and hasattr(loaders_config, "file_mappings"):
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

        self.logger.info(f"Loaded {len(source_loaders)} source loaders")
        return source_loaders

    def get_default_prompt_templates(self) -> Dict[str, str]:
        """
        Get the default prompt templates with actual content from files.
        Maps keys from LLMPromptsConfig to their corresponding prompt templates.
        
        Returns:
            Dictionary with prompt templates using the config keys.
        """
        prompt_templates = {}
        
        # Get the mapping from our LLMPromptsConfig model
        prompt_mapping = self.llm_prompts_config.model_dump()
        
        # Iterate through each key-value pair in the prompt mapping
        for config_key, prompt_name in prompt_mapping.items():
            try:
                # Get the prompt template content directly using the filename from the config
                prompt_template = get_prompt(prompt_name)
                
                if prompt_template:
                    # Store using the config key
                    prompt_templates[config_key] = prompt_template
                    self.logger.debug(f"Loaded prompt template: {prompt_name} as {config_key}")
                else:
                    self.logger.warning(f"Prompt template {prompt_name} not found for {config_key}.")
            except Exception as e:
                self.logger.warning(f"Error loading prompt template {prompt_name} for {config_key}: {e}")

        return prompt_templates

    def print_graph_creation_pipeline(self, config: Dict[str, Any]) -> str:
        """
        Generates a formatted string describing the graph creation pipeline configuration.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            A formatted string describing the graph creation pipeline.
        """
        log_string = "Graph creation pipeline:\n"
        step_number = 1

        ee_methods = config.get('entity_extraction_methods', [])
        ee_chunking = config.get('chunking_strategy', 'none')

        coref_method = config.get('coreference_resolution_method', 'none')
        log_string += f"\t{step_number}. Coreference Resolution: method='{coref_method}'\n"
        step_number += 1

        log_string += f"\t{step_number}. Entity Extraction: chunking_strategy='{ee_chunking}', methods={ee_methods}"
        if "ner" in ee_methods:
            log_string += f', ner method/spacy model = {config.get("spacy_model", "en_core_web_sm")}'
        log_string += "\n"
        step_number += 1

        linking_enabled = config.get('entity_linking_enabled', False)
        linking_method = config.get('entity_linking_method', 'none')
        linking_info = f"enabled={linking_enabled}, method='{linking_method}'"
        log_string += f"\t{step_number}. Entity Linking: {linking_info}\n"
        step_number += 1

        relation_method = config.get('relationship_extraction_method', 'none')
        log_string += f"\t{step_number}. Relationship Extraction: method='{relation_method}'\n"
        step_number += 1

        gleaning_enabled = config.get('gleaning_enabled', False)
        max_gleaning = config.get('max_gleaning_steps', 'none') if gleaning_enabled else 'none'
        gleaning_info = f"enabled={gleaning_enabled}, max_steps={max_gleaning}"
        log_string += f"\t{step_number}. Gleaning: {gleaning_info}\n"
        step_number += 1

        postprocessing = config.get('postprocessing_steps', [])
        log_string += f"\t{step_number}. Postprocessing: steps={postprocessing if postprocessing else 'none'}\n"
        step_number += 1

        log_string += "\n"
        # Create a single formatted string for entity_types, 5 per line
        entity_types = config.get("entity_types", [])
        if entity_types:
            lines = ["Configured entity_types:"]
            for i in range(0, len(entity_types), 7):
                lines.append("  " + ", ".join(entity_types[i:i+5]))
            output_str = "\n\t".join(lines)
        else:
            output_str = "No entity_types configured."

        log_string += output_str + "\n"
        relationship_types = config.get("relationship_types", [])
        if relationship_types:
            lines = ["Configured relationship_types:"]
            for i in range(0, len(relationship_types), 7):
                lines.append("  " + ", ".join(relationship_types[i:i+5]))
            output_str = "\n\t".join(lines)
        else:
            output_str = "No relationship_types configured."

        log_string += output_str + "\n"
        return log_string

