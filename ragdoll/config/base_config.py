from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Type, Union, Literal
from typing import Optional, Dict, Any

class BaseConfig(BaseModel):    
    """Base configuration class that all configs should inherit from"""
    enabled: bool = Field(default=True, description="Whether this component is enabled")

class LoaderConfig(BaseConfig):
    """Configuration for document loaders"""
    loader_type: str = Field(..., description="Type of loader to use")
    recursive: bool = Field(default=False, description="Whether to recursively process directories")
    file_types: List[str] = Field(default=[], description="List of file extensions to process")

class ChunkerConfig(BaseConfig):
    """Configuration for text chunkers"""
    chunk_size: int = Field(default=1000, description="Size of each chunk in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    separator: str = Field(default="\n\n", description="Separator to use when splitting text")

class ClientConfig(BaseConfig):
    """Configuration for a specific embedding client."""
    client: str = Field(..., description="The name of the client (e.g., 'openai', 'huggingface').")
    model: str = Field(..., description="The name of the embedding model.")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional settings for the client.")

class EmbeddingsConfig(BaseConfig):
    """
    Configuration for embedding models.

    This class allows configuration for different embedding clients and models.

    Attributes:
        enabled (bool): Whether embeddings are enabled or not.
        default_client (str): The default client to use.
        clients (Dict[str, ClientConfig]): Configurations for each embedding client.
    """
    enabled: bool = Field(..., description="Whether to use embeddings or not.")
    default_client: str = Field(..., description="The default client to use.")
    clients: Dict[str, ClientConfig] = Field(..., description="Configurations for each embedding client.")

class MonitorConfig(BaseConfig):
    """Configuration for the monitor."""
    collect_metrics: bool = Field(
        default=True, description="Whether to collect metrics or not."
    )

class VectorStoreConfig(BaseConfig):
    """Configuration for vector stores"""
    store_type: str = Field(default="chroma", description="Type of vector store")

class LLMConfig(BaseConfig):
    """Configuration for language models"""
    model: str = Field(default="gpt-3.5-turbo", description="Name of the language model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=500, description="Maximum tokens for generation")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM service")

class CacheConfig(BaseConfig):
    """Configuration for the cache"""
    """
    Configuration for the cache.

    Attributes:
        cache_ttl (int): Time to live for cached items in seconds.
    """
    cache_ttl: int = Field(default=86400, description="Time to live for cached items in seconds.")

class LoadersConfig(BaseConfig):
    """Configuration for file extension to loader mappings"""
    file_mappings: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping of file extensions to loader class paths",
    )
class IngestionConfig(BaseConfig):
    """Configuration for ingestion service"""
    max_threads: int = Field(default=10, description="Maximum threads for concurrent processing")
    batch_size: int = Field(default=100, description="Number of documents to process in one batch")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed ingestion")
    retry_delay: int = Field(default=1, description="Delay between retries in seconds")
    retry_backoff: int = Field(default=2, description="Backoff multiplier for retries")
    loaders: LoadersConfig = Field(default_factory=LoadersConfig, description="Loaders configuration")
    

class LLMPromptsConfig(BaseModel):
    """Configuration for LLM prompts"""
    entity_extraction: str = Field(default="entity_extraction")
    extract_relationships: str = Field(default="relationship_extraction")
    coreference_resolution: str = Field(default="coreference_resolution")
    entity_relationship_gleaning: str = Field(default="entity_relationship_continue")
    entity_relationship_gleaning_continue: str = Field(default="entity_relationship_gleaning_continue")


class GraphDatabaseConfig(BaseModel):
    """Configuration for graph database output"""
    # Basic settings
    default_store: str = Field(default="json", description="Default graph store type to use")
    output_format: str = Field(default="json", description="Format for graph output (json, neo4j, networkx, memgraph)")
    output_file: Optional[str] = Field(default="graph_output.json", description="Output file path for graph data")
    input_file: Optional[str] = Field(default=None, description="Input file path to load graph from")
    
    # Neo4j specific settings
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="password", description="Neo4j password")
    
    # Memgraph specific settings
    host: str = Field(default="localhost", description="Memgraph server host")
    port: int = Field(default=7687, description="Memgraph server port")
    username: str = Field(default="", description="Memgraph username (if auth enabled)")
    password: str = Field(default="", description="Memgraph password (if auth enabled)")
    clear_database: bool = Field(default=False, description="Whether to clear the database before storing")
    clear_before_save: bool = Field(default=False, description="Whether to clear the database before each save")
    
    # Additional settings
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration options")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key, checking field values first then extra_config."""
        if key in self.__fields__ and key != "extra_config":
            return getattr(self, key, default)
        return self.extra_config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = self.dict(exclude={"extra_config"})
        result.update(self.extra_config)
        return result


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
    relationship_type_mapping: Dict[str, str] = {
        "works for": "WORKS_FOR",
        "is a": "IS_A",
        "is an": "IS_A",
        "located in": "LOCATED_IN",
        "located at": "LOCATED_IN",
        "born in": "BORN_IN",
        "lives in": "LOCATED_IN",
        "married to": "SPOUSE_OF",
        "spouse of": "SPOUSE_OF",
        "parent of": "PARENT_OF",
        "child of": "PARENT_OF",
        "works with": "AFFILIATED_WITH"
    }
    gleaning_enabled: bool = Field(default=True)
    max_gleaning_steps: int = Field(default=2)
    entity_linking_enabled: bool = Field(default=True)
    entity_linking_method: str = Field(default="string_similarity")
    entity_linking_threshold: float = Field(default=0.8)
    postprocessing_steps: List[str] = Field(default=["merge_similar_entities", "normalize_relations"])
    output_format: str = Field(default="json")
    graph_database_config: GraphDatabaseConfig = Field(default_factory=GraphDatabaseConfig)

