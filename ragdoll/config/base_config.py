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



class VectorStoreConfig(BaseConfig):
    """Configuration for vector stores"""
    store_type: str = Field(default="chroma", description="Type of vector store")




class LLMConfig(BaseConfig):
    """Configuration for language models"""
    model_name: str = Field(default="gpt-3.5-turbo", description="Name of the language model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=500, description="Maximum tokens for generation")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM service")

class CacheConfig(BaseConfig):
    """Configuration for the cache"""
    ttl: int = Field(default=86400, description="Time to live for cached items")


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
    loaders: LoadersConfig = Field(default_factory=LoadersConfig, 
                                               description="Loaders configuration")