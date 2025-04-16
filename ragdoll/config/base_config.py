from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Type, Union

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

class EmbeddingsConfig(BaseConfig):
    """Configuration for embedding models"""
    model_name: str = Field(default="text-embedding-ada-002", description="Name of embedding model to use")
    dimensions: int = Field(default=1536, description="Dimensions of the embedding vectors")
    api_key: Optional[str] = Field(default=None, description="API key for the embedding service")

class VectorStoreConfig(BaseConfig):
    """Configuration for vector stores"""
    store_type: str = Field(default="chroma", description="Type of vector store")
    path: str = Field(default="./vector_store", description="Path to store vector database")
    collection_name: str = Field(default="documents", description="Name of the collection")

class LLMConfig(BaseConfig):
    """Configuration for language models"""
    model_name: str = Field(default="gpt-3.5-turbo", description="Name of the language model")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=512, description="Maximum tokens in generated responses")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM service")