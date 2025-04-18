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

class LLMConfig(BaseConfig):
    """Configuration for language models"""
    model_name: str = Field(default="gpt-3.5-turbo", description="Name of the language model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=500, description="Maximum tokens for generation")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM service")

class LoaderMappingConfig(BaseConfig):
    """Configuration for file extension to loader mappings"""
    mappings: Dict[str, str] = Field(
        default={
            ".json": "langchain_community.document_loaders.JSONLoader",
            ".jsonl": "langchain_community.document_loaders.JSONLoader",
            ".yaml": "langchain_community.document_loaders.JSONLoader",
            ".csv": "langchain_community.document_loaders.CSVLoader",
            ".txt": "langchain_community.document_loaders.TextLoader",
            ".md": "langchain_community.document_loaders.TextLoader",
            ".pdf": "langchain_community.document_loaders.PyMuPDFLoader",
        },
        description="Mapping of file extensions to loader class paths"
    )

class IngestionConfig(BaseConfig):
    """Configuration for ingestion service"""
    max_threads: int = Field(default=10, description="Maximum threads for concurrent processing")
    batch_size: int = Field(default=100, description="Number of documents to process in one batch")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed ingestion")
    retry_delay: int = Field(default=1, description="Delay between retries in seconds")
    retry_backoff: int = Field(default=2, description="Backoff multiplier for retries")
    loader_mappings: LoaderMappingConfig = Field(default_factory=LoaderMappingConfig, 
                                               description="Loader mappings configuration")