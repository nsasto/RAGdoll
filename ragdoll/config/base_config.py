from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    """Base configuration class that all configs should inherit from."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    enabled: bool = Field(default=True, description="Whether this component is enabled.")


class LoaderConfig(BaseConfig):
    """Configuration for document loaders."""

    loader_type: str = Field(..., description="Type of loader to use.")
    recursive: bool = Field(
        default=False, description="Whether to recursively process directories."
    )
    file_types: List[str] = Field(
        default_factory=list,
        description="List of file extensions to process.",
    )


class ChunkerConfig(BaseConfig):
    """Configuration for text chunkers."""

    chunk_size: int = Field(
        default=1000,
        description="Size of each chunk in characters.",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between adjacent chunks.",
    )
    separator: str = Field(
        default="\n\n",
        description="Separator to use when splitting text.",
    )
    default_splitter: str = Field(
        default="recursive",
        description="Default splitting strategy to use.",
    )
    chunking_strategy: str = Field(
        default="markdown",
        description="Chunking strategy such as 'none', 'recursive', 'markdown', etc.",
    )


class EmbeddingsConfig(BaseConfig):
    """
    Configuration for embedding models.

    RAGdoll supports multiple named embedding model configurations under ``models``.
    ``openai`` and ``huggingface`` keys are kept for backward compatibility.
    """

    default_model: Optional[str] = Field(
        default=None,
        description="Alias of the default embedding model to use.",
    )
    models: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Mapping of model aliases to provider-specific parameters.",
    )
    openai: Dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy OpenAI embedding configuration.",
    )
    huggingface: Dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy HuggingFace embedding configuration.",
    )


class MonitorConfig(BaseConfig):
    """Configuration for runtime monitoring."""

    collect_metrics: bool = Field(
        default=True, description="Whether runtime metrics should be captured."
    )


class VectorStoreConfig(BaseConfig):
    """Configuration for LangChain vector stores."""

    store_type: str = Field(
        default="chroma",
        description="Registry key or dotted path to a LangChain VectorStore.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments supplied to the vector store constructor.",
    )


class LLMConfig(BaseConfig):
    """Configuration for language models."""

    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Name of the language model to use.",
    )
    temperature: float = Field(
        default=0.7,
        description="Generation temperature passed to the model.",
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens the model may generate.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider.",
    )


class CacheConfig(BaseConfig):
    """Configuration for the cache."""

    cache_ttl: int = Field(
        default=86400,
        description="Time to live for cached items in seconds.",
    )


class LoadersConfig(BaseConfig):
    """Configuration for file extension to loader mappings."""

    file_mappings: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping of file extensions to loader class paths.",
    )


class IngestionConfig(BaseConfig):
    """Configuration for the ingestion service."""

    max_threads: int = Field(
        default=10,
        description="Maximum threads for concurrent processing.",
    )
    batch_size: int = Field(
        default=100,
        description="Number of documents to process in one batch.",
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed ingestion tasks.",
    )
    retry_delay: int = Field(
        default=1,
        description="Delay between retries in seconds.",
    )
    retry_backoff: int = Field(
        default=2,
        description="Backoff multiplier applied between retries.",
    )
    loaders: LoadersConfig = Field(
        default_factory=LoadersConfig,
        description="Loaders configuration.",
    )


class LLMPromptsConfig(BaseModel):
    """Configuration for default LLM prompt template names."""

    entity_extraction: str = Field(default="entity_extraction")
    extract_relationships: str = Field(default="relationship_extraction")
    coreference_resolution: str = Field(default="coreference_resolution")
    entity_relationship_gleaning: str = Field(default="entity_relationship_gleaning")
    entity_relationship_gleaning_continue: str = Field(
        default="entity_relationship_gleaning_continue"
    )


class GraphDatabaseConfig(BaseModel):
    """Configuration for graph database output."""

    model_config = ConfigDict(protected_namespaces=())
    default_store: str = Field(
        default="json",
        description="Default graph store type to use.",
    )
    output_format: str = Field(
        default="json",
        description="Format for graph output (json, neo4j, networkx, memgraph).",
    )
    output_file: Optional[str] = Field(
        default="graph_output.json",
        description="Output file path for graph data.",
    )
    input_file: Optional[str] = Field(
        default=None,
        description="Input file path to load graph data from.",
    )

    # Neo4j specific settings
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    user: str = Field(default="neo4j", description="Neo4j username.")
    password: str = Field(default="password", description="Neo4j password.")

    # Memgraph specific settings
    host: str = Field(default="localhost", description="Memgraph server host.")
    port: int = Field(default=7687, description="Memgraph server port.")
    memgraph_username: str = Field(
        default="",
        description="Memgraph username when authentication is enabled.",
    )
    memgraph_password: str = Field(
        default="",
        description="Memgraph password when authentication is enabled.",
    )
    clear_database: bool = Field(
        default=False, description="Clear the database before storing results."
    )
    clear_before_save: bool = Field(
        default=False,
        description="Whether to clear the database before each save.",
    )
    extra_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration options forwarded to the graph store.",
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key, checking field values first."""
        if key in self.model_fields and key != "extra_config":
            return getattr(self, key, default)
        return self.extra_config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = self.model_dump(exclude={"extra_config"})
        result.update(self.extra_config)
        return result


class EntityExtractionConfig(BaseModel):
    """Configuration for entity extraction and graph creation."""

    model_config = ConfigDict(protected_namespaces=())
    enabled: bool = Field(default=True)
    spacy_model: str = Field(default="en_core_web_sm")
    chunking_strategy: str = Field(default="default")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=50)
    splitter_type: Optional[str] = Field(default=None)
    coreference_resolution_method: str = Field(default="llm")
    entity_extraction_methods: List[str] = Field(default_factory=lambda: ["ner", "llm"])
    relationship_extraction_method: str = Field(default="llm")
    entity_types: List[str] = Field(
        default_factory=lambda: ["PERSON", "ORGANIZATION", "GPE", "DATE", "LOCATION"]
    )
    relationship_types: List[str] = Field(
        default_factory=lambda: ["HAS_ROLE", "WORKS_FOR"]
    )
    relationship_type_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
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
            "works with": "AFFILIATED_WITH",
        }
    )
    gleaning_enabled: bool = Field(default=True)
    max_gleaning_steps: int = Field(default=2)
    entity_linking_enabled: bool = Field(default=True)
    entity_linking_method: str = Field(default="string_similarity")
    entity_linking_threshold: float = Field(default=0.8)
    postprocessing_steps: List[str] = Field(
        default_factory=lambda: ["merge_similar_entities", "normalize_relations"]
    )
    output_format: str = Field(default="json")
    graph_database_config: GraphDatabaseConfig = Field(
        default_factory=GraphDatabaseConfig
    )
    relationship_parsing: Dict[str, Any] = Field(
        default_factory=lambda: {"preferred_format": "auto"},
        description="Parser configuration for relationship extraction outputs.",
    )
    relationship_prompts: Dict[str, Any] = Field(
        default_factory=lambda: {"default": "relationship_extraction", "providers": {}},
        description="Prompt template mappings for relationship extraction per provider.",
    )
    graph_retriever: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for optional graph retriever creation.",
    )
    llm_provider_hint: Optional[str] = Field(
        default=None,
        description="Optional override for the active LLM provider (e.g., 'openai').",
    )
