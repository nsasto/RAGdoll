# Configuration

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Configuration](#extending-configuration)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/config/`

## Purpose

Configuration modules manage settings for all components, enabling reproducible, flexible, and environment-specific control of the RAGdoll system.

## Key Components

- `base_config.py`: Base configuration logic and Pydantic models (e.g., `EmbeddingsConfig`, `EntityExtractionConfig`, `GraphDatabaseConfig`).
- `config_manager.py`: Loads YAML, resolves environment references, and exposes typed accessors.
- `default_config.yaml`: Shipping defaults that can be copied/overridden.

## Features

- YAML-based configuration.
- Environment-specific overrides.
- Centralized management for all modules.

---

## How It Works

1. **Config Models**: All configs inherit from `BaseConfig` (Pydantic), providing validation and documentation.
2. **Config Loading**: `ConfigManager` loads YAML config and merges with environment variables.
3. **Component Access**: Each module accesses its config via the manager's properties.

---

## Public API and Function Documentation

### `BaseConfig` and Subclasses

- `BaseConfig`: Base class for feature toggles (`enabled`) and shared validation.
- `LoaderConfig`, `ChunkerConfig`, `EmbeddingsConfig`, `MonitorConfig`, `VectorStoreConfig`, `LLMConfig`, `CacheConfig`, `LoadersConfig`, `IngestionConfig`, `EntityExtractionConfig`, `GraphDatabaseConfig`, `LLMPromptsConfig`: Typed configs for each RAGdoll subsystem with explicit field descriptions.

### `ConfigManager`

#### Constructor

```python
ConfigManager(config_path: str = None)
```

Initializes the config manager with a path to the YAML config file (or uses default).

#### Typed Properties

- `embeddings_config`, `cache_config`, `monitor_config`, `ingestion_config`, `entity_extraction_config`, `vector_store_config`, `graph_database_config`, `llm_prompts_config`: properties exposing the matching Pydantic object.

#### `get_loader_mapping() -> Dict[str, Type | str]`

Returns a mapping of file extensions to loader classes or import strings. Loader modules are imported lazily by the ingestion service when each extension is encountered.

---

## Usage Example

```python
from ragdoll.config.config_manager import ConfigManager

config = ConfigManager("my_config.yaml")
embeddings_cfg = config.embeddings_config
print(embeddings_cfg.default_model)

entity_cfg = config.entity_extraction_config
print(entity_cfg.graph_database_config.output_file)
```

---

## Extending Configuration

- Add new config models by subclassing `BaseConfig`.
- Add new config sections to YAML and update `ConfigManager` properties.

### Entity Extraction Options

Recent refactors introduced richer controls for graph/entity workflows:

- `entity_extraction.relationship_parsing`: configure how LLM relationship output is parsed. Keys include `preferred_format` (`json`, `markdown`, `auto`), `parser_class` (dotted path to a custom parser), `schema` (custom Pydantic model), and `parser_kwargs`.
- `entity_extraction.relationship_prompts`: map provider hints to prompt template names. The service picks `providers[provider_name]` when the LLM caller advertises its provider, otherwise it falls back to `default`.
- `entity_extraction.graph_retriever`: enable optional retriever creation (`enabled: true`) and choose a backend (`simple` or `neo4j`). Additional keys like `top_k` or `include_edges` are forwarded to the retriever factory, and Neo4j credentials are taken from `graph_database_config`.
- `entity_extraction.llm_provider_hint`: manually specify the provider string if the chosen LLM caller cannot be auto-detected. This drives both prompt selection and downstream logging.

When these sections are present in `default_config.yaml`, `ConfigManager` merges them into `entity_extraction_config`, so `EntityExtractionService` and `IngestionPipeline` pick up the new behavior automatically.

---

## Best Practices

- Use YAML for human-readable, version-controlled configs.
- Validate configs with Pydantic models.
- Use environment variables (referenced via `os.environ/KEY` or `#KEY`) for secrets and overrides; `ConfigManager` resolves them through `ragdoll.utils.env.resolve_env_reference`.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Embeddings](embeddings.md)
