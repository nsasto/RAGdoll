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

- `base_config.py`: Base configuration logic and Pydantic models for all config types.
- `config_manager.py`: Loads and manages configs from YAML and environment.
- `default_config.yaml`: Default settings.

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

- `BaseConfig`: Base class for all configs, with an `enabled` flag.
- `LoaderConfig`, `ChunkerConfig`, `ClientConfig`, `EmbeddingsConfig`, `MonitorConfig`, `VectorStoreConfig`, `LLMConfig`, `CacheConfig`, `LoadersConfig`, `IngestionConfig`: Typed configs for each module, with detailed fields and descriptions.

### `ConfigManager`

#### Constructor

```python
ConfigManager(config_path: str = None)
```

Initializes the config manager with a path to the YAML config file (or uses default).

#### `embeddings_config`, `cache_config`, `monitor_config`, `ingestion_config`

Properties to access validated config objects for each module.

#### `get_loader_mapping() -> Dict[str, Type]`

Returns a mapping of file extensions to loader classes, loaded dynamically from config.

---

## Usage Example

```python
from ragdoll.config.config_manager import ConfigManager
config = ConfigManager("my_config.yaml")
embeddings_cfg = config.embeddings_config
print(embeddings_cfg.default_client)
```

---

## Extending Configuration

- Add new config models by subclassing `BaseConfig`.
- Add new config sections to YAML and update `ConfigManager` properties.

---

## Best Practices

- Use YAML for human-readable, version-controlled configs.
- Validate configs with Pydantic models.
- Use environment variables for secrets and overrides.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Embeddings](embeddings.md)
