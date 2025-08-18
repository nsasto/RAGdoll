# Caching

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Caching](#extending-caching)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/cache/`

## Purpose

Caching improves performance by storing intermediate results (e.g., embeddings, retrievals) to avoid redundant computation and speed up pipelines.

## Key Components

- `cache_manager.py`: Manages cache operations.
- `__init__.py`: Module exports.

## Features

- Configurable cache strategies.
- Reduces redundant computation.
- Integrates with ingestion and retrieval.

---

## How It Works

1. **Initialization**: The cache manager is initialized with a cache directory and TTL (time-to-live).
2. **Caching**: Results from network or expensive operations are stored in cache (disk and memory).
3. **Retrieval**: Cached results are loaded if available and valid; otherwise, new results are computed and cached.

---

## Public API and Function Documentation

### `CacheManager`

#### Constructor

```python
CacheManager(cache_dir: str = None, ttl_seconds: int = 86400)
```

Initializes the cache manager with a directory and TTL.

#### `get_from_cache(source_type: str, identifier: str) -> Optional[List]`

Retrieve documents from cache for a given source type and identifier.

#### `clear_cache(source_type: Optional[str] = None, identifier: Optional[str] = None) -> int`

Clear cached data for a given source type and/or identifier. Returns the number of cache entries cleared.

---

## Usage Example

```python
from ragdoll.cache.cache_manager import CacheManager
cache = CacheManager()
docs = cache.get_from_cache("website", "https://arxiv.org/abs/1234.5678")
```

---

## Extending Caching

- Add new cache backends or strategies by extending `CacheManager`.
- Tune TTL and cache size for your workload.

---

## Best Practices

- Use caching for expensive or repeated operations.
- Monitor cache hit/miss rates for optimization.
- Periodically clear or refresh cache to avoid stale data.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Vector Stores](vector_stores.md)
