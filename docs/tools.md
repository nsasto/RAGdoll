# Tools

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Tools](#extending-tools)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/tools/`, `examples/tools.py`

## Purpose

Utility tools for search, data manipulation, and other helper functions. These tools support core RAGdoll workflows and can be used independently or as part of pipelines.

## Key Components

- `search_tools.py`: Implements search utilities.
- Example: `examples/tools.py` for demonstration.

## Features

- Search and filtering utilities.
- Helper functions for pipelines.

---

## How It Works

1. **Search Tools**: Provide internet and local search capabilities using APIs and LLMs.
2. **Helper Functions**: Support data manipulation and pipeline integration.

---

## Public API and Function Documentation

### `SearchInternetTool`

Tool for performing internet searches using Google. Returns a list of search results (title, URL, snippet).

#### Constructor

```python
SearchInternetTool(config: Config)
```

#### `_run(query: str, num_results: int = 3) -> List[Dict]`

Performs a Google search with the given query and returns results.

### `SuggestedSearchTermsTool`

Tool for generating suggested search terms using an LLM.

#### Constructor

```python
SuggestedSearchTermsTool(config: Config, llm: LLM)
```

#### `_run(query: str, num_suggestions: int = 3) -> List[str]`

Generates suggested search terms for a given query using an LLM.

---

## Usage Example

```python
from ragdoll.tools.search_tools import SearchInternetTool
tool = SearchInternetTool(config)
results = tool._run("retrieval augmented generation", num_results=5)
```

---

## Extending Tools

- Add new tools by subclassing `BaseTool` and implementing required methods.
- Integrate with new APIs or LLMs as needed.

---

## Best Practices

- Validate and sanitize search queries.
- Handle API errors and rate limits gracefully.
- Log tool usage for monitoring and debugging.

---

## Related Modules

- [LLM Integration](llm_integration.md)
- [Ingestion](ingestion.md)
