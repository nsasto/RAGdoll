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

- `search_tools.py`: Implements Google search + suggestion helpers built on LangChain tools.
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

- Inherits from LangChain `BaseTool`.
- Constructor signature:

```python
from langchain_google_community import GoogleSearchAPIWrapper
from ragdoll.tools.search_tools import SearchInternetTool

tool = SearchInternetTool(
    google_search=GoogleSearchAPIWrapper(),  # optional, auto-created when omitted
    log_level=logging.INFO,
)
```

- `_run(query: str, num_results: int = 3)` performs a Google Custom Search (via `langchain-google-community`) and returns dictionaries with `title`, `href`, and `snippet`. YouTube links are filtered out.
- `_arun` is intentionally unimplemented; these tools are synchronous today.

### `SuggestedSearchTermsTool`

- Generates related search queries using an LLM + prompt template.
- Constructor supports several dependency injection points:

```python
from ragdoll.tools.search_tools import SuggestedSearchTermsTool
from ragdoll.llms import get_llm_caller

tool = SuggestedSearchTermsTool(
    llm_caller=get_llm_caller(),       # preferred: supply a BaseLLMCaller
    prompt_key="search_queries",       # optional override; defaults to config template
    config_manager=my_config_manager,  # optional; falls back to settings.get_app()
    app_config=my_app_config,          # optional; used to read prompt templates
    log_level=logging.DEBUG,
)
```

- `_run(query: str, num_suggestions: int = 3)` formats the chosen prompt, calls the LLM synchronously via `call_llm_sync`, parses the result (JSON/list/lines), deduplicates, and returns up to `num_suggestions` strings.
- `_arun` is not implemented; wrap the tool manually if you need async behavior.

---

## Usage Example

```python
from ragdoll.tools.search_tools import SearchInternetTool, SuggestedSearchTermsTool
from ragdoll.llms import get_llm_caller

search_tool = SearchInternetTool()
print(search_tool._run("retrieval augmented generation", num_results=5))

suggest_tool = SuggestedSearchTermsTool(llm_caller=get_llm_caller())
print(suggest_tool._run("retrieval augmented generation", num_suggestions=4))
```

---

## Extending Tools

- Add new tools by subclassing LangChain `BaseTool` and implementing `_run`/`_arun`.
- Reuse the prompt templating helpers (`get_prompt`, `AppConfig.get_prompt_templates`) when your tool needs configurable wording.

---

## Best Practices

- Validate and sanitize search queries before hitting remote APIs.
- Handle API errors/rate limits gracefully and return structured, empty defaults when failures occur.
- Use the module loggers (already wired up) so downstream users can adjust verbosity per tool.

---

## Related Modules

- [LLM Integration](llm_integration.md)
- [Ingestion](ingestion.md)
