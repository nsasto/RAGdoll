# LLM Integration

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending LLM Integration](#extending-llm-integration)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/llms/`

## Purpose

LLM utilities provide a thin, configurable bridge between Ragdoll components and LangChain chat models. They centralize model initialization, API-key resolution, and expose a lightweight caller protocol that downstream services can depend on without importing LangChain directly.

## Key Components

- `__init__.py`: High-level helpers such as `get_llm` and `get_llm_caller` that know how to read config and instantiate LangChain chat models.
- `callers.py`: Defines the `BaseLLMCaller` protocol, a `LangChainLLMCaller` adapter, and a `call_llm_sync` helper so synchronous call sites can reuse async callers.

## Features

- Centralized config loading (API keys, provider-specific params) through `ConfigManager`.
- Optional lazy wrapping of LangChain `BaseChatModel` / `BaseLanguageModel`.
- Simple async protocol (`BaseLLMCaller`) that can be mocked in unit tests or swapped for non-LangChain backends.

---

## How It Works

1. **Model Resolution**: `get_llm()` reads `config.llms` (or an explicit dict) to instantiate a LangChain chat model with the correct provider SDK.
2. **Caller Wrapping**: `get_llm_caller()` takes either a model name/config or an already-constructed model and returns a `BaseLLMCaller` (typically a `LangChainLLMCaller`).
3. **Consumption**: High-level services (Ragdoll orchestrator, ingestion pipeline, tools) accept a `BaseLLMCaller`, allowing tests to pass in lightweight fakes while production code reuses the shared helper.

---

## Public API and Function Documentation

### `get_llm(model_name_or_config=None, config_manager=None)`

Return a LangChain chat/language model using the configured provider defaults. Accepts either a model name (looked up in config) or a dict of parameters.

### `get_llm_caller(model_name_or_config=None, config_manager=None, llm=None)`

Return a `BaseLLMCaller`. When `llm` is provided, it is wrapped directly; otherwise `get_llm` is invoked under the hood. Components should prefer passing this caller around instead of the raw LangChain object.

### `BaseLLMCaller`

Protocol with a single `async call(prompt: str) -> str` method. `LangChainLLMCaller` is the default implementation, but tests can implement their own fakes without touching LangChain.

### `call_llm_sync(llm_caller, prompt)`

Utility for synchronous contexts that still want to use the async caller protocol. It transparently spins up an event loop if needed.

---

## Usage Example

```python
from ragdoll.llms import get_llm_caller
from ragdoll.ragdoll import Ragdoll

# Resolve a caller using whatever model is configured as default.
llm_caller = get_llm_caller()

# Inject into the public orchestrator (works for pipelines/tools too).
app = Ragdoll(llm_caller=llm_caller)
result = app.query("What is retrieval-augmented generation?")
print(result["answer"])
```

---

## Extending LLM Integration

- Implement your own `BaseLLMCaller` (e.g., to call a local model server) and pass it into Ragdoll, the ingestion pipeline, or individual tools.
- Wrap third-party clients in an adapter that satisfies the protocol—no need to depend on LangChain if you don't want to.

---

## Best Practices

- Prefer `get_llm_caller()` over `get_llm()` in higher-level modules to keep dependencies lightweight.
- Inject callers via constructors (e.g., `Ragdoll(llm_caller=...)`, `IngestionOptions(llm_caller=...)`) so unit tests can provide fakes.
- Use `call_llm_sync` when you must invoke a caller from synchronous code; call the async `llm_caller.call()` directly when already in an async context.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Chunking](chunking.md)
