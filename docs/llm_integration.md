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

LLM modules provide interfaces to large language models for generation and augmentation. They enable prompt-based generation, retrieval-augmented generation, and more.

## Key Components

- `base_llm.py`: Abstract base for LLM integration.
- `__init__.py`: Module exports.

## Features

- Pluggable LLM backends (OpenAI, local models, etc.).
- Prompt management and augmentation.
- Supports RAG workflows.

---

## How It Works

1. **LLM Abstraction**: The base class defines the interface for LLMs.
2. **Prompting**: Prompts are sent to the LLM for generation.
3. **Singleton Pattern**: Shared LLM instances can be managed via class methods.

---

## Public API and Function Documentation

### `BaseLLM`

#### `call(prompt: str) -> str`

Abstract method to generate a response from a language model given a prompt.

#### `get_llm(*args, **kwargs) -> BaseLLM`

Class method to get or create a shared LLM instance.

---

## Usage Example

```python
from ragdoll.llms.base_llm import BaseLLM
# Implement a concrete LLM subclass, then:
llm = MyLLM.get_llm()
response = llm.call("What is retrieval-augmented generation?")
```

---

## Extending LLM Integration

- Subclass `BaseLLM` and implement the `call` method for new LLM backends.
- Add prompt management and augmentation as needed.

---

## Best Practices

- Use the singleton pattern for shared LLM resources.
- Validate prompt formatting for your LLM backend.
- Log and monitor LLM usage for cost and performance.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Chunking](chunking.md)
