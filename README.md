![Ragdoll](img/github-header-image.png)

# RAGdoll: A Flexible and Extensible RAG Framework

Welcome to Ragdoll 2.0! This release marks a significant overhaul of the Ragdoll project, focusing on enhanced flexibility, extensibility, and maintainability. We've completely refactored the core architecture to make it easier than ever to adapt Ragdoll to your specific needs and integrate it with the broader LangChain ecosystem. This document outlines the major changes and improvements you'll find in this exciting new version.


# 🧭 Project Overview 

RAGdoll 2 is an extensible framework for building Retrieval-Augmented Generation (RAG) applications. It provides a modular architecture that allows you to easily integrate various data sources, chunking strategies, embedding models, vector stores, large language models (LLMs), and graph stores. RAGdoll is designed to be flexible and fast, without any third party dependencies. It's also designed to accomodate a broad array of file types without any initial dependency on third party hosted services using [langchain-markitdown](https://github.com/nsasto/langchain-markitdown). The loaders can easily be swapped out with any compatible lanchain loader when ready for production.

Note that RAGdoll 2 is a complete overhaul of the initial RAGdoll project and is not backwards compatible in any respect. 

## Quick Start Guide

Here's a quick example of how to get started with RAGdoll:
```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.config import Config
# Create a new configuration
config = Config(vector_store_path="./my_vector_db", chunk_size=500)
# Create a Ragdoll instance
ragdoll = Ragdoll(config=config)

# Run a prompt
result = ragdoll.run("What is the capital of France?")

# Print the result
print(result)
```
## Installation

To install RAGdoll, follow these steps:

1.  **Clone the Repository:**
```
bash
    git clone <repository_url>
    cd RAGdoll
```
2.  **Install Dependencies:**
```
bash
    pip install -e .
```
This will install the required dependencies, including Langchain and Pydantic.
3. **Install extra dependencies**: if you need some specific models or libraries, install them here as well.

## Architecture

RAGdoll's architecture is built around modular components and abstract base classes, making it highly extensible. Here's an overview of the key modules:

### Modules

*   **`loaders`:** Responsible for loading data from various sources (e.g., directories, JSON files, web pages). 
*   **`chunkers`:** Handles the splitting of large text documents into smaller chunks.
*   **`embeddings`:** Provides an interface for embedding models, allowing you to generate vector representations of text.
*   **`vector_stores`:** Manages the storage and retrieval of vector embeddings.
*   **`llms`:** Provides an interface to interact with different large language models.
*   **`graph_stores`:** Manages the storage and querying of knowledge graphs.
*   **`chains`:** Defines different types of chains, like retrieval QA.

### Abstract Base Classes

Each module has an abstract base class (`BaseLoader`, `BaseChunker`, `BaseEmbeddings`, `BaseVectorStore`, `BaseLLM`, `BaseGraphStore`, `BaseChain`) that defines a standard interface for that component type.

### Default Implementations

RAGdoll provides default implementations for most components, allowing you to quickly get started without having to write everything from scratch:

*   **`Lanchain-Markitdown`:** A default loader for most major file types.
*   **`RecursiveCharacterTextSplitter`:** A default text splitter.
*   **`OpenAIEmbeddings`:** Default embeddings that use OpenAI's API.
*   **`MyChroma`:** A default Chroma vector store.
*   **`OpenAILLM`**: A default OpenAI LLM.
* **`BaseGraphStore`**: A BaseGraphStore, it needs to be implemented.

## Extensibility

RAGdoll is designed to be highly extensible. You can easily create custom components by following these steps:

1.  **Subclass the Base Class:** Create a new class that inherits from the relevant base class (e.g., `BaseLoader`, `BaseEmbeddings`).
2.  **Implement Abstract Methods:** Implement the abstract methods defined in the base class to provide your custom functionality.
3.  **Integrate into RAGdoll:** Pass an instance of your custom component to the `Ragdoll` class when you create it.

## Configuration

RAGdoll uses Pydantic to manage its configuration. This allows for:

*   **Data Validation:** Automatic validation of configuration values.
*   **Type Hints:** Clear type definitions for configuration settings.
*   **Default Values:** Convenient default values for configuration options.

You can create a `Config` object and pass it to the `Ragdoll` class.
```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.config import Config
# Create a new configuration
config = Config(vector_store_path="./my_vector_db", chunk_size=500)
# Create a Ragdoll instance
ragdoll = Ragdoll(config=config)
```
## Contributing

Contributions to RAGdoll are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Make your changes and write tests.
4.  Submit a pull request.

## License

RAGdoll is licensed under the [MIT License](LICENSE).