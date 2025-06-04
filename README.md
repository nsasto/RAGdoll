![Ragdoll](img/github-header-image.png)
 
[![Build](https://github.com/nsasto/RAGdoll/actions/workflows/python-publish.yml/badge.svg)](https://github.com/nsasto/RAGdoll/actions)
[![License](https://img.shields.io/github/license/nsasto/RAGdoll)](https://github.com/nsasto/RAGdoll/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/nsasto/RAGdoll)](https://github.com/nsasto/RAGdoll/commits)
 [![PyPI version](https://img.shields.io/pypi/v/python-ragdoll.svg)](https://pypi.org/project/python-ragdoll/) 

# RAGdoll: A Flexible and Extensible RAG Framework

Welcome to Ragdoll 2.0! This release marks a significant overhaul of the Ragdoll project, focusing on enhanced flexibility, extensibility, and maintainability. We've completely refactored the core architecture to make it easier than ever to adapt Ragdoll to your specific needs and integrate it with the broader LangChain ecosystem.

# 🧭 Project Overview 

RAGdoll 2 is an extensible framework for building Retrieval-Augmented Generation (RAG) applications. It provides a modular architecture that allows you to easily integrate various data sources, chunking strategies, embedding models, vector stores, large language models (LLMs), and graph stores. RAGdoll is designed to be flexible and fast, with minimal third-party dependencies. It accommodates a broad array of file types without any initial dependency on third-party hosted services using [langchain-markitdown](https://github.com/nsasto/langchain-markitdown). The loaders can easily be swapped out with any compatible LangChain loader when ready for production.

**Note:** RAGdoll 2 is a complete overhaul of the initial RAGdoll project and is not backwards compatible. It should be considered alpha software. RAGdoll v1.2 is the latest stable version available on PyPI.

## What's New

### Key Improvements in RAGdoll 2.0

- **Factory Pattern Architecture:** Implemented a factory pattern for all major components, making it easy to swap implementations or create custom ones.
- **LangChain Integration:** Core components now inherit from LangChain base classes, providing seamless interoperability with the LangChain ecosystem.
- **Pydantic Configuration:** Replaced custom configuration with a robust Pydantic-based system for better type safety and validation.
- **Graph Store Integration:** Added support for knowledge graphs alongside vector storage for more sophisticated retrieval.
- **Enhanced Entity Extraction:** Improved entity and relationship extraction from documents for better knowledge representation.
- **Unified Ingestion Pipeline:** Streamlined document processing with a comprehensive ingestion service.

### Enhanced Features in RAGdoll 2.0

This version of RAGdoll introduces several key features that improve the flexibility and usability of the framework:

- **Caching:** RAGdoll now supports caching, allowing you to store and reuse results from previous operations. This can significantly speed up the execution of your RAG applications by avoiding redundant computations.
    
- **Auto Loader Selection**: RAGdoll now includes loaders for multiple file types (not only pdf). The loader defaults to Langchain-Markitdown loaders, but can be configured to use any LangChain compatible loader. 
    
- **Monitoring:** A new monitoring capability has been added to RAGdoll. This allows you to track and understand the performance and behavior of your RAG applications over time.
    
```yaml
# Enable monitoring in config
monitor:
  enabled: true
```


## Quick Start Guide

Here's a quick example of how to get started with RAGdoll:
```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.config import Config

# Create a new configuration
config = Config(vector_store_path="./my_vector_db", chunk_size=500)

# Create a Ragdoll instance
ragdoll = Ragdoll(config=config)

# Load and process documents
ragdoll.add_documents("path/to/documents")

# Run a query
result = ragdoll.query("What is the capital of France?")

# Print the result
print(result)
```
For more advanced usage, you can customize each component:

```python
from ragdoll.ragdoll import Ragdoll
from ragdoll.config import Config
from ragdoll.embeddings.openai_embeddings import MyOpenAIEmbeddings
from ragdoll.chunkers.recursive_character_text_splitter import MyRecursiveCharacterTextSplitter

# Create custom components
embeddings = MyOpenAIEmbeddings(model="text-embedding-3-large")
chunker = MyRecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Create Ragdoll with custom components
ragdoll = Ragdoll(embeddings=embeddings, chunker=chunker)
```
### Factory Pattern Usage ###
RAGdoll 2.0 uses a factory pattern to create and configure components:

```python
from ragdoll.config import Config
from ragdoll.factories import get_embeddings_model, get_text_splitter, get_vector_store, get_graph_store

# Create configuration
config = Config(
    embeddings_model="openai",
    chunker_type="recursive",
    vector_store_type="chroma"
)

# Get components using factories
embeddings = get_embeddings_model(config)
chunker = get_text_splitter(config)
vector_store = get_vector_store(config)
graph_store = get_graph_store(config)

```

## Installation

To install RAGdoll, follow these steps:

### Stable version install

`pip install python-ragdoll`

### Latest version install

1.  **Clone the Repository:**
```bash
git clone https://github.com/nsasto/RAGdoll.git
cd RAGdoll
```
2.  **Install Dependencies:**
```bash
    pip install -e .
```

This will install the required dependencies, including Langchain and Pydantic.
3. **Install extra dependencies**: if you need some specific models or libraries, install them here as well.
4. **Set up Environment Variables**:Create a .env file in your project root with required API keys:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
```

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
*   **`ingestion`:** Handles the complete document processing pipeline.

### Abstract Base Classes

Each module has an abstract base class (`BaseLoader`, `BaseChunker`, `BaseEmbeddings`, `BaseVectorStore`, `BaseLLM`, `BaseGraphStore`, `BaseChain`) that defines a standard interface for that component type.

### Default Implementations

RAGdoll provides default implementations for most components, allowing you to quickly get started without having to write everything from scratch:

*   **`Lanchain-Markitdown`:** A default loader for most major file types.
*   **`RecursiveCharacterTextSplitter`:** A default text splitter.
*   **`OpenAIEmbeddings`:** Default embeddings that use OpenAI's API.
*   **`MyChroma`:** A default Chroma vector store.
*   **`OpenAILLM`**: A default OpenAI LLM.
*   **`BaseGraphStore`**: A BaseGraphStore, it needs to be implemented.

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
from ragdoll.ragdoll import Ragdoll
from ragdoll.config import Config

# Create a new configuration
config = Config(
    vector_store_path="./my_vectors",
    chunk_size=500,
    chunk_overlap=50,
    embeddings_model="openai",
    llm_model="gpt-4o",
    monitoring_enabled=True
)

# Create Ragdoll with this configuration
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