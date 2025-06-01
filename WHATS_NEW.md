# What's New in Ragdoll 2.0

## Introduction

Welcome to Ragdoll 2.0! This release marks a significant overhaul of the Ragdoll project, focusing on enhanced flexibility, extensibility, and maintainability. We've completely refactored the core architecture to make it easier than ever to adapt Ragdoll to your specific needs and integrate it with the broader LangChain ecosystem. This document outlines the major changes and improvements you'll find in this exciting new version.

## Abstraction: The Core of Extensibility

The central theme of Ragdoll 2.0 is **abstraction**. We've introduced a series of abstract base classes (ABCs) that define clear contracts for various components within the system. This allows you to easily swap out different implementations or create entirely new ones without modifying the core Ragdoll logic.

Here's a breakdown of the key abstractions:

*   **`BaseLoader`:**
    *   **Purpose:** Defines the contract for loading data from various sources (e.g., directories, JSON files, web pages).
    *   **Key Method:** `load()` - Returns a list of LangChain `Document` objects.
    *   **Benefits:** Enables users to easily integrate custom data sources by implementing this interface.
*   **`BaseChunker`:**
    *   **Purpose:** Defines the contract for splitting text into chunks.
    *   **Key Method:** `split_text(text)` - Returns a list of text chunks.
    *   **Benefits:** Allows users to implement custom chunking algorithms (e.g., by paragraph, sentence, or token).
*   **`BaseEmbeddings`:**
    *   **Purpose:** Defines the contract for embedding models.
    *   **Key Methods:** `embed_documents(texts)`, `embed_query(text)`.
    *   **Benefits:** Makes it easy to integrate different embedding providers (e.g., OpenAI, Hugging Face, custom models).
*   **`BaseVectorStore`:**
    *   **Purpose:** Defines the contract for vector databases.
    *   **Key Methods:** `add_documents()`, `similarity_search()`, `from_documents()`.
    *   **Benefits:** Enables users to use different vector storage solutions (e.g., Chroma, FAISS, Pinecone).
*   **`BaseLLM`:**
    *   **Purpose:** Defines the contract for language models.
    *   **Key Method:** `_call(prompt, stop)` - Processes a prompt and returns the model's response.
    *   **Benefits:** Allows users to switch between different LLMs (e.g., OpenAI, Hugging Face models) with ease.
*   **`BaseChain`:**
    *   **Purpose:** Defines the contract for chains.
    * **Key Method**: `run(prompt)`- Process a prompt.
    * **Benefits**: allows the creation of custom chains.
*   **`BaseGraphStore`**:
    *   **Purpose:** defines the contract for the graph database.
    * **Key Methods**: `add_node(node)`, `add_edge(node1, node2, edge)`, `query_graph(query)`.
    * **Benefits**: allows to use different graph databases.

## LangChain Compatibility

Ragdoll 2.0 now embraces the LangChain ecosystem by directly inheriting from LangChain's base classes for several components:

*   **`BaseLoader`:** Your `BaseLoader` now inherits from LangChain's `BaseLoader`.
*   **`BaseEmbeddings`:** Inherits from LangChain's `Embeddings`.
*   **`BaseLLM`**: Inherits from Langchain's `LLM`.
*   **`BaseVectorStore`:** Inherits from LangChain's `VectorStore`.

**Benefits of LangChain Compatibility:**

*   **Seamless Integration:** Ragdoll components can now be easily swapped with LangChain's implementations.
*   **Consistency:** Your code is more consistent with the LangChain framework.
*   **Future-Proofing:** You benefit from updates and improvements made to LangChain's base classes.

## Pydantic Configuration

We've replaced our previous configuration management system with **Pydantic**. This change brings significant advantages:

*   **Data Validation:** Pydantic automatically validates the types and constraints of your configuration settings.
*   **Type Safety:** Type hints make your configuration more readable and help catch errors.
*   **Data Parsing:** Load configuration from dictionaries or JSON strings.
*   **Default Values:** Easily define default settings.
* **Extensibility**: You can extend the configuration class and add more logic.

**How to Use the New Configuration:**

1.  Define your configuration parameters in the `ragdoll/config.py` file with the class `Config`.
2. You can now access the configuration values as attributes.
3. If not provided, default values will be used.

## New Graph Store

Ragdoll 2.0 introduces a new **`BaseGraphStore`** abstraction, allowing you to:

*   **Add Nodes:** `add_node(node)`
*   **Add Edges:** `add_edge(node1, node2, edge)`
*   **Query the Graph:** `query_graph(query)`

**Benefits of the Graph Store:**

*   **Knowledge Representation:** You can now represent complex relationships between entities.
* **Extensibility**: allows to use any graph database.

## Conclusion

Ragdoll 2.0 is a major step forward for the project. The new abstractions, LangChain compatibility, Pydantic configuration, and the graph store provide a significantly more powerful and flexible foundation for building RAG-based applications. We're excited to see what you create with it!