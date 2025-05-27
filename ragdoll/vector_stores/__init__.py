"""
Vector store utilities for RAGdoll.

This module provides utilities for working with vector stores through LangChain's implementations.
It supports loading vector stores by name from config with automatic persistence management.
"""
import os
import logging
from typing import Dict, Any, Optional, Union, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import config manager
from ragdoll.config import ConfigManager

logger = logging.getLogger("ragdoll.vector_stores")

def get_vector_store(
    store_type: str = None,
    embedding_model: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    texts: Optional[List[str]] = None,
    documents: Optional[List[Document]] = None,
    config_manager=None,
    allow_dangerous_deserialization: bool = False,  # Add this parameter for loading stores
    **kwargs
) -> Optional[Any]:
    """
    Get a LangChain vector store based on store type and configuration.
    
    Args:
        store_type: Type of vector store ('faiss', 'chroma', 'pinecone', etc.)
        embedding_model: Embedding model to use with the vector store
        persist_directory: Directory to persist the vector store (for disk-based stores)
        texts: Optional list of texts to add to the store upon creation
        documents: Optional list of Documents to add to the store upon creation
        config_manager: Optional ConfigManager instance
        allow_dangerous_deserialization: Whether to allow dangerous deserialization when loading stores
        **kwargs: Additional arguments to pass to the vector store constructor
        
    Returns:
        The instantiated vector store, or None if an error occurs
    """
    if config_manager is None:
        try:
            config_manager = ConfigManager()
        except ImportError:
            logger.warning("ConfigManager not found. Using provided parameters only.")
    
    # If config_manager exists, load vector store settings from config
    if config_manager:
        vector_config = config_manager._config.get("vector_stores", {})
        default_store = vector_config.get("default_store")
        
        # Use default store if no type specified
        if store_type is None and default_store:
            store_type = default_store
            logger.info(f"Using default vector store: {default_store}")
            
        # Get store-specific configuration
        store_configs = vector_config.get("stores", {})
        store_specific_config = store_configs.get(store_type, {})
        
        # Merge config with provided kwargs (kwargs take precedence)
        for key, value in store_specific_config.items():
            if key not in kwargs:
                kwargs[key] = value
    
    if store_type is None:
        logger.error("No vector store type specified and no default found in config.")
        return None
    
    try:
        # Import based on store type
        if store_type.lower() == 'faiss':
            from langchain_community.vectorstores import FAISS
            
            # If loading from disk
            if persist_directory and not (texts or documents):
                if os.path.exists(persist_directory):
                    logger.info(f"Loading FAISS index from {persist_directory}")
                    # Pass the parameter here
                    return FAISS.load_local(
                        persist_directory,
                        embeddings=embedding_model,
                        allow_dangerous_deserialization=allow_dangerous_deserialization,
                        **kwargs
                    )
                else:
                    logger.error(f"Directory {persist_directory} does not exist")
                    return None
            # Create from documents if provided
            if documents:
                return FAISS.from_documents(documents, embedding_model, **kwargs)
            # Create from texts if provided
            elif texts:
                return FAISS.from_texts(texts, embedding_model, **kwargs)
            # Load from disk if path provided and exists
            elif persist_directory and os.path.exists(os.path.join(persist_directory, "index.faiss")):
                return FAISS.load_local(persist_directory, embedding_model, **kwargs)
            # Otherwise create empty store with placeholder
            else:
                return FAISS.from_texts(["RAGdoll placeholder document"], embedding_model, **kwargs)
                
        elif store_type.lower() == 'chroma':
            from langchain_community.vectorstores import Chroma
            
            # Always send persist_directory if provided
            if persist_directory:
                kwargs['persist_directory'] = persist_directory
                
            if documents:
                return Chroma.from_documents(documents, embedding_model, **kwargs)
            elif texts:
                return Chroma.from_texts(texts, embedding_model, **kwargs)
            else:
                return Chroma(embedding_function=embedding_model, **kwargs)
                
        elif store_type.lower() == 'pinecone':
            try:
                from langchain_community.vectorstores import Pinecone
                import pinecone
            except ImportError:
                logger.error(
                    "Could not import Pinecone. Please install the package with:"
                    " `pip install pinecone-client`"
                )
                return None
            
            # Initialize Pinecone
            api_key = kwargs.pop('api_key', os.environ.get('PINECONE_API_KEY'))
            environment = kwargs.pop('environment', os.environ.get('PINECONE_ENVIRONMENT'))
            
            if not api_key or not environment:
                logger.error("Pinecone requires API key and environment")
                return None
                
            pinecone.init(api_key=api_key, environment=environment)
            
            index_name = kwargs.pop('index_name', None)
            if not index_name:
                logger.error("Pinecone requires an index_name")
                return None
                
            if documents:
                return Pinecone.from_documents(documents, embedding_model, index_name=index_name, **kwargs)
            elif texts:
                return Pinecone.from_texts(texts, embedding_model, index_name=index_name, **kwargs)
            else:
                return Pinecone(index_name=index_name, embedding=embedding_model, **kwargs)
        
        else:
            logger.error(f"Unsupported vector store type: {store_type}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize vector store '{store_type}': {e}")
        return None

