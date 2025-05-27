"""
Vector store utilities for RAGdoll.

This module provides utilities for working with vector stores through LangChain's implementations.
It supports loading vector stores by name from config with automatic persistence management.
"""
import logging
from typing import Optional, List, Dict, Any, Union
import os

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS, Chroma, Pinecone

from ragdoll.config import ConfigManager
from ragdoll.embeddings import get_embedding_model

logger = logging.getLogger("ragdoll.vector_stores")

# Cache for vector stores to avoid recreation with same parameters
_vector_store_cache = {}

def get_vector_store(
    store_type: str = None, 
    embedding_model = None,
    documents: List[Document] = None,
    texts: List[str] = None,
    config_manager = None,
    config: Dict[str, Any] = None,
    persist_directory: str = None,
    collection_name: str = None,
    allow_dangerous_deserialization: bool = False,  # Add this parameter 
    **kwargs
) -> Any:
    """
    Factory function to get a vector store instance.
    
    Args:
        store_type: Type of vector store ('faiss', 'chroma', 'pinecone', etc.)
        embedding_model: Embedding model to use with the vector store
        documents: Optional list of documents to load into the vector store
        texts: Optional list of texts to load into the vector store
        config_manager: Optional ConfigManager instance
        config: Optional configuration dictionary
        persist_directory: Directory to persist vector store data
        collection_name: Name of the collection/index in the vector store
        allow_dangerous_deserialization: Whether to allow loading pickled data (for FAISS)
        **kwargs: Additional arguments for specific vector store types
        
    Returns:
        A configured vector store instance
    """
    # Initialize config
    vector_config = {}
    if config_manager is not None:
        vector_config = config_manager._config.get("vector_store", {})
    elif config is not None:
        if isinstance(config, dict):
            if "vector_store" in config:
                vector_config = config["vector_store"]
            else:
                vector_config = config
    
    # Determine store type (priority: parameter > config > default)
    actual_store_type = store_type or vector_config.get("store_type", "faiss")
    
    # Get embedding model if not provided
    if embedding_model is None:
        embedding_model = get_embedding_model(
            config_manager=config_manager,
            config=config
        )
    
    # Get persistence settings
    actual_persist_dir = persist_directory or vector_config.get("persist_directory", "./vector_store")
    actual_collection = collection_name or vector_config.get("collection_name", "default_collection")
    
    # Create vector store based on type
    if actual_store_type.lower() == "faiss":
        return _create_faiss_store(
            embedding_model=embedding_model,
            documents=documents,
            texts=texts,
            persist_directory=actual_persist_dir,
            allow_dangerous_deserialization=allow_dangerous_deserialization,  # Pass this parameter
            **kwargs
        )
    elif actual_store_type.lower() == "chroma":
        return _create_chroma_store(
            embedding_model=embedding_model,
            documents=documents,
            texts=texts,  # Pass texts parameter
            persist_directory=actual_persist_dir,
            collection_name=actual_collection,
            **kwargs
        )
    elif actual_store_type.lower() == "pinecone":
        return _create_pinecone_store(
            embedding_model=embedding_model,
            documents=documents,
            texts=texts,  # Pass texts parameter
            index_name=actual_collection,
            **kwargs
        )
    else:
        logger.warning(f"Unknown vector store type: {actual_store_type}, defaulting to FAISS")
        return _create_faiss_store(
            embedding_model=embedding_model,
            documents=documents,
            texts=texts,  # Pass texts parameter
            persist_directory=actual_persist_dir,
            **kwargs
        )

def _create_faiss_store(
    embedding_model: Embeddings,
    documents: Optional[List[Document]] = None,
    texts: Optional[List[str]] = None,
    persist_directory: Optional[str] = None,
    allow_dangerous_deserialization: bool = False,  # Add this parameter
    **kwargs
) -> FAISS:
    """Create a FAISS vector store."""
    try:
        # If persist directory is specified and exists, try to load
        if persist_directory and os.path.exists(os.path.join(persist_directory, "index.faiss")):
            logger.info(f"Loading existing FAISS index from {persist_directory}")
            return FAISS.load_local(
                persist_directory,
                embedding_model,
                allow_dangerous_deserialization=allow_dangerous_deserialization,  # Pass it here
                **kwargs
            )
        
        # Handle texts parameter if provided
        if texts:
            logger.info(f"Creating new FAISS index with {len(texts)} texts")
            vector_store = FAISS.from_texts(
                texts,
                embedding=embedding_model,
                **kwargs
            )
            
            # Persist if directory specified
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                vector_store.save_local(persist_directory)
                
            return vector_store
        
        # Otherwise create with documents
        elif documents:
            logger.info(f"Creating new FAISS index with {len(documents)} documents")
            vector_store = FAISS.from_documents(
                documents,
                embedding=embedding_model,
                **kwargs
            )
            
            # Persist if directory specified
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                vector_store.save_local(persist_directory)
                
            return vector_store
            
        else:
            # Create empty store with a placeholder
            logger.info("Creating empty FAISS index")
            return FAISS.from_texts(
                ["placeholder"],
                embedding=embedding_model,
                **kwargs
            )
    
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise

def _create_chroma_store(
    embedding_model: Embeddings,
    documents: Optional[List[Document]] = None,
    texts: Optional[List[str]] = None,  # Added texts parameter
    persist_directory: Optional[str] = None,
    collection_name: str = "default_collection",
    **kwargs
) -> Chroma:
    """Create a Chroma vector store."""
    try:
        # Check for chromadb first
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            logger.error("chromadb package not installed. Install with 'pip install chromadb'")
            raise ImportError("Could not import chromadb. Please install it with `pip install chromadb`.")
        
        # Create client settings
        chroma_settings = kwargs.pop("client_settings", None)
        
        # Create with texts if provided
        if texts:
            logger.info(f"Creating/updating Chroma collection '{collection_name}' with {len(texts)} texts")
            return Chroma.from_texts(
                texts,
                embedding_function=embedding_model,
                collection_name=collection_name,
                persist_directory=persist_directory,
                client_settings=chroma_settings,
                **kwargs
            )
        # Create with documents
        elif documents:
            logger.info(f"Creating/updating Chroma collection '{collection_name}' with {len(documents)} documents")
            return Chroma.from_documents(
                documents,
                embedding_function=embedding_model,
                collection_name=collection_name,
                persist_directory=persist_directory,
                client_settings=chroma_settings,
                **kwargs
            )
        else:
            logger.info(f"Loading Chroma collection '{collection_name}'")
            return Chroma(
                embedding_function=embedding_model,
                collection_name=collection_name,
                persist_directory=persist_directory,
                client_settings=chroma_settings,
                **kwargs
            )
    
    except Exception as e:
        logger.error(f"Error creating Chroma vector store: {e}")
        raise

def _create_pinecone_store(
    embedding_model: Embeddings,
    documents: Optional[List[Document]] = None,
    index_name: str = "ragdoll-index",
    **kwargs
) -> Any:
    """Create a Pinecone vector store."""
    try:
        import pinecone
        
        # Initialize Pinecone
        api_key = kwargs.pop("api_key", os.environ.get("PINECONE_API_KEY"))
        environment = kwargs.pop("environment", os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp"))
        
        if not api_key:
            raise ValueError("Pinecone API key is required. Provide as 'api_key' parameter or set PINECONE_API_KEY environment variable.")
        
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            # Get vector dimension from embedding model
            sample_embedding = embedding_model.embed_query("Sample text")
            dimension = len(sample_embedding)
            
            # Create index
            logger.info(f"Creating Pinecone index '{index_name}' with dimension {dimension}")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                pods=1,
                pod_type="p1.x1"
            )
        
        # Create Pinecone vector store
        from langchain.vectorstores import Pinecone as LangchainPinecone
        
        namespace = kwargs.pop("namespace", "")
        
        if documents:
            logger.info(f"Adding {len(documents)} documents to Pinecone index '{index_name}'")
            return LangchainPinecone.from_documents(
                documents,
                embedding_model,
                index_name=index_name,
                namespace=namespace,
                **kwargs
            )
        else:
            logger.info(f"Connecting to existing Pinecone index '{index_name}'")
            return LangchainPinecone(
                embedding_function=embedding_model,
                index_name=index_name,
                namespace=namespace,
                **kwargs
            )
    
    except ImportError:
        logger.error("Pinecone package not installed. Install it with 'pip install pinecone-client'")
        raise
    except Exception as e:
        logger.error(f"Error creating Pinecone vector store: {e}")
        raise

def add_documents(
    vector_store,
    documents: List[Document],
    **kwargs
) -> None:
    """Add documents to an existing vector store."""
    if hasattr(vector_store, "add_documents") and vector_store.add_documents is not None:
        vector_store.add_documents(documents, **kwargs)
    elif hasattr(vector_store, "from_documents") and vector_store.from_documents is not None:
        # Some stores don't have add_documents but do have from_documents
        # This is a bit of a hack but works for some implementations
        vector_store.from_documents(documents, **kwargs)
    else:
        raise ValueError(f"Vector store of type {type(vector_store)} doesn't support adding documents")

