"""
Embeddings module for RAGdoll.

This module provides utilities for working with embedding models through LangChain's implementations.
It uses a factory pattern to create embedding models based on configuration.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List

from langchain_core.embeddings import Embeddings
from ragdoll import settings
from ragdoll.app_config import AppConfig

# Import embedding models for direct access in tests
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.fake import FakeEmbeddings

logger = logging.getLogger("ragdoll.embeddings")


def get_embedding_model(
    model_name: str = None,
    config_manager=None,
    provider=None,
    app_config: Optional[AppConfig] = None,
    **kwargs,
) -> Optional[Embeddings]:
    """
    Get an embedding model based on configuration.

    Args:
        model_name: Name of the embedding model to use
        config_manager: Optional ConfigManager instance
        provider: Direct provider specification, overrides model settings
        **kwargs: Additional arguments to pass to the embedding model constructor

    Returns:
        An initialized embedding model, or None if an error occurs
    """
    if app_config is not None:
        config_manager = app_config.config
    elif config_manager is None:
        config_manager = settings.get_app().config

    # If provider is directly specified, skip config and use direct initialization
    if provider:
        return _initialize_model_by_provider(provider, kwargs)

    # If config_manager exists, load embedding settings from config
    if config_manager:
        embedding_config = config_manager._config.get("embeddings", {})
        default_model = embedding_config.get("default_model")

        # Use default model if no name specified
        if model_name is None and default_model:
            model_name = default_model
            logger.info(f"Using default embedding model: {default_model}")

        # Handle both configuration formats
        models_config = embedding_config.get("models", {})
        # For backward compatibility
        openai_config = embedding_config.get("openai", {})
        huggingface_config = embedding_config.get("huggingface", {})

        # If model_name matches an entry in models, use that config
        if model_name in models_config:
            model_specific_config = models_config[model_name].copy()

            # Extract provider
            provider = model_specific_config.pop(
                "provider", kwargs.pop("provider", None)
            )

            # Merge with provided kwargs (kwargs take precedence)
            for key, value in kwargs.items():
                model_specific_config[key] = value

            # If provider is specified, use it to initialize the model
            if provider:
                return _initialize_model_by_provider(provider, model_specific_config)

        # Backward compatibility with direct openai/huggingface keys
        elif model_name == "openai" or (model_name is None and openai_config):
            # Use openai config
            model_config = openai_config.copy()

            # Merge with provided kwargs
            for key, value in kwargs.items():
                model_config[key] = value

            return _create_openai_embeddings(model_config)

        elif model_name == "huggingface" or (
            model_name is None and huggingface_config and not openai_config
        ):
            # Use huggingface config
            model_config = huggingface_config.copy()

            # Merge with provided kwargs
            for key, value in kwargs.items():
                model_config[key] = value

            return _create_huggingface_embeddings(model_config)

    # If we get here with a model_name but no provider, try to infer provider
    if model_name and not provider:
        if "openai" in model_name.lower() or model_name.startswith("text-embedding"):
            return _create_openai_embeddings(kwargs)
        elif "huggingface" in model_name.lower() or "/" in model_name:
            kwargs["model_name"] = model_name
            return _create_huggingface_embeddings(kwargs)

    logger.error(f"Could not determine provider for model {model_name}")
    return None


def _initialize_model_by_provider(
    provider: str, config: Dict[str, Any]
) -> Optional[Embeddings]:
    """
    Initialize an embedding model based on provider.

    Args:
        provider: Provider name (openai, huggingface, etc.)
        config: Configuration for the model

    Returns:
        Initialized embedding model, or None if provider not supported
    """
    try:
        if provider.lower() == "openai":
            return _create_openai_embeddings(config)
        elif provider.lower() == "huggingface":
            return _create_huggingface_embeddings(config)
        elif provider.lower() == "google":
            return _create_google_embeddings(config)
        elif provider.lower() == "cohere":
            return _create_cohere_embeddings(config)
        elif provider.lower() == "fake" or provider.lower() == "mock":
            return _create_fake_embeddings(config)
        else:
            logger.error(f"Unsupported embedding provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"Error initializing embeddings with provider {provider}: {e}")
        return None


def _create_openai_embeddings(model_params: Dict[str, Any]) -> Optional[Embeddings]:
    """Create OpenAI embeddings model."""
    try:
        # Extract model-specific params
        model = model_params.pop("model", "text-embedding-3-large")
        dimensions = model_params.pop("dimensions", None)
        api_key = model_params.pop("api_key", os.environ.get("OPENAI_API_KEY"))

        # Handle environment variable references
        if api_key and isinstance(api_key, str) and api_key.startswith("#"):
            env_var = api_key[1:]
            api_key = os.environ.get(env_var)

        # Create embeddings model
        if dimensions is not None:
            return OpenAIEmbeddings(
                model=model,
                dimensions=dimensions,
                openai_api_key=api_key,
                **model_params,
            )
        else:
            return OpenAIEmbeddings(model=model, openai_api_key=api_key, **model_params)
    except Exception as e:
        logger.error(f"Failed to create OpenAI embeddings: {e}")
        return None


def _create_huggingface_embeddings(
    model_params: Dict[str, Any],
) -> Optional[Embeddings]:
    """Create HuggingFace embeddings model."""
    try:
        # Extract model-specific params
        model_name = model_params.pop(
            "model_name", "sentence-transformers/all-mpnet-base-v2"
        )

        # Create embeddings model
        return HuggingFaceEmbeddings(model_name=model_name, **model_params)
    except Exception as e:
        logger.error(f"Failed to create HuggingFace embeddings: {e}")
        return None


def _create_google_embeddings(model_params: Dict[str, Any]) -> Optional[Embeddings]:
    """Create Google Vertex AI embeddings model."""
    try:
        from langchain_google_vertexai import VertexAIEmbeddings

        # Create embeddings model
        return VertexAIEmbeddings(**model_params)
    except ImportError:
        logger.error(
            "Failed to import VertexAIEmbeddings. Install with: pip install langchain-google-vertexai"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to create Google Vertex AI embeddings: {e}")
        return None


def _create_cohere_embeddings(model_params: Dict[str, Any]) -> Optional[Embeddings]:
    """Create Cohere embeddings model."""
    try:
        from langchain_cohere import CohereEmbeddings

        # Extract API key
        api_key = model_params.pop("api_key", os.environ.get("COHERE_API_KEY"))

        # Handle environment variable references
        if api_key and isinstance(api_key, str) and api_key.startswith("#"):
            env_var = api_key[1:]
            api_key = os.environ.get(env_var)

        # Create embeddings model
        return CohereEmbeddings(cohere_api_key=api_key, **model_params)
    except ImportError:
        logger.error(
            "Failed to import CohereEmbeddings. Install with: pip install langchain-cohere"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to create Cohere embeddings: {e}")
        return None


def _create_fake_embeddings(model_params: Dict[str, Any]) -> FakeEmbeddings:
    """Create fake embeddings for testing."""
    size = model_params.get("size", 1536)
    return FakeEmbeddings(size=size)


# Export for direct import in tests
__all__ = [
    "get_embedding_model",
    "OpenAIEmbeddings",
    "HuggingFaceEmbeddings",
    "FakeEmbeddings",
]
