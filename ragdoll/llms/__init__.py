"""
LLM utilities for RAGdoll (Transitioning to LangChain).

This module provides utilities for working with chat-based language models through LangChain's init_chat_model.
It supports loading LLMs by name from config with automatic default config loading.
"""
import os
import logging
import importlib.util
from typing import Dict, Any, Optional, Union
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from .utils import call_llm

logger = logging.getLogger("ragdoll.llms")

def is_langchain_available() -> bool:
    """Check if LangChain core is available."""
    return importlib.util.find_spec("langchain_core") is not None

def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves environment variable references in a configuration dictionary."""
    processed_config = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("os.environ/"):
            env_var = value.split("/", 1)[1]
            env_value = os.environ.get(env_var)
            if env_value is None:
                logger.warning(f"Environment variable '{env_var}' not found for key '{key}'")
            processed_config[key] = env_value
        else:
            processed_config[key] = value
    return processed_config

def _set_api_key_from_config(provider: Optional[str], config: Dict[str, Any]) -> None:
    """Sets the API key as an environment variable if provided in the config."""
    api_key = config.pop("api_key", None)
    if api_key:
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "google":
            os.environ["GOOGLE_API_KEY"] = api_key

def _load_default_config():
    """Loads the default ConfigManager instance."""
    try:
        from ragdoll.config.config_manager import ConfigManager
        return ConfigManager()
    except ImportError:
        logger.warning("ConfigManager not found. Ensure 'ragdoll' is installed for automatic config loading.")
        return None
    except Exception as e:
        logger.error(f"Failed to load default ConfigManager: {e}")
        return None

def get_llm(model_name_or_config: Union[str, Dict[str, Any]] = None, config_manager=None) -> Optional[Union[BaseChatModel, BaseLanguageModel]]:
    """
    Get a LangChain model based on a model name (from config) or a direct configuration.
    Automatically loads the default config if config_manager is not provided.

    Args:
        model_name_or_config: Either a string model name (looked up in config)
                              or a dict with the model configuration. If None, uses the default model.
        config_manager: Optional ConfigManager instance. If None, the default config is loaded.

    Returns:
        The instantiated LangChain ChatModel or LanguageModel object, or None if an error occurs.
    """
    if not is_langchain_available():
        logger.error("LangChain core not installed. Please install with: pip install langchain-core")
        logger.info("You might also need provider-specific packages like langchain-openai, langchain-anthropic, langchain-google-genai.")
        return None

    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        logger.error("Failed to import LangChain chat models. Ensure you have the necessary LangChain integrations installed.")
        return None

    if config_manager is None:
        config_manager = _load_default_config()
        if config_manager is None:
            logger.warning("No ConfigManager provided and default loading failed. Cannot load LLM.")
            return None

    llm_config = config_manager._config.get("llms", {})
    default_model_name = llm_config.get("default_model")

    if model_name_or_config is None:
        if default_model_name:
            model_name_or_config = default_model_name
            logger.info(f"Using default model: {default_model_name}")
        else:
            logger.error("No model specified and no default model found in config.")
            return None

    model_config: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    config_model_name: Optional[str] = None  # For lookup in config
    actual_model_name: Optional[str] = None  # For passing to LangChain

    if isinstance(model_name_or_config, str):
        config_model_name = model_name_or_config
        model_list = llm_config.get("model_list", [])
        for model in model_list:
            if model.get("model_name") == config_model_name:
                model_config = model.get("params", {}).copy()  # Make a copy to avoid modifying the original
                provider = model.get("provider")
                # Get the actual model name from params if available, otherwise use model_name as fallback
                actual_model_name = model_config.pop("model", config_model_name)
                break

        if model_config is None or provider is None:
            available_models = [model.get("model_name") for model in model_list if "model_name" in model]
            logger.error(f"Model '{config_model_name}' not found in config or missing provider. Available models: {available_models}")
            return None
    elif isinstance(model_name_or_config, dict):
        model_config = model_name_or_config.copy()  # Make a copy to avoid modifying the original
        provider = model_config.pop("provider", None)  # Remove to avoid duplicate
        actual_model_name = model_config.pop("model", None)  # Remove to avoid duplicate
        if provider is None:
            logger.error("Provider not specified in model configuration.")
            return None
        if actual_model_name is None:
            logger.warning("Model name not specified in direct configuration.")
    else:
        logger.error(f"Invalid model_name_or_config type: {type(model_name_or_config)}. Expected str or dict.")
        return None

    if model_config is None or provider is None or actual_model_name is None:
        return None

    processed_config = _resolve_env_vars(model_config)
    _set_api_key_from_config(provider, processed_config)

    try:
        logger.debug(f"Initializing {provider} model '{actual_model_name}' with config keys: {list(processed_config.keys())}")
        model = init_chat_model(
            model=actual_model_name,
            model_provider=provider,
            **processed_config
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize {provider} model '{actual_model_name}': {e}")
        return None

# Example Usage (assuming you have ragdoll installed):
if __name__ == "__main__":
    # Get an LLM by name without providing config_manager (will load default)
    default_gpt = get_llm("gpt-3.5-turbo")
    if default_gpt:
        print(f"Loaded model: {default_gpt}")
    
    # Get the default LLM using no model name (uses default from config)
    default_llm = get_llm()
    if default_llm:
        print(f"Default model from config: {default_llm}")
    
    # Get a model using a direct configuration
    custom_config = {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7
    }
    custom_llm = get_llm(custom_config)
    if custom_llm:
        print(f"Custom configured model: {custom_llm}")

    # Example with custom config file
    try:
        from ragdoll.config.config_manager import ConfigManager
        custom_config_manager = ConfigManager(config_path="path/to/your/custom_config.yaml")  # Replace with actual path
        claude = get_llm("claude-3-5-sonnet", custom_config_manager)
        if claude:
            print(f"Loaded with custom config: {claude}")
    except ImportError:
        print("ConfigManager not available for custom config example.")
    except FileNotFoundError:
        print("Custom config file not found for example.")