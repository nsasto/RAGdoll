"""
LLM utilities for RAGdoll (Transitioning to LangChain).

This module provides utilities for working with chat-based language models through LangChain's init_chat_model.
It now supports loading LLMs by name from config and caching basic LLMs, with automatic default config loading.
"""
import os
import logging
import importlib.util
from typing import Dict, Any, Optional, Union
from langchain_core.language_models import BaseChatModel, BaseLanguageModel

logger = logging.getLogger(__name__)

# Global cache for basic LLMs
_llm_cache: Dict[str, Union[BaseChatModel, BaseLanguageModel]] = {}

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
    model_name: Optional[str] = None

    if isinstance(model_name_or_config, str):
        model_name = model_name_or_config
        model_list = llm_config.get("model_list", [])
        for model in model_list:
            if model.get("model_name") == model_name:
                model_config = model.get("params", {})
                provider = model.get("provider")
                break

        if model_config is None or provider is None:
            logger.error(f"Model '{model_name}' not found in config or missing provider.")
            return None
    elif isinstance(model_name_or_config, dict):
        model_config = model_name_or_config
        provider = model_config.get("provider")
        model_name = model_config.get("model") # Try to get model name from direct config
        if provider is None:
            logger.error("Provider not specified in model configuration.")
            return None
        if model_name is None:
            logger.warning("Model name not specified in direct configuration.")
    else:
        logger.error(f"Invalid model_name_or_config type: {type(model_name_or_config)}. Expected str or dict.")
        return None

    if model_config is None or provider is None or model_name is None:
        return None

    processed_config = _resolve_env_vars(model_config)
    _set_api_key_from_config(provider, processed_config)

    try:
        model = init_chat_model(
            model=model_name,
            model_provider=provider,
            **processed_config
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize {provider} model '{model_name}': {e}")
        return None

def get_basic_llm(model_type: str = "default", config_manager=None) -> Optional[Union[BaseChatModel, BaseLanguageModel]]:
    """
    Retrieves a basic LLM (default, reasoning, or vision) from the cache or loads it.
    Automatically loads the default config if config_manager is not provided.

    Args:
        model_type: The type of basic model to retrieve ('default', 'reasoning', 'vision'). Defaults to 'default'.
        config_manager: Optional ConfigManager instance. If None, the default config is loaded.

    Returns:
        The cached or newly loaded LangChain ChatModel or LanguageModel object, or None if an error occurs.
    """
    if model_type not in ["default", "reasoning", "vision"]:
        logger.error(f"Invalid model_type: '{model_type}'. Must be 'default', 'reasoning', or 'vision'.")
        return None

    if model_type in _llm_cache:
        return _llm_cache[model_type]

    if config_manager is None:
        config_manager = _load_default_config()
        if config_manager is None:
            logger.warning("No ConfigManager provided and default loading failed. Cannot retrieve basic LLMs.")
            return None

    if config_manager:
        llm_config = config_manager._config.get("llms", {})
        model_name = llm_config.get(model_type + "_model")
        if model_name:
            llm = get_llm(model_name, config_manager)
            if llm:
                _llm_cache[model_type] = llm
            return llm
        else:
            logger.warning(f"'{model_type}_model' not defined in the 'llms' section of the config.")
            return None
    return None

# Example Usage (assuming you have ragdoll installed):
if __name__ == "__main__":
    # Get an LLM by name without providing config_manager (will load default)
    default_gpt = get_llm("gpt-3.5-turbo")
    if default_gpt:
        print(f"Loaded (default config): {default_gpt}")

    # Get the default LLM (will load default config and be cached)
    default_reasoning_llm = get_basic_llm("default")
    if default_reasoning_llm:
        print(f"Default LLM (default config): {default_reasoning_llm}")
        default_llm_again = get_basic_llm("default")
        print(f"Default LLM (cached, default config): {default_llm_again}")
        assert default_reasoning_llm is default_llm_again

    # Get the default LLM using no model name
    default_llm_no_name = get_llm()
    if default_llm_no_name:
        print(f"Default LLM (no name specified): {default_llm_no_name}")

    # You can still provide a specific config_manager if needed
    try:
        from ragdoll.config.config_manager import ConfigManager
        custom_config_manager = ConfigManager(config_path="path/to/your/custom_config.yaml") # Replace with actual path
        claude = get_llm("claude-3-5-sonnet", custom_config_manager)
        if claude:
            print(f"Loaded (custom config): {claude}")
    except ImportError:
        print("ConfigManager not available for custom config example.")
    except FileNotFoundError:
        print("Custom config file not found for example.")