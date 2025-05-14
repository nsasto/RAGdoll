"""
LLM utilities for RAGdoll.

This module provides utilities for working with language models through LiteLLM.
"""
import os
import logging
from typing import Dict, Any, Optional, Union, Callable
import importlib.util

logger = logging.getLogger(__name__)

def is_litellm_available() -> bool:
    """Check if LiteLLM is available."""
    return importlib.util.find_spec("litellm") is not None

def get_litellm_model(model_name_or_config: Union[str, Dict[str, Any]], config_manager=None) -> Optional[Callable[[str], str]]:
    """
    Get a LiteLLM model based on a model name or configuration.
    
    Args:
        model_name_or_config: Either a string model name or a dict with model configuration
        config_manager: Optional ConfigManager instance to load models from config
    
    Returns:
        A callable function that takes a prompt and returns a response
    """
    if not is_litellm_available():
        logger.error("LiteLLM not installed. Please install with: pip install litellm")
        return None
        
    # Import here to avoid dependency if not needed
    import litellm
    
    # If a string was provided, look up in config
    if isinstance(model_name_or_config, str):
        model_name = model_name_or_config
        
        # If no config manager provided, try to load it
        if config_manager is None:
            try:
                from ragdoll.config.config_manager import ConfigManager
                config_manager = ConfigManager()
            except Exception as e:
                logger.error(f"Failed to load ConfigManager: {e}")
                return None
                
        # Get the model list from config
        model_list = config_manager._config.get("llms", {}).get("model_list", [])
        
        # Find the requested model in the list
        model_config = None
        for model in model_list:
            if model.get("model_name") == model_name:
                model_config = model.get("litellm_params", {})
                break
                
        # If model not found, try to use default parameters
        if model_config is None:
            logger.warning(f"Model '{model_name}' not found in config, using basic parameters")
            model_config = {"model": model_name}
    else:
        # Use the provided config directly
        model_config = model_name_or_config
        
    # Process environment variable references (os.environ/VAR_NAME)
    processed_config = {}
    for key, value in model_config.items():
        if isinstance(value, str) and value.startswith("os.environ/"):
            env_var = value.split("/", 1)[1]
            env_value = os.environ.get(env_var)
            if env_value is None:
                logger.warning(f"Environment variable '{env_var}' not found for key '{key}'")
            processed_config[key] = env_value
        else:
            processed_config[key] = value
            
    # Return a function that calls LiteLLM with the configuration
    def completion_fn(prompt: str, **kwargs) -> str:
        try:
            # Merge default config with any overrides
            params = processed_config.copy()
            params.update(kwargs)
            
            # Extract the model and remove from params
            model = params.pop("model", "gpt-3.5-turbo")
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            # Extract and return the content
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content.strip()
                elif hasattr(response.choices[0], 'text'):
                    return response.choices[0].text.strip()
            
            # Fallback for unusual response formats
            return str(response)
            
        except Exception as e:
            logger.error(f"Error calling LiteLLM: {e}")
            raise
            
    return completion_fn