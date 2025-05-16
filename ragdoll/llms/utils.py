import asyncio
import logging
from typing import Union, Optional, Any

logger = logging.getLogger("ragdoll.llms.utils")

async def call_llm(
    llm: Any, 
    prompt: str, 
    return_raw_response: bool = False
) -> Union[str, Any]:
    """
    Unified helper function to call any LLM with proper error handling and response processing.
    
    Args:
        llm: LLM instance (LangChain model or custom callable)
        prompt: The text prompt to send to the LLM
        return_raw_response: If True, returns the raw response object instead of extracted text
        
    Returns:
        The LLM's response as a string, or the raw response object if return_raw_response is True
    """
    if not prompt:
        logger.warning("Empty prompt passed to call_llm")
        return "" if not return_raw_response else None
        
    try:
        # Ensure LLM is properly defined
        if not llm:
            logger.error("LLM is not defined")
            return "" if not return_raw_response else None
            
        # Import here to avoid circular imports
        try:
            from langchain_core.language_models import BaseChatModel, BaseLanguageModel
            from langchain_core.messages import HumanMessage
            
            is_langchain_model = isinstance(llm, (BaseChatModel, BaseLanguageModel))
        except ImportError:
            # If langchain isn't available, assume it's not a langchain model
            is_langchain_model = False
            
        if is_langchain_model:
            # Handle LangChain models with .invoke method
            if asyncio.iscoroutinefunction(llm.invoke):
                # For async LLMs
                response = await llm.invoke(prompt)
            else:
                # For sync LLMs
                response = llm.invoke(prompt)
                
            if return_raw_response:
                return response
                
            # Handle different response types
            if hasattr(response, "content"):
                # If response is a message with content attribute
                return response.content
            elif isinstance(response, str):
                # If response is a string
                return response
            else:
                # Try to get a useful string representation
                logger.debug(f"Unexpected LLM response type: {type(response)}")
                return str(response)
        else:
            # Handle custom callables (non-LangChain models)
            if asyncio.iscoroutinefunction(llm):
                response = await llm(prompt)
            else:
                response = llm(prompt)
                
            return response if return_raw_response else str(response)
            
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return "" if not return_raw_response else None