from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    _llm_instance = None  # Class-level attribute to store the instance

    @abstractmethod
    def call(self, prompt: str) -> str:
        """
        Abstract method to generate a response from a language model.

        Args:
           prompt: The input prompt string.

        Returns:
           The language model's response string.
        """
        pass


    @classmethod
    def get_llm(cls, *args, **kwargs) -> "BaseLLM":
        """
        Class method to get the shared LLM instance.

        If an instance doesn't exist, it creates a new one and stores it.
        """
        if cls._llm_instance is None:
            cls._llm_instance = cls(*args, **kwargs)  # Create a new instance if none exists
        return cls._llm_instance

    
