from abc import ABC, abstractmethod

class BaseLLM(ABC):
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