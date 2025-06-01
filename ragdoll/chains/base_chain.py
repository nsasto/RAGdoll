from abc import ABC, abstractmethod

class BaseChain(ABC):
    @abstractmethod
    def run(self, query: str) -> str:
        """
        Runs the chain with the given query.

        Args:
            query: The input string.

        Returns:
            The output string.
        """
        pass