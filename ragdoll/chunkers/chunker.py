from abc import ABC
from typing import Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    MarkdownHeaderTextSplitter,
)
from ragdoll.config.config_manager import ConfigManager  # Import ConfigManager


class Chunker(ABC):
    def __init__(self, config: Optional[dict] = None, text_splitter: Optional[TextSplitter] = None):
        """
        Initializes the BaseChunker with an optional text splitter.

        Args:
            config: An optional configuration object. If None, the default configuration is loaded.
            text_splitter (Optional[TextSplitter]): An optional LangChain text splitter to use.
        """
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager._config  # Load default config
        else:
            self.config = config
        self.text_splitter = text_splitter
    @classmethod
    def from_config(cls):
        config_manager = ConfigManager()
        config = config_manager._config
        return cls(config=config)

    def get_text_splitter(
        self,) -> TextSplitter:
        """Returns a text splitter object, configured from default_config.yaml."""
        
        if hasattr(self, "text_splitter") and self.text_splitter is not None:
            return self.text_splitter

        chunk_size = self.config["chunker"].get("chunk_size", 1000)
        chunk_overlap = self.config["chunker"].get("chunk_overlap", 200)

        length_function = len
        is_separator_regex = False
        add_start_index = True
        if callable(length_function):
            pass
            
        default_splitter_type = self.config["chunker"]["default_splitter"]
        if default_splitter_type == "markdown":
            text_splitter = MarkdownHeaderTextSplitter(
               headers_to_split = [
                    ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
               ] )
        elif default_splitter_type == "recursive":

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                is_separator_regex=is_separator_regex,
                add_start_index=add_start_index)
        else:
            raise ValueError(f"Invalid default_splitter type: {default_splitter_type}")
        return text_splitter