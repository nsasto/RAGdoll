from typing import Optional, Dict, Any

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragdoll.config.config_manager import ConfigManager # type: ignore
 

class RagdollEmbeddings:
    def __init__(self, config: Optional[dict] = None, embeddings_model: Optional[OpenAIEmbeddings] = None) -> None:
        if config is None:
            config_manager = ConfigManager()  # type: ignore
            self.config = config_manager._config #type: ignore
        else:
            self.config = config
        self.embeddings_model = embeddings_model

    @classmethod
    def from_config(cls) -> "RagdollEmbeddings":
        config_manager = ConfigManager()  # type: ignore
        config = config_manager._config  # type: ignore
        return cls(config=config)

    def get_embeddings_model(self) -> OpenAIEmbeddings | HuggingFaceEmbeddings:
        if self.embeddings_model is not None:
            return self.embeddings_model

        embeddings_config = self.config.get("embeddings", {})
        model_type = embeddings_config.get("default_model", "openai")


        if model_type == "openai":    
            model_params = embeddings_config.get("openai", {})
            return self._create_openai_embeddings(model_params=model_params)
        elif model_type == "huggingface":
            model_params = embeddings_config.get("huggingface", {})
            return self._create_huggingface_embeddings(model_params=model_params)
        else:        
            return OpenAIEmbeddings()  # type: ignore

    def _create_openai_embeddings(self, model_params: Dict[str, Any]) -> OpenAIEmbeddings:
        """Creates an OpenAIEmbeddings model with parameters from config.

        Args:
            model_params: Parameters for OpenAIEmbeddings.
        Returns:
            OpenAIEmbeddings: An instance of OpenAIEmbeddings.
        """
        model_params = model_params or {}
        model_name = model_params.get("model", "text-embedding-3-large")
        dimensions = model_params.get("dimensions")

        # Remove 'model' from model_params to avoid passing it twice
        model_params.pop("model", None)

        # Remove 'dimensions' from model_params to avoid passing it twice
        model_params.pop("dimensions", None)

        if dimensions is not None:
            return OpenAIEmbeddings(model=model_name, dimensions=dimensions, **model_params)
        else:
            return OpenAIEmbeddings(model=model_name, **model_params)


    def _create_huggingface_embeddings(self, model_params: Dict[str, Any]) -> HuggingFaceEmbeddings:
        """Creates a HuggingFaceEmbeddings model with parameters from config.

        Args:
            model_params: Parameters for HuggingFaceEmbeddings.
        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings.
        """
        model_params = model_params or {}
        model_name = model_params.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        
        # Remove 'model_name' from model_params to avoid passing it twice
        model_params.pop("model_name", None)
        return HuggingFaceEmbeddings(model_name=model_name, **model_params)