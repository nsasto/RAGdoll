class Config:
    """Config class for Ragdoll Class."""

    _RETRIEVERS = ['SINGLE_QUERY','MULTI_QUERY']
    _LLM_PROVIDERS = ['OpenAI']
    _EMBEDDINGS_MODELS = ['OpenAIEmbeddings']
    _VECTOR_DB = ['FAISS','Chroma']
    _DEFAULT_CONFIG = {
        "vector_db":"FAISS",
        "base_retriever": "SINGLE_QUERY",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        "max_search_results_per_query" : 3,
        "alternative_query_term_count" : 2,
        "max_workers" : 3,
        "llm": "OpenAI",
        "embeddings": "OpenAIEmbeddings",
        "log_level": 30, #logging.WARN
    }

    def __init__(self, config_settings=None):
        """Initialize the config class."""
        self.load_config(config_settings)

    def load_config(self, config_settings=None) -> None:
        """Load the config file."""
 
        for key, value in self._DEFAULT_CONFIG.items():
            self.__dict__[key] = value

        if config_settings is None:
            return None
        
        for key, value in config_settings.items():
            self.__dict__[key] = value
    
    def get_config(self):
        """Get the current configuration."""
        return self.__dict__
