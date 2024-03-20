class Config:
    """Config class for Ragdoll Class."""

    LLM_PROVIDERS = ['OpenAI','LMStudio','google/flan-t5-large']
    EMBEDDING_MODELS = ['OpenAIEmbeddings','intfloat/e5-large-v2','multi-qa-MiniLM-L6-cos-v1']
    VECTOR_DB = ['FAISS','Chroma']
    DEFAULT_CONFIG = {
        "vector_db":"FAISS",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        "max_search_results_per_query" : 3,
        "alternative_query_term_count" : 2,
        "max_workers" : 3,
        "llm": "OpenAI",
        "base_url":"http://localhost:1234/v1",
        "embeddings": "OpenAIEmbeddings",
        "log_level": 30, #logging.WARN
        "temperature": 0,
    }

    def __init__(self, config_settings=None):
        """Initialize the config class."""
        self.load_config(config_settings)

    def load_config(self, config_settings=None) -> None:
        """Load the config file."""
 
        for key, value in self.DEFAULT_CONFIG.items():
            self.__dict__[key] = value

        if not bool(config_settings):
            return None
        
        for key, value in config_settings.items():
            self.__dict__[key] = value
    
    def get_config(self):
        """Get the current configuration."""
        return self.__dict__


    def set_config(self, config_settings=None) -> None:
        """Load the config file."""
 
        if not bool(config_settings):
            return None
        
        for key, value in config_settings.items():
            self.__dict__[key] = value
            