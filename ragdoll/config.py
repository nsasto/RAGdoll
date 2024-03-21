class Config:
    """Config class for Ragdoll Class."""

    LLM_PROVIDERS = ['OpenAI','LMStudio','google/flan-t5-large', 'gpt-3.5-turbo-16k','gpt-4']
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
        "streaming": False,
    }

    def __init__(self, config_settings=None):
        """Initialize the config class. config_settings can be either a dictionary of settings or a Config object."""
        if not isinstance(config_settings, (Config, dict)):
            raise TypeError("config_settings must be of type Config or a dictionary")
    
        self.load_config(config_settings)

    def load_config(self, config_settings=None) -> None:
        """Load the config file."""
        
        for key, value in self.DEFAULT_CONFIG.items():
            self.__dict__[key] = value

        if not bool(config_settings):
            return None

        cfg = config_settings.get_config() if isinstance(config_settings, Config) else config_settings
        
        for key, value in cfg.items():
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
            