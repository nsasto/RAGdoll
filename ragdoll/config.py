class Config:
    """Config class for GPT Researcher."""
    _RETRIEVERS = ['BASE','MULTIQUERY']
    _LLM_PROVIDERS = ['OpenAI']
    _EMBEDDINGS_MODELS = ['OpenAIEmbeddings']
    _VECTOR_DB = ['FAISS','Chroma']

    def __init__(self, config_settings=None):
        """Initialize the config class."""

        self.vector_db = "FAISS"
        self.retriever = "BASE"
        self.user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0")
        self.max_search_results_per_query = 3
        self.alternative_query_term_count = 2
        self.max_workers = 3
        self.embeddings = "OpenAIEmbeddings"
        self.enable_logging = False
        self.load_config(config_settings)

    def load_config(self, config_settings=None) -> None:
        """Load the config file."""
        if config_settings is None:
            return None

        for key, value in config_settings.items():
            self.__dict__[key] = value
