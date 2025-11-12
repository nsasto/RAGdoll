from typing import Optional, List, Dict
from langchain_core.llms import LLM
from ragdoll.ingestion.ingestion_service import IngestionService
from ragdoll.llms.openai_llm import MyOpenAI
from ragdoll.loaders.base_loader import BaseLoader
from ragdoll.chunkers.base_chunker import BaseChunker
from ragdoll.embeddings.base_embeddings import BaseEmbeddings
from ragdoll.vector_stores.base_vector_store import BaseVectorStore
from ragdoll.vector_stores.factory import vector_store_from_config
from ragdoll.llms.base_llm import BaseLLM
from ragdoll.tools.search_tools import SearchInternetTool, SuggestedSearchTermsTool
from ragdoll.graph_stores.base_graph_store import BaseGraphStore

from ragdoll.config import Config
from ragdoll.config.config_manager import ConfigManager

class Ragdoll:
    def __init__(
        self,
        
        loader: Optional[BaseLoader] = None,
        chunker: Optional[BaseChunker] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        vector_store: Optional[BaseVectorStore] = None,
        llm: Optional[LLM] = None,  # Optional LLM instance passed in
        graph_store: Optional[BaseGraphStore] = None,
    ):
        self.config = Config()
        self.config_manager = ConfigManager()

        # Use the provided LLM or create a default one
        if llm:
            self.llm = llm
        else:
            self.llm = MyOpenAI()
        # Instantiate tools
        self.search_tool = SearchInternetTool(self.config)
        self.suggest_terms_tool = SuggestedSearchTermsTool(self.config, self.llm)


        self.loader = loader 
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.graph_store = graph_store

        # Default implementations
        if self.loader is None:
            from ragdoll.loaders.directory_loader import DirectoryLoader
            self.loader = DirectoryLoader()
        
        if self.chunker is None:
            from ragdoll.chunkers.recursive_character_text_splitter import MyRecursiveCharacterTextSplitter
            self.chunker = MyRecursiveCharacterTextSplitter()
        
        if self.embeddings is None:
            from ragdoll.embeddings.openai_embeddings import MyOpenAIEmbeddings
            self.embeddings = MyOpenAIEmbeddings()

        if self.vector_store is None:
            vector_config = self.config_manager.vector_store_config
            self.vector_store = vector_store_from_config(
                vector_config,
                embedding=self.embeddings,
            )
        
        if graph_store is None:
            from ragdoll.graph_stores.networkx_graph_store import MyNetworkxGraphStore
            self.graph_store = MyNetworkxGraphStore()



    def ingest_data(self, sources):
        """Ingests data from a list of sources."""
        service = IngestionService()
        return service.ingest_documents(sources)

    def run(self, prompt: str) -> str:
        """Run the LLM with the given prompt."""
        response = self.llm._call(prompt) 
        return response
    
    def run_index_pipeline(self, query: str, **kwargs) :
        """Run the entire process, taking a query as input."""
        
        # 1. Get suggested search terms
        suggested_terms = self.suggest_terms_tool.run(query=query, num_suggestions=3)
        
        # 2. Search the internet using the suggested terms
        all_results = []
        for term in suggested_terms:
            results = self.search_tool.run(query=term, num_results=3)
            all_results.extend(results)
        search_results = all_results
        
        # ... rest of the pipeline using the search_results ...
        print(f"Search Results: {search_results}") # For debugging - remove in production
        # ... rest of the pipeline ...
        
        return  # Or return something meaningful from the pipeline
        
        
