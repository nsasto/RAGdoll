from typing import Optional, List
from langchain.docstore.document import Document  # Assuming you're using LangChain's Document

from ragdoll.loaders.base_loader import BaseLoader
from ragdoll.chunkers.base_chunker import BaseChunker
from ragdoll.embeddings.base_embeddings import BaseEmbeddings
from ragdoll.vector_stores.base_vector_store import BaseVectorStore
from ragdoll.llms.base_llm import BaseLLM
from ragdoll.graph_stores.base_graph_store import BaseGraphStore


class Ragdoll:
    def __init__(
        self,
        
        loader: Optional[BaseLoader] = None,
        chunker: Optional[BaseChunker] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        vector_store: Optional[BaseVectorStore] = None,
        llm: Optional[BaseLLM] = None,
        graph_store: Optional[BaseGraphStore] = None,
    ):

        # Default implementations


        self.loader = loader
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm

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
            from ragdoll.vector_stores.chroma_vector_store import MyChroma
            self.vector_store = MyChroma()
        
        if self.llm is None:
            from ragdoll.llms.openai_llm import MyOpenAI
            self.llm = MyOpenAI()
        
        if graph_store is None:
            from ragdoll.graph_stores.networkx_graph_store import MyNetworkxGraphStore
            self.graph_store = MyNetworkxGraphStore()

    def run(self, prompt: str) -> str:
        """Run the LLM with the given prompt."""
        response = self.llm.call(prompt)
        return response

    def run(self, prompt: str) -> str:
        response = self.llm.call(prompt)
        return response