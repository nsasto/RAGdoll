from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

import numpy as np

import logging

from .config import Config
from .multi_query import MultiQueryRetriever
from .helpers import dotDict

class RagdollRetriever:
    def __init__(self, config={}):
        """
        Initializes a RagdollIndex object.

        Args:
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.cfg = Config(config)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.cfg.log_level)

        # Initialize
        self.db = None
        self.llm = self.get_llm()

    def get_llm(self, model=None, streaming=False, temperature=0):
        self.logger.info("retrieving LLM model")

        model = self.cfg.llm if model is None else model
        if model == "OpenAI":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo-16k",
                # model="gpt-4",
                streaming=streaming,
                temperature=temperature,
            )
        else:
            raise TypeError(
                "LLM model not specified. Set this in the config dictionary"
            )

        return self.llm

    def get_embeddings(self, model=None):
        self.logger.info("retrieving embeddings")
        model = self.cfg.embeddings = "OpenAIEmbeddings" if model is None else model

        if model == "OpenAIEmbeddings":
            embeddings = OpenAIEmbeddings()
        else:
            raise TypeError(
                "Embeddings model not specified. Set this in the config dictionary"
            )

        return embeddings

    def get_db(self, documents=None, vector_store=None, embeddings=None):
        """
        Retrieves the vector database.

        Args:
            documents (list, optional): List of documents to create a new vector store. Defaults to None.
            vector_store (str, optional): Type of vector store. Defaults to None.
            embeddings (numpy.ndarray, optional): Pre-computed embeddings. Defaults to None.

        Returns:
            vectordb: The vector database if it exists. A new db from documents if not

        Raises:
            ValueError: If documents is None and db does not yet exists
            TypeError: If vector store is not specified in the config dictionary.
        """
        self.logger.info("retrieving vector database")
        if self.db is not None:
            return self.db
        if documents is None:
            raise ValueError(
                "The argument documents is required to create a new vector store"
            )

        vector_store = self.cfg.vector_db if vector_store is None else vector_store
        # get embeddings
        embeddings = self.get_embeddings() if embeddings is None else embeddings

        vectordb = None
        if vector_store.lower() == "faiss":
            from langchain_community.vectorstores import FAISS
            vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
        elif vector_store.lower() == "chroma":
            from langchain_community.vectorstores import Chroma

            vectordb = Chroma.from_documents(documents=documents, embedding=embeddings)
        else:
            raise TypeError(
                "Vector store not specified. Set this in the config dictionary"
            )

        self.db = vectordb
        return vectordb

    def get_mq_retriever(self, documents=None, db=None):
        """
        Returns a multi query retriever object based on the specified vector store.

        Args:
            documents (list): List of documents to be used for creating the retriever or
            db (vetor database): a populated vector db for conversion to a retriever

        Returns:
            retriever: The retriever object based on the specified vector store.
                       If the vector store already exists, will convert it to a langchain retriever
                       If documents are provided (and no db), a new db will be created

        Raises:
            TypeError: If the vector store is not specified in the config dictionary.
        """
        self.logger.info("retrieving multi query document retriever")
        if db == None:
            vector_db = self.get_db(documents)
        else:
            vector_db = db

        retriever = vector_db.as_retriever()
        self.logger.info("Remember that the multi query retriever will incur additional calls to your LLM")
        return MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)


    def get_retriever(self, documents=None, db=None):
        """
        Returns a retriever object based on the specified vector store.

        Args:
            documents (list): List of documents to be used for creating the retriever or
            db (vetor database): a populated vector db for conversion to a retriever

        Returns:
            retriever: The retriever object based on the specified vector store.
                       If the vector store already exists, will convert it to a langchain retriever
                       If documents are provided (and no db), a new db will be created

        Raises:
            TypeError: If the vector store is not specified in the config dictionary.
        """
        self.logger.info("retrieving document retriever")
        if db == None:
            vector_db = self.get_db(documents)
        else:
            vector_db = db

        retriever = vector_db.as_retriever()
        
        return retriever

    def _default_compressor_config(self):
        return {
            "use_embeddings_filter":True, 
            "use_splitter":True, 
            "use_redundant_filter":True, 
            "use_relevant_filter":True,
            "embeddings":None,
            "similarity_threshold":0.76, #embeddings filter settings
            "chunk_size":500,   #text filter settings
            "chunk_overlap":0, 
            "separator":". ",
        }

    def get_compression_retriever(self, base_retriever, compressor_config={}):
        """
        Returns a compression retriever object based on the provided base retriever and compressor configuration.

        Args:
            base_retriever: The base retriever object.
            compressor_config: A dictionary containing the compressor configuration parameters.

        Returns:
            compression_retriever: The compression retriever object.

        Raises:
            ValueError: If no compression objects were selected.
        """
        crcfg=self._default_compressor_config()
        for key, value in compressor_config.items():
            crcfg.__dict__[key] = value
        
        crcfg = dotDict(crcfg)

        #embeddings filter
        embeddings = self.get_embeddings() if crcfg.embeddings is None else crcfg.embeddings
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=crcfg.similarity_threshold)
        # Split documents into chunks of half size, 500 characters, with no characters overlap.
        splitter = CharacterTextSplitter(chunk_size=crcfg.chunk_size, chunk_overlap=crcfg.chunk_overlap, separator=crcfg.separator)

        # Remove redundant chunks, based on cosine similarity of embeddings.
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

        # Remove irrelevant chunks, based on cosine similarity of embeddings.
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=crcfg.similarity_threshold)

        # boolean vector 
        config_switches = np.array([crcfg.use_embeddings_filter, crcfg.use_splitter, crcfg.use_redundant_filter, crcfg.use_relevant_filter])

        # list of objects
        compression_objects = [embeddings_filter, splitter, redundant_filter, relevant_filter]
        
        if len(compression_objects) == 0:
            raise ValueError("No compression objects were selected")
        
        compression_objects_log = ['embeddings_filter', 'splitter', 'redundant_filter', 'relevant_filter']

        log = [obj for flag, obj in zip(config_switches, compression_objects_log) if flag]
        self.logger.info(f"Compression object pipeline: {' ➤ '.join(log)}")

        pipeline = [obj for flag, obj in zip(config_switches, compression_objects) if flag]
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=pipeline
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        
        return compression_retriever

if __name__ == "main":
    print("RAGdoll Retriever")
