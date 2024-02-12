from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

import logging

from .config import Config
from .multi_query import MultiQueryRetriever


class RagdollRetriever:
    def __init__(self, config={}):
        """
        Initializes a RagdollIndex object.

        Args:
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.cfg = Config(config)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.cfg.enable_logging else logging.INFO)
        self.logger.propagate = True

        # Initialize
        self.db = None
        self.llm = self.get_llm()

    def get_llm(self, model=None, streaming=False, temperature=0):
        self.logger.debug("retrieving LLM model")

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
        self.logger.debug("retrieving embeddings")
        model = self.cfg.embeddings = "OpenAIEmbeddings" if model is None else model

        if model == "OpenAIEmbeddings":
            embeddings = OpenAIEmbeddings()
        else:
            raise TypeError(
                "Embeddings model not specified. Set this in the config dictionary"
            )

        self.embeddings = embeddings
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
        self.logger.debug("retrieving vector database")
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

    def _get_mq_retriever(self, retriever):
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
        self.logger.debug("retrieving document retriever")
        if db == None:
            vector_db = self.get_db() if documents is None else self.get_db(documents)
        else:
            vector_db = db

        retriever = vector_db.as_retriever()
        
        #if base retriever type is set as a multi query retriever, then get that
        if (self.cfg.base_retriever=='MULTI_QUERY'):
            self.logger.debug("retrieving multi query document retriever")
            self.retriever = self._get_mq_retriever(retriever)
        else:
            self.logger.debug("retrieving base document retriever")
            self.retriever = retriever

        return self.retriever


if __name__ == "main":
    print("RAGdoll Retriever")
