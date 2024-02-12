from langchain_openai import OpenAIEmbeddings
import logging

from .config import Config

class RagdollRetriever:
    def __init__(self, config = {}):
        """
        Initializes a RagdollIndex object.

        Args:
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.cfg = Config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.cfg.enable_logging else logging.INFO)
        #Initialize
        self.db = None

    def get_embeddings(self, model=None):
        model = self.cfg.embeddings='OpenAIEmbeddings' if model is None else model    
        
        if (model == "OpenAIEmbeddings"):
            from langchain_community.vectorstores import FAISS
            embeddings = OpenAIEmbeddings()
        else:
            raise TypeError("Embeddings model not specified. Set this in the config dictionary")

        self.embeddings=embeddings
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
        if self.db is not None:
            return self.db
        if documents is None:
            raise ValueError("The argument documents is required to create a new vector store")

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
            raise TypeError("Vector store not specified. Set this in the config dictionary")

        self.db = vectordb
        return vectordb


    def get_retriever(self,documents=None, db=None):
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
        if (db==None):
            vector_db = self.get_db() if documents is None else self.get_db(documents)
        else:
            vector_db=db

        self.retriever = vector_db.as_retriever()
        return self.retriever
    
if (__name__=='main'):
    print('RAGdoll Retriever')

