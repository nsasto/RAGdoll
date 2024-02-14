
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from operator import itemgetter
#from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import numpy as np

import logging

from .config import Config
from .multi_query import MultiQueryRetriever
from .helpers import dotDict
from .helpers import pretty_print_docs
from .prompts import generate_RAG_template

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

    def get_llm(self, model=None, streaming=False, temperature=0, log_msg=''):
        self.logger.info(f"🤖 retrieving LLM model {log_msg}")

        model = self.cfg.llm if model is None else model
        
        if model == "OpenAI":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo-16k",
                # model="gpt-4",
                streaming=streaming,
                temperature=temperature,
            )
        elif model=="LMStudio":
            if self.cfg.base_url is None:
                raise ValueError("Local LLM model base url not set")
            
            self.llm = ChatOpenAI(
                base_url=self.cfg.base_url,
                api_key="not_need",
                streaming=streaming,
                temperature=temperature,
            )
            
        
        else:
            raise TypeError(
                "LLM model not specified. Set this in the config dictionary"
            )

        return self.llm

    def get_embeddings(self, model=None):
        model = self.cfg.embeddings if model is None else model
        self.logger.info(f"💬 retrieving embeddings for model {model}")

        if model == "OpenAIEmbeddings":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
        elif model == 'intfloat/e5-large-v2':
            #from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name='intfloat/e5-large-v2')
        elif model == 'multi-qa-MiniLM-L6-cos-v1':
            #from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name='multi-qa-MiniLM-L6-cos-v1')
        else:
            raise TypeError(
                "Embeddings model not specified. Set this in the config dictionary"
            )

        return embeddings

    def get_db(self, documents=None, embeddings=None):
        """
        Retrieves the vector database.

        Args:
            documents (list, optional): List of documents to create a new vector store. Defaults to None.
            embeddings (numpy.ndarray, optional): Pre-computed embeddings. Defaults to None.

        Returns:
            vectordb: The vector database if it exists. A new db from documents if not

        Raises:
            ValueError: If documents is None and db does not yet exists
            TypeError: If vector store is not specified in the config dictionary.
        """
        
        if (documents is None) & (self.db is None):
            raise ValueError(
                "The argument documents is required to create a new vector store unless one already exists."
            )
        elif (documents is None) & (self.db is not None):
            return self.db

        vector_store = self.cfg.vector_db 
        # get embeddings
        embeddings = self.get_embeddings() if embeddings is None else embeddings

        vectordb = None
        if vector_store.lower() == "faiss":
            self.logger.info("🗃️  retrieving vector database (FAISS)...")
            from langchain_community.vectorstores import FAISS
            vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
        elif vector_store.lower() == "chroma":
            self.logger.info("🗃️  retrieving vector database (ChromaDb)...")
            from langchain_community.vectorstores import Chroma

            vectordb = Chroma.from_documents(documents=documents, embedding=embeddings)
        else:
            raise TypeError(
                "Vector store not specified. Set this in the config dictionary"
            )

        self.db = vectordb
        return vectordb

    def save_db(self, path='ragdoll_db', db=None):
        """
        Saves the vector database to a specified path. TODO: Add support for Chroma (autoamtically saves when created...)

        Args:
            db: The vector database to be saved.
            path: The path where the vector database will be saved.
        """
        db = self.db if db is None else db
        vector_store = self.cfg.vector_db 
        
        if vector_store.lower() == "faiss":
            self.logger.info("💾 saving vector database (FAISS)...")
            from langchain_community.vectorstores import FAISS
            db.save_local(path)
        elif vector_store.lower() == "chroma":
            self.logger.error("🗃️ ChromaDb does not offer explicit save. Set persist dir on create ...")
        else:
            raise TypeError(
                "Vector store not specified. Set this in the config dictionary"
            )

        return 


    def load_db(self, path='ragdoll_db', embeddings=None):
        """
        loads the vector database from a specified path. 

        Args:
            path: The path where the vector database will be saved.
            embeddings (numpy.ndarray, optional): Pre-computed embeddings. Defaults to None.
        """
   
        vector_store = self.cfg.vector_db 
        # get embeddings
        embeddings = self.get_embeddings() if embeddings is None else embeddings

        if vector_store.lower() == "faiss":
            self.logger.info("📂 loading vector database (FAISS)...")
            from langchain_community.vectorstores import FAISS
            db=FAISS.load_local(path, embeddings)
        elif vector_store.lower() == "chroma":
            self.logger.info("🗃️ loading vector database (ChromaDb)...")
            from langchain_community.vectorstores import Chroma
            db = Chroma(persist_directory=path, embedding_function=embeddings)
        else:
            raise TypeError(
                "Vector store not specified. Set this in the config dictionary"
            )

        self.db=db
        return db
    

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
        self.logger.info("📋 getting multi query retriever")
        if db == None:
            vector_db = self.get_db(documents)
        else:
            vector_db = db

        retriever = vector_db.as_retriever()
        self.logger.info("💭 Remember that the multi query retriever will incur additional calls to your LLM")
        llm = self.get_llm(log_msg='for multi query retriever')
        return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)


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
        self.logger.info("📋 getting retriever")
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
            crcfg[key] = value
        
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
        
        compression_objects_log = ['embeddings_filter', 'splitter', 'redundant_filter', 'relevant_filter']

        log = [obj for flag, obj in zip(config_switches, compression_objects_log) if flag]
        self.logger.info(f"🗜️ Compression object pipeline: {' ➤ '.join(log)}")

        pipeline = [obj for flag, obj in zip(config_switches, compression_objects) if flag]
        
        if len(pipeline) == 0:
            raise ValueError("No compression objects were selected")
                
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=pipeline
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def answer_me_this(self, question, retriever, report_format="apa", min_words=1000):
        """
        Answers the given question using the RAG chain.

        Args:
            question (str): The question to be answered.
            retriever: The retriever object used for retrieving relevant documents.
            report_format (str, optional): The format of the generated report. Defaults to "apa".
            min_words (int, optional): The minimum number of words required in the generated report. Defaults to 1000.

        Returns:
            str: The answer to the question.
        """
        self.logger.info('🔗 Running RAG chain')
        research_prompt = PromptTemplate.from_template(template=generate_RAG_template(report_format, min_words))
        
        llm = self.get_llm(log_msg='for RAG chain')
        retrieval_chain = (
            {
                "context": itemgetter("question") | retriever | pretty_print_docs,
                "question": itemgetter('question')
            }
            | research_prompt
            | llm
            | StrOutputParser()
        )

        return retrieval_chain.invoke({"question": question})


if __name__ == "main":
    print("RAGdoll Retriever")
