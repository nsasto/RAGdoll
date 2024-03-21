import logging

from .config import Config

class RagdollLLM:
    def __init__(self, config = {}, log_msg=''):
        """
        Initializes a Ragdoll LLM object.

        Args:
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.cfg = Config(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.cfg.log_level)

        # Initialize
        self.llm = self.get_llm(self.cfg, log_msg=log_msg )
    
    def get_llm(self, cfg, log_msg):
        llm = cfg.llm
        streaming = cfg.streaming
        temperature = cfg.temperature
        base_url = cfg.base_url

        #if llm not in Config.LLM_PROVIDERS then return error
        if llm not in Config.LLM_PROVIDERS:
            raise ValueError(f"Specified LLM provider {llm} not found in {Config.LLM_PROVIDERS}")
        
        self.logger.info(f"🤖 retrieving {llm} model {log_msg}")  
        if (llm == "OpenAI") or (llm == "gpt-3.5-turbo-16k") or (llm == "gpt-4"):
            from langchain_openai import ChatOpenAI
            model = "gpt-3.5-turbo-16k" if llm == "OpenAI" else llm

            result_llm = ChatOpenAI(
                model=model,
                streaming=streaming,
                temperature=temperature,
            )

        elif llm=="LMStudio":
            from langchain_openai import ChatOpenAI

            if base_url is None:
                raise ValueError("Local LLM model base url not set")
            
            result_llm = ChatOpenAI(
                base_url=base_url,
                api_key="not_need",
                streaming=streaming,
                temperature=temperature,
            )

        elif llm=="google/flan-t5-large": 
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

            model_id = 'google/flan-t5-large'
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            pipe = pipeline(
                'text2text-generation',
                model=model,
                tokenizer=tokenizer,
                max_length=500
            )
            result_llm = HuggingFacePipeline(pipeline=pipe)

        return result_llm 

class RagdollEmbeddings:
    def __init__(self, model, log_level=30):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Initialize
        self.embeddings = self.get_embeddings(model)
   
    def get_embeddings(self, model=None):
        #if model not in Config.EMBEDDING_MODELS then return error
        if model not in Config.EMBEDDING_MODELS:
            raise ValueError(f"Specified embeddings provider {model} not found in {Config.EMBEDDING_MODELS}")
        
        self.logger.info(f"💬 retrieving embeddings for model {model} ")

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

        return embeddings

if __name__ == "main":
    print("RAGdoll models (LLM and Embeddings models)")