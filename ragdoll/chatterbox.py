from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

#from langchain_openai import OpenAIEmbeddings

import logging
from .config import Config
from .prompts import (
    generate_standalone_history_prompt,
    generate_context_chat_prompt
)
from .models import RagdollLLM
import random


class ChatterBox:
    def __init__(self, retriever, config = {}, chat_prompt=None, history_prompt=None, session_id=None):
        """
        Initializes a ChatterBox chat object.

        Args:
            retriever: a langchain retriever object
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.cfg = Config(config)
        #session id is random number is session_id is None
        self.session_id = session_id if session_id is not None else random.randint(0, 1000000)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.cfg.log_level)
        self.logger.info(f'🧠 Initializing ChatterBox')
        # define the LLM
        llm = RagdollLLM(self.cfg).llm
        ### Statefully manage chat history ###
        self.chat_history_store = {}

        history_prompt = generate_standalone_history_prompt(history_prompt) 
        history_aware_retriever = create_history_aware_retriever(llm, retriever, history_prompt)
        chat_prompt = generate_context_chat_prompt(chat_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        self.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        
    def get_config(self):
        return self.cfg

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = ChatMessageHistory()
        return self.chat_history_store[session_id]
    
    def chat(self, input):
        response = self.conversational_rag_chain.invoke(
            {"input": input},
            config={
                "configurable": {"session_id": self.session_id}
            },  # constructs a key "abc123" in `store`.
        )
        

        # Extract metadata sources and start indexes
        metadata_sources = {}
        for doc in response['context']:
            source = doc.metadata['source']
            start_index = doc.metadata['start_index']
            if source not in metadata_sources:
                metadata_sources[source] = []
            metadata_sources[source].append(start_index)

        #add metadata to response
        response['metadata'] = metadata_sources
        return response
        #todo: add history to chat_history object

if (__name__=='main'):
    print('RAGdoll Chatterbox...')