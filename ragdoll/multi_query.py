# Description: This file contains a custom implementation of the langchain MultiQueryRetriever class, 
# which is a retriever that uses an LLM to write a set of queries and retrieve documents for each query. 
# It then returns the unique union of all retrieved documents. Langhchain's implementation seems is temperamental and currently has a bug
# that causes it to return an empty list of documents or simply bomb out with a pydantic dic error. This custom implementation is a workaround for that bug.
# if it's working - feel free to switch back :)

import asyncio
import logging
from typing import List, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever

from langchain.chains.llm import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

logger = logging.getLogger(__name__)



# Default prompt
DEFAULT_QUERY_PROMPT =  PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate {alternative_count} 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.\n 
    Your response should be a list of comma separated values, eg: `foo, bar, baz`. IMPORTANT: Do not use line numbering or new line characters in your response.
    Original question: {question}""",
)

def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


class MultiQueryRetriever(BaseRetriever):
    """Given a query, use an LLM to write a set of queries.

    Retrieve docs for each query. Return the unique union of all retrieved docs.
    """

    retriever: BaseRetriever
    llm_chain: LLMChain
    verbose: bool = True
    parser_key: str = "text"
    include_original: bool = False
    """Whether to include the original query in the list of generated queries."""
    alternative_count: int = 3
    """Number of alternative queries to generate."""

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt: PromptTemplate = DEFAULT_QUERY_PROMPT,
        parser_key: str = "text",
        include_original: bool = False,
        alternative_count: int = 3,
    ) -> "MultiQueryRetriever":
        """Initialize from llm using default template.

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation using DEFAULT_QUERY_PROMPT
            include_original: Whether to include the original query in the list of
                generated queries.

        Returns:
            MultiQueryRetriever
        """
        output_parser = CommaSeparatedListOutputParser()
        prompt = prompt.partial(alternative_count=alternative_count)
        llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            parser_key=parser_key,
            include_original=include_original,
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            question: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = await self.agenerate_queries(query, run_manager)
        print(queries)
        if self.include_original:
            queries.append(query)
        documents = await self.aretrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    async def agenerate_queries(
        self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = await self.llm_chain.invoke(
            inputs={"question": question}, callbacks=run_manager.get_child()
        )
        lines = response.get(self.parser_key, [])
        if (self.verbose | logger.getEffectiveLevel() == logging.INFO):
            logger.info(f"Generated queries: {lines}")
        return lines

    async def aretrieve_documents(
        self, queries: List[str], run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        document_lists = await asyncio.gather(
            *(
                self.retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
                for query in queries
            )
        )
        return [doc for docs in document_lists for doc in docs]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            question: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = self.generate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        documents = self.retrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    def generate_queries(
        self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = self.llm_chain.invoke(
            {"question": question}, callbacks=run_manager.get_child()
        )
        #lines = text value from the response dictionary
        
        lines = response.get(self.parser_key, [])
        if (self.verbose | logger.getEffectiveLevel() == logging.INFO):
            logger.info(f"Generated queries: {lines}")
        return lines

    def retrieve_documents(
        self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
            documents.extend(docs)
        return documents

    def unique_union(self, documents: List[Document]) -> List[Document]:
        """Get unique Documents.

        Args:
            documents: List of retrieved Documents

        Returns:
            List of unique retrieved Documents
        """
        return _unique_documents(documents)
