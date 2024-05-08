from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def generate_search_queries_prompt(query, query_count=3):
    """ Generates the search queries prompt for the given question.
    Args: query (str): The question to generate the search queries prompt for
          query_count (int): number of results to return. defaults to 3
    Returns: str: The search queries prompt for the given question
    """

    return f'Write exactly {query_count} unique google search queries to search online that form an objective opinion from the following: "{query}"' \
           f'Use the current date if needed: {datetime.now().strftime("%B %d, %Y")}.\n'\
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3", etc].'

def generate_RAG_template(report_format="apa", min_words=1000):
    """ Generates the report prompt for the given question and research summary.
    Args: report_format (str): report format
          min_words (int): minimum word count
    Returns: PromptTemplate: The report prompt template for the given question and research summary
    """

    template = "Use the following pieces of context to answer the question at the end." \
            "If you don't know the answer, just say that you don't know, don't try to make up an answer." \
            "Add supporting information but keep the answer as concise as possible." \
            "The response should focus on the answer to the query, should be well structured, informative," \
            f"in depth and comprehensive, with facts and numbers if available and a minimum of {min_words} words.\n" \
            "You must write the report with markdown syntax.\n " \
            "Use an unbiased and journalistic tone. \n" \
            "You MUST determine your own concrete and valid opinion based on the given information. Do NOT reach general and meaningless conclusions.\n" \
            f"You MUST write the report in {report_format} format.\n " \
            f"Assume that the current date is {datetime.now().strftime('%B %d, %Y')}" \
            "-Context Start-\n" \
            "{context}\n" \
            "-Context End-\n" \
            "Question: {question}\n" \
            "Helpful Answer:"

    return template


def generate_standalone_history_prompt(prompt=None):
      ### Contextualize question ###
      contextualize_q_system_prompt = """Given a chat history and the latest user question \
      which might reference context in the chat history, formulate a standalone question \
      which can be understood without the chat history. Do NOT answer the question, \
      just reformulate it if needed and otherwise return it as is.""" if prompt is None else prompt
      
      contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
      ]
      )

      return contextualize_q_prompt

def generate_context_chat_prompt(prompt=None):
     ### Answer question ###
      qa_system_prompt = """You are an assistant for question-answering tasks. \
      Use the following pieces of retrieved context to answer the question. \
      If you don't know the answer, just say that you don't have enough research to respond accurately, but give a high level description of what you can answer being \
      clear that this information is from your general knowledge. \
      Use the minimum number of sentences to give a complete answer while keeping the answer concise.\

      {context}""" if prompt is None else prompt

      qa_prompt = ChatPromptTemplate.from_messages(
      [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
      ]
      )

      return qa_prompt
