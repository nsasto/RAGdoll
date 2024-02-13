from datetime import datetime

def generate_search_queries_prompt(query, query_count=3):
    """ Generates the search queries prompt for the given question.
    Args: query (str): The question to generate the search queries prompt for
          query_count (int): number of results to return. defaults to 3
    Returns: str: The search queries prompt for the given question
    """

    return f'Write exactly {query_count} unique google search queries to search online that form an objective opinion from the following: "{query}"' \
           f'Use the current date if needed: {datetime.now().strftime("%B %d, %Y")}.\n'\
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3", etc].'