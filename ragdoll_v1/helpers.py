import colorlog
import logging
import re
import pickle
import os
import pyperclip
import importlib
import sys
from contextlib import contextmanager
import warnings

class dotDict(dict):
    """
    A dictionary subclass that allows attribute-style access to its elements.
    Usage:
    d = dotDict({'key': 'value'})
    print(d.key)  # prints 'value'
    """
    __getattr__= dict.get
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

def set_logger(loglevel=logging.WARN, include_time_stamp=False):
    msg_format = '%(log_color)s[%(module)s] %(asctime)s: %(message)s' if include_time_stamp else '%(log_color)s[%(module)s] %(message)s'
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        msg_format,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    ))

    logging.basicConfig(level=loglevel, handlers=[handler])
    return None

def is_notebook(print_output=False):
    """Checks if the code is running in a Jupyter Notebook environment.
    
    Returns:
        bool: True if running in a Jupyter Notebook or JupyterLab, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__  # Attempt to access IPython shell
        if shell == "ZMQInteractiveShell":
            if (print_output):
                print('Running in a Jupyter Notebook or JupyterLab environment.')
            return True  # Running in Jupyter Notebook or JupyterLab
        elif shell == "TerminalInteractiveShell":
            if (print_output):
                print('Running in a terminal-based IPython session.')
            return False  # Running in a terminal-based IPython session
        else:
            if (print_output):
                print('Probably running in a standard Python environment.')
            return False  # Probably running in a standard Python environment
    except NameError:
        if (print_output):
                print('Not running within an IPython environment.')
        return False  # Not running within an IPython environment

def pretty_print_docs(docs, top_n=None, for_llm=True):
    """Formats and prints the metadata and content of a list of documents. 
    Useful for creating context for an LLM input RAG process

    Args:
        docs (list): A list of langchain documents.
        top_n (int, optional): The number of documents to print. Defaults to all documents.
        for_llm (bool, optional): Indicates if the output is for an LLM input RAG process.  
                                  if True, the output excludes the 100 character divider to save tokens. Defaults to True.
    Returns:
        str: The formatted string containing the metadata and content of the documents.
    """
    top_n = len(docs) if top_n is None else top_n
    divider = '' if (for_llm) else '-' * 100+'\n'
                              
    return f"\n{divider}".join(f"Source: {d.metadata.get('source')}\n"
                        f"Title: {d.metadata.get('title')}\n"
                        f"Content: {d.page_content}\n"
                        for i, d in enumerate(docs) if i < top_n)

def pretty_print_dict(dictionary, indent=''):
    result = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result += f'\033[1m{indent}{key}\033[0m :\n'
            result += pretty_print_dict(value, indent + '  ')
        else:
            result += f'\033[1m{indent}{key}\033[0m : {value}\n'
    return result

def remove_set_duplicates(results, key='link', log=False):
    """
    Removes duplicate links from a list of dictionaries.

    Args:
        results (list): A list of dictionaries containing 'link' key.
        key (str, optional): The key to check for duplicates. Defaults to 'link'.

    Returns:
        list: A new list of dictionaries with duplicate links removed.
    """
    seen = set()
    output = []
    for d in results:
        if d[key] not in seen:
            seen.add(d[key])
            output.append(d)
    if (log):
        print(f"Removed {len(results) - len(output)} duplicate links from {len(output)}")
    return output


def parse_response_ByTag(response, tag='output'):
    """
    Parses the response string and extracts the content within the specified tag.

    Args:
        response (str): The response string to parse.
        tag (str, optional): The tag to search for within the response string. Defaults to 'output'.

    Returns:
        str: The content within the specified tag, or None if the tag is not found.
    """
    # Regular expression patterns to match thinking and section_content tags
    tag_pattern = f'<{tag}>(.*?)</{tag}>'
    # Find the thinking and section_content blocks
    tag_match = re.search(tag_pattern, response, re.DOTALL)
    # Extract the content 
    output = tag_match.group(1).strip() if tag_match else None
    return output


def parse_response(response, thinking_tag='thinking', output_tag='output', additional_props={}):
    """
    Parses the response string and extracts the thinking and output content.

    Args:
        response (str): The response string to parse.
        thinking_tag (str, optional): The tag used to identify the thinking content. Defaults to 'thinking'.
        output_tag (str, optional): The tag used to identify the output content. Defaults to 'output'.
        additional_props (dict, optional): Additional properties to include in the parsed data. Defaults to {}.

    Returns:
        dict: A dictionary containing the parsed data, including the thinking and output content.
    """
    # Regular expression patterns to match thinking and output tags
    thinking_pattern = f'<{thinking_tag}>(.*?)</{thinking_tag}>'
    topic_content_pattern = f'<{output_tag}>(.*?)</{output_tag}>'
    
    # Find the thinking and output blocks
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)
    topic_content_match = re.search(topic_content_pattern, response, re.DOTALL)
    
    # Extract the thinking and output 
    thinking = thinking_match.group(1).strip() if thinking_match else None
    output_content = topic_content_match.group(1).strip() if topic_content_match else None
    
    # Create a dictionary to store the extracted data
    parsed_data = {}
    parsed_data[thinking_tag] = thinking
    parsed_data[output_tag] = output_content
    
    
    result = {**parsed_data, **additional_props} if bool(additional_props) else parsed_data
    return result

def has_extension(filename):
    """
    Check if a filename has an extension.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the filename has an extension, False otherwise.
    """
    _, file_extension = os.path.splitext(filename)
    return bool(file_extension)

def save_dict_to_pickle(filename, data, default_extension='article'):
    """
    Save a dictionary to a file using pickle serialization.

    Args:
        filename (str): The filename to save the dictionary to.
        data (dict): The dictionary to save.
        default_extension (str, optional): The default extension to use if the filename doesn't have one. Defaults to 'article'.

    Raises:
        Exception: If an error occurs during file saving.
    """
    if not has_extension(filename):
        filename = f'{filename}.{default_extension}'
    try:
        with open(filename, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
    except Exception as e:
        print(f"Error occurred while saving file '{filename}': {e}")

def load_dict_from_pickle(filename, default_extension='article'):
    """
    Load a dictionary from a file using pickle deserialization.

    Args:
        filename (str): The filename to load the dictionary from.
        default_extension (str, optional): The default extension to use if the filename doesn't have one. Defaults to 'article'.

    Returns:
        dict: The loaded dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(filename):
        filename = f'{filename}.{default_extension}' if not has_extension(filename) else filename
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def dictionary_from_keys(original_dict, keys_to_retrieve):
    """
    Create a new dictionary containing only the specified keys from the original dictionary.

    Args:
        original_dict (dict): The original dictionary.
        keys_to_retrieve (list): A list of keys to retrieve from the original dictionary.

    Returns:
        dict: A new dictionary containing only the specified keys and their corresponding values from the original dictionary.
    """
    new_dict = {}
    for key in keys_to_retrieve:
        if key in original_dict:
            new_dict[key] = original_dict[key]
    return new_dict

def dictionary_except_keys(original_dict, keys_to_exlude):
    """
    Create a new dictionary containing only the specified keys from the original dictionary that arent in the keys_to_exlude.

    Args:
        original_dict (dict): The original dictionary.
        keys_to_exlude (list): A list of keys to exclude from the original dictionary.

    Returns:
        dict: A new dictionary containing only the specified keys and their corresponding values from the original dictionary.
    """
    new_dict = {}
    for key in original_dict.keys():
        if key not in keys_to_exlude:
            new_dict[key] = original_dict[key]
    return new_dict


def reload_module(module):
    """
    Function that reloads a python module. Useful in development when changes 
    are made to a module and those changes need to be reloaded in the interpreter.

    Parameters:
    module (module): The module that needs to be reloaded.

    Returns: 
    None
    """
    # Check if the string contains a period
    importlib.reload(module)


def to_clipboard(object):
    """
    Function that copies an object to the system clipboard. 

    Parameters:
    object (str): The string that needs to be copied to the clipboard.
    
    Returns: 
    None
    """
    pyperclip.copy(object)

def remove_numbering(string):
    """
    Removes numbering from the beginning of a string.

    Args:
        string (str): The input string with numbering.

    Returns:
        str: The string with numbering removed.

    Example:
        >>> remove_numbering("1. Hello World")
        "Hello World"
    """
    return re.sub("^\d+\.\s+", "", string)


@contextmanager
def suppress_print():
    """Context manager to temporarily suppress print statements.
    # Example usage:
    def function_with_print():
        print("This is some output from function_with_print")

    # Turning off print
    with suppress_print():
        function_with_print()  # This print statement will be suppressed

    function_with_print()  # This print statement will not be suppressed
    """
    original_stdout = sys.stdout
    sys.stdout = open("/dev/null", "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

@contextmanager
def suppress_warnings():
    """Context manager to temporarily suppress warnings.
    
    # Example usage:
    def function_with_warning():
        warnings.warn("This is a warning from function_with_warning")

    # Turning off warnings
    with suppress_warnings():
        function_with_warning()  # This warning will be suppressed

    function_with_warning()  # This warning will not be suppressed
    """
    original_filters = warnings.filters[:]
    warnings.simplefilter("ignore")
    try:
        yield
    finally:
        warnings.filters = original_filters

if __name__=='__main__':
    check_notebook = is_notebook(print_output=True)
    