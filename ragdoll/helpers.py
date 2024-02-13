import colorlog
import logging

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

if __name__=='__main__':
    check_notebook = is_notebook(print_output=True)
    