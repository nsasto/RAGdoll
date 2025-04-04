import os

def create_empty_file(filepath):
    """Creates an empty file at the specified filepath."""
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))  # Create directories if needed
    open(filepath, 'w').close()

# Example usage:
# create_empty_file("path/to/your/empty_file.txt")