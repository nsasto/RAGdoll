from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent

install_requires = [
    # Core dependencies
    "numpy>=2.0.0",
    "python-dotenv>=1.0.1",
    "pyarrow>=19.0.1",
    "pandas>=2.2.3",
    "colored>=2.2.4",
    "colorlog>=6.8.2",
    "pyperclip>=1.8.2",
    # LangChain ecosystem
    "langchain>=0.3.23",
    "langchain-core>=0.3.59",
    "langchain-text-splitters>=0.3.8",
    "langchain_community>=0.3.21",
    "langchain_openai>=0.3.12",
    "langchain-google-community>=2.0.7",
    # Document processing
    "lxml>=5.1.0",
    "PyMuPDF>=1.25.5",
    "langchain-markitdown>=0.1.6",
    "docx>=0.2.4",
    # ML/AI dependencies
    "openai>=1.71.0",
    "google-api-python-client>=2.166.0",
    "sentence_transformers>=4.1.0",
    "faiss-cpu>=1.11.0",
]

setup(
    name="python-ragdoll",
    version="2.0.1",
    description="A set of helper classes that abstract some of the more common tasks of a typical RAG process including document loading/web scraping.",
    author="Nathan Sasto",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Markup :: Markdown',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    long_description=(this_directory / "README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    license='MIT',
)
