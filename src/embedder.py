"""
embedder.py
-----------
Thin wrapper around OpenAI's embedding API.

Keeps all embedding concerns in one place so that swapping the model
(e.g. text-embedding-3-small → text-embedding-3-large) or the provider
requires changes in exactly one file.

Why a wrapper instead of calling OpenAIEmbeddings directly in the indexer?
  - Centralises the model name and dimension config
  - Makes it easy to add batching, retry logic, or cost logging later
  - The indexer stays focused on Chroma operations, not API details
"""

import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def get_embedder() -> OpenAIEmbeddings:
    """
    Returns a configured OpenAIEmbeddings instance.

    Model is read from the OPENAI_EMBEDDING_MODEL env var, defaulting to
    text-embedding-3-small (1536 dimensions, best cost/performance ratio
    for semantic similarity tasks like this one).

    Raises EnvironmentError if OPENAI_API_KEY is not set — better to
    fail fast here than get a cryptic error mid-ingestion.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    print(f"  Embedder: {model}")
    return OpenAIEmbeddings(model=model)
