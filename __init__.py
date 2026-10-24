"""RAG Document Assistant: offline retrieval over your own documents.

The default retriever is TF-IDF + cosine similarity, so the project runs
with no API key and no external model download. The ``DocumentStore``
interface is deliberately small so a dense embedder (for example
sentence-transformers) can replace TF-IDF later without changing the
API, the UI, or the tests.
"""

from .answering import answer_question
from .chunking import chunk_text
from .store import DocumentStore

__all__ = ["DocumentStore", "answer_question", "chunk_text"]
__version__ = "0.1.0"
