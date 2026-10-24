"""Retrieval store: TF-IDF vectors over chunks + cosine similarity.

Design notes
------------
* The vectorizer is fitted once per ``rebuild()``. Adding documents
  calls ``rebuild()`` again, which keeps the API simple and correct
  for a personal-scale corpus (hundreds of documents, not millions).
  A production version would use an incremental vector index.
* Scores are cosine similarities in [0, 1] for TF-IDF vectors, so a
  threshold has a stable meaning across questions.
* Retrieval never generates text. Generation / summarisation lives in
  ``answering.py``, so a wrong retrieval cannot silently become a
  confident-sounding hallucinated answer.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import chunk_text


class DocumentStore:
    """An in-memory collection of chunks with TF-IDF retrieval."""

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    # -- ingestion -----------------------------------------------------
    def add_document(self, source: str, text: str) -> int:
        """Chunk one document and rebuild the index. Returns chunk count."""
        new_chunks = chunk_text(text, source, self.chunk_size, self.overlap)
        self.chunks.extend(new_chunks)
        self._rebuild()
        return len(new_chunks)

    def add_documents(self, documents: list[tuple[str, str]]) -> int:
        """Add several documents with a single index rebuild."""
        total = 0
        for source, text in documents:
            new_chunks = chunk_text(text, source, self.chunk_size, self.overlap)
            self.chunks.extend(new_chunks)
            total += len(new_chunks)
        self._rebuild()
        return total

    def _rebuild(self) -> None:
        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    # -- retrieval -----------------------------------------------------
    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the ``top_k`` chunks most similar to ``question``.

        Each result is a chunk dict plus a ``score`` (cosine similarity).
        An empty question, an empty store, or ``top_k <= 0`` returns [].
        Results are sorted by score, highest first.
        """
        if not question.strip() or not self.chunks or top_k <= 0:
            return []
        assert self._vectorizer is not None and self._matrix is not None
        query_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for idx in ranked[:top_k]:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    @property
    def document_count(self) -> int:
        return len({chunk["source"] for chunk in self.chunks})

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
