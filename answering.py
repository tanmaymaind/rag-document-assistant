"""Grounded answering: cite retrieved chunks, never invent an answer.

This is deliberately extractive, not generative. The answer is built
from the retrieved chunk text, and every claim can be traced to a
``source`` + ``chunk id``. If retrieval is weak, the honest answer is
"I couldn't find that in the documents" -- that failure mode is a
feature for a portfolio project, and it is covered by a test.
"""

from __future__ import annotations

DEFAULT_THRESHOLD = 0.10


def _best_sentences(chunk_text: str, question: str, limit: int = 2) -> list[str]:
    """Pick sentences from a chunk that share words with the question."""
    question_words = {
        word.strip("?,.!:;\"'").lower()
        for word in question.split()
        if len(word) > 3
    }
    sentences = [s.strip() for s in chunk_text.replace("\n", " ").split(". ") if s.strip()]
    scored = []
    for sentence in sentences:
        words = {w.strip("?,.!:;\"'").lower() for w in sentence.split()}
        scored.append((len(words & question_words), sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [sentence for overlap, sentence in scored if overlap > 0][:limit]
    if not picked and sentences:
        picked = [sentences[0]]
    return picked


def answer_question(
    store,
    question: str,
    top_k: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Answer ``question`` from ``store``, with citations.

    Returns a dict with:
      - ``answer``: grounded text, or a "not found" message
      - ``sources``: ``[{source, chunk_id, score}]`` for cited chunks
      - ``confidence``: best retrieval score, or 0.0 when nothing matched
      - ``found``: True only when a chunk cleared ``threshold``
    """
    if not question.strip():
        return {
            "answer": "Please ask a question.",
            "sources": [],
            "confidence": 0.0,
            "found": False,
        }

    retrieved = store.retrieve(question, top_k=top_k)
    if not retrieved or retrieved[0]["score"] < threshold:
        return {
            "answer": (
                "I couldn't find that in the uploaded documents. "
                "Try rephrasing the question, or add a document that covers it."
            ),
            "sources": [],
            "confidence": float(retrieved[0]["score"]) if retrieved else 0.0,
            "found": False,
        }

    cited = [chunk for chunk in retrieved if chunk["score"] >= threshold]
    parts: list[str] = []
    for chunk in cited:
        for sentence in _best_sentences(chunk["text"], question):
            parts.append(sentence if sentence.endswith(".") else sentence + ".")
    return {
        "answer": " ".join(parts),
        "sources": [
            {
                "source": chunk["source"],
                "chunk_id": chunk["id"],
                "score": round(chunk["score"], 4),
            }
            for chunk in cited
        ],
        "confidence": round(cited[0]["score"], 4),
        "found": True,
    }
