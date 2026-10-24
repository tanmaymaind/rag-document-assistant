"""Word-window chunking with overlap.

Chunking matters for RAG: chunks that are too long dilute retrieval,
chunks that are too short lose context. A sliding word window is a
transparent, reproducible baseline. Sentence-aware splitting can be
added later, but it should not be needed to understand this code.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """Split ``text`` into overlapping word-window chunks.

    Args:
        text: Document text. Empty / whitespace-only text yields no chunks.
        source: Human-readable document name, kept on every chunk so an
            answer can always cite where it came from.
        chunk_size: Words per chunk. Must be positive.
        overlap: Words shared with the previous chunk. Must be smaller
            than ``chunk_size`` so the window always moves forward.

    Returns:
        A list of ``{"id", "source", "text"}`` dicts. Chunk ids are
        ``f"{source}#chunk-{i}"`` and are stable for a given input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[dict] = []
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(
            {
                "id": f"{source}#chunk-{i}",
                "source": source,
                "text": " ".join(window),
            }
        )
        if start + chunk_size >= len(words):
            break
    return chunks
