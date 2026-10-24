import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rag_assistant.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("", "doc.txt") == []
    assert chunk_text("   \n  ", "doc.txt") == []


def test_short_text_is_one_chunk():
    chunks = chunk_text("one two three", "doc.txt")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "one two three"
    assert chunks[0]["source"] == "doc.txt"


def test_overlap_and_forward_progress():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), "doc.txt", chunk_size=200, overlap=40)
    assert len(chunks) == 3  # 0-199, 160-359, 320-499
    assert chunks[1]["text"].startswith("w160")
    # Reconstructed chunks must cover every word at least once.
    covered = set()
    for chunk in chunks:
        covered.update(chunk["text"].split())
    assert covered == set(words)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("hello world", "doc.txt", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello world", "doc.txt", chunk_size=10, overlap=10)
