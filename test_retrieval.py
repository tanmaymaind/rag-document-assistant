import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.answering import answer_question
from rag_assistant.store import DocumentStore


def make_store() -> DocumentStore:
    store = DocumentStore()
    store.add_document(
        "solar.txt",
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
    )
    store.add_document(
        "library.txt",
        "The campus library is open from 8 am to 10 pm on weekdays.",
    )
    return store


def test_retrieval_returns_correct_source_first():
    store = make_store()
    results = store.retrieve("How do solar panels convert sunlight?", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "solar.txt"
    assert 0.0 < results[0]["score"] <= 1.0


def test_retrieval_is_sorted_by_score():
    store = make_store()
    results = store.retrieve("library opening hours", top_k=2)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_question_and_empty_store():
    store = make_store()
    assert store.retrieve("", top_k=3) == []
    assert DocumentStore().retrieve("anything", top_k=3) == []


def test_answer_is_grounded_and_cited():
    store = make_store()
    result = answer_question(store, "How do solar panels convert sunlight?")
    assert result["found"] is True
    assert "photovoltaic" in result["answer"].lower()
    assert result["sources"][0]["source"] == "solar.txt"


def test_unrelated_question_says_not_found():
    store = make_store()
    result = answer_question(store, "What is the capital of France?")
    assert result["found"] is False
    assert "couldn't find" in result["answer"]
    assert result["sources"] == []
