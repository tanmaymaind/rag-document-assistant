import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)


def test_health_reports_sample_documents():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["documents"] >= 3


def test_ask_returns_grounded_answer():
    response = client.post(
        "/ask", json={"question": "How do solar panels convert sunlight into electricity?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["sources"][0]["source"] == "solar-energy-faq.txt"


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_add_document_then_ask():
    response = client.post(
        "/documents",
        json={"source": "test-note.txt", "text": "The blue test widget costs forty-two rupees and ships on Mondays."},
    )
    assert response.status_code == 200
    assert response.json()["chunks_added"] >= 1
    answer = client.post(
        "/ask", json={"question": "How much does the blue test widget cost?"}
    ).json()
    assert answer["found"] is True
