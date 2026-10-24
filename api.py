"""FastAPI service for the RAG Document Assistant.

Run locally:
    uvicorn app.api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API docs.

On startup the service loads the sample documents in
``data/sample_docs`` so it works immediately. Use POST /documents to
add your own text documents at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.answering import answer_question  # noqa: E402
from rag_assistant.ingestion import load_directory  # noqa: E402
from rag_assistant.store import DocumentStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
store = DocumentStore()
sample_docs = load_directory(ROOT / "data" / "sample_docs")
store.add_documents(sample_docs)

app = FastAPI(
    title="RAG Document Assistant",
    description="Ask questions over your own documents. Answers are grounded in retrieved chunks, with citations.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question in plain English")
    top_k: int = Field(3, ge=1, le=10)


class DocumentRequest(BaseModel):
    source: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": store.document_count,
        "chunks": store.chunk_count,
    }


@app.get("/documents")
def list_documents() -> dict:
    return {
        "documents": sorted({chunk["source"] for chunk in store.chunks}),
        "chunk_count": store.chunk_count,
    }


@app.post("/documents")
def add_document(payload: DocumentRequest) -> dict:
    chunks_added = store.add_document(payload.source, payload.text)
    return {"source": payload.source, "chunks_added": chunks_added}


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")
    return answer_question(store, payload.question, top_k=payload.top_k)
