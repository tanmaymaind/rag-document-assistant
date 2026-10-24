"""Streamlit UI for the RAG Document Assistant.

Run locally:
    streamlit run app/streamlit_app.py

The UI loads the sample documents, lets you add your own text, and
shows the grounded answer together with its source citations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.answering import answer_question  # noqa: E402
from rag_assistant.ingestion import load_directory  # noqa: E402
from rag_assistant.store import DocumentStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def get_store() -> DocumentStore:
    store = DocumentStore()
    store.add_documents(load_directory(ROOT / "data" / "sample_docs"))
    return store


st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
st.write(
    "Ask a question about the loaded documents. Answers come only from "
    "retrieved text, with sources shown — if the documents do not cover "
    "your question, the assistant says so instead of guessing."
)

store = get_store()

with st.sidebar:
    st.header("Documents")
    st.write(f"{store.document_count} documents, {store.chunk_count} chunks")
    for source in sorted({c['source'] for c in store.chunks}):
        st.write(f"- {source}")
    st.header("Add a document")
    new_source = st.text_input("Document name", placeholder="my-notes.txt")
    new_text = st.text_area("Document text", height=150)
    if st.button("Add document") and new_source.strip() and new_text.strip():
        store.add_document(new_source.strip(), new_text.strip())
        st.success(f"Added {new_source.strip()}")

question = st.text_input("Your question", placeholder="How do solar panels work?")
top_k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)

if st.button("Ask") and question.strip():
    result = answer_question(store, question, top_k=top_k)
    st.subheader("Answer")
    st.write(result["answer"])
    st.write(f"Confidence (best retrieval score): {result['confidence']}")
    if result["sources"]:
        st.subheader("Sources")
        for source in result["sources"]:
            st.write(f"- {source['source']} ({source['chunk_id']}, score {source['score']})")
