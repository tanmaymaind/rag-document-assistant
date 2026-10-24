# RAG Document Assistant

Ask questions over **your own documents** and get answers that are grounded in the retrieved text, with citations — not confident guesses.

Built as a portfolio project for AI / Data Science roles. It demonstrates retrieval-augmented generation (RAG) end to end: ingestion, chunking, retrieval, grounded answering, a REST API, a web UI, evaluation, and automated tests.

## Why this is different from a generic chatbot demo

- **No API key, no external model download.** Retrieval uses TF-IDF + cosine similarity (scikit-learn), so it runs fully offline.
- **Grounded by design.** Answers are built from retrieved chunks and cite `source + chunk id`. If nothing relevant is retrieved, the assistant says it could not find the answer instead of hallucinating.
- **Swappable retriever.** `DocumentStore` is a small interface, so a dense embedder (e.g. sentence-transformers) can replace TF-IDF later without changing the API, UI, or tests.
- **Tested and evaluated**, not just demoed (see Results below).

## Features

- Ingest `.txt`, `.md`, and `.pdf` documents (PDF via `pypdf`)
- Overlapping word-window chunking (200 words, 40-word overlap by default)
- TF-IDF retrieval with cosine similarity scores
- Extractive, cited answering with a configurable relevance threshold
- FastAPI service with interactive docs at `/docs`
- Streamlit UI: ask questions, add documents, inspect sources
- Sample documents + a 6-question QA set for reproducible evaluation
- GitHub Actions CI (tests + evaluation on every push)

## Project structure

```
rag-document-assistant/
├── src/rag_assistant/
│   ├── chunking.py      # overlapping word-window chunking
│   ├── ingestion.py     # .txt / .md / .pdf loading
│   ├── store.py         # TF-IDF DocumentStore + cosine retrieval
│   └── answering.py     # grounded, cited answering + not-found fallback
├── app/
│   ├── api.py           # FastAPI: /health, /documents, /ask
│   └── streamlit_app.py # Streamlit UI
├── data/
│   ├── sample_docs/     # 3 original sample documents
│   └── sample_qa.json   # 6-question evaluation set
├── scripts/evaluate.py  # top-3 retrieval hit rate on the sample set
├── tests/               # chunking, retrieval, answering, and API tests
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the tests
PYTHONPATH=src:. pytest -q

# Evaluate retrieval on the sample QA set
PYTHONPATH=src python scripts/evaluate.py

# Run the API (http://127.0.0.1:8000/docs)
uvicorn app.api:app --reload

# Run the UI
streamlit run app/streamlit_app.py
```

## API examples

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do solar panels convert sunlight into electricity?"}'
```

Example response (from the sample documents):

```json
{
  "answer": "Solar panels convert sunlight into electricity using photovoltaic cells.",
  "sources": [
    {"source": "solar-energy-faq.txt", "chunk_id": "solar-energy-faq.txt#chunk-0", "score": 0.45}
  ],
  "confidence": 0.45,
  "found": true
}
```

## Results (measured, not claimed)

Run in this repository's environment (Python 3.12, scikit-learn 1.9.1):

- **Tests: 13 passed, 0 failed** (`pytest -q`)
- **Retrieval evaluation: 6/6 top-3 hits = 1.00** on the included 6-question sample set (`scripts/evaluate.py`)

Scope note: the sample set is deliberately small and topical, so 1.00 is a sanity check that retrieval works on the sample corpus — not a claim about general accuracy. Add your own documents and questions to `data/sample_qa.json` to evaluate on a harder set.

## Limitations

- TF-IDF is keyword-based: it handles paraphrase and synonyms worse than dense embeddings. That trade-off is intentional for an offline, dependency-light baseline.
- The store is in-memory and rebuilt on every document add, so it suits personal-scale corpora (hundreds of documents), not millions.
- Answers are extractive, so phrasing follows the source documents rather than being rewritten in fluent prose.
- No user authentication: do not expose the API publicly without adding auth.

## Future improvements

- Dense embeddings (sentence-transformers) with a vector index (FAISS/Chroma)
- LLM-based answer synthesis over the retrieved chunks, with citations retained
- Persistent storage, document deletion, and multi-user workspaces
- Chunking by sentence/paragraph boundaries and hybrid (keyword + dense) retrieval
- Larger, human-labelled evaluation set with answer-quality scoring

