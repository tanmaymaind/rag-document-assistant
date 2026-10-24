"""Evaluate retrieval on the sample QA set.

Metric: top-3 hit rate -- the share of questions whose expected source
appears in the top 3 retrieved chunks. This is a retrieval metric, not
a claim about answer quality, and it is only meaningful on this small
sample set. Run: python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.ingestion import load_directory  # noqa: E402
from rag_assistant.store import DocumentStore  # noqa: E402


def main() -> dict:
    store = DocumentStore()
    store.add_documents(load_directory(ROOT / "data" / "sample_docs"))
    qa = json.loads((ROOT / "data" / "sample_qa.json").read_text(encoding="utf-8"))

    details = []
    hits = 0
    for item in qa:
        retrieved = store.retrieve(item["question"], top_k=3)
        sources = [chunk["source"] for chunk in retrieved]
        hit = item["expected_source"] in sources
        hits += int(hit)
        details.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": sources,
                "hit": hit,
            }
        )

    report = {
        "questions": len(qa),
        "hits": hits,
        "top3_hit_rate": round(hits / len(qa), 4) if qa else 0.0,
        "details": details,
    }
    out = ROOT / "reports" / "evaluation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Top-3 hit rate: {hits}/{len(qa)} = {report['top3_hit_rate']}")
    print(f"Report written to {out}")
    return report


if __name__ == "__main__":
    main()
