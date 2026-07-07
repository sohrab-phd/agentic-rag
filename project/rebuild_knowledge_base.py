"""Rebuild knowledge base after PDF extraction fix. Place source PDFs in docs/ first."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory

QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_after_fix.json"


def _retrieval_snapshot(collection, query: str) -> dict:
    tools = ToolFactory(collection).create_tools()
    search_tool = next(t for t in tools if t.name == "search_child_chunks")
    no_thresh = collection.similarity_search_with_score(query, k=10, score_threshold=None)
    with_thresh = collection.similarity_search_with_score(query, k=10, score_threshold=0.7)
    return {
        "top10_no_threshold": [
            {
                "score": float(s),
                "parent_id": d.metadata.get("parent_id"),
                "source": d.metadata.get("source"),
                "text": d.page_content,
            }
            for d, s in no_thresh
        ],
        "threshold_0_7_count": len(with_thresh),
        "tool_output": search_tool.invoke({"query": query, "limit": 7}),
    }


def main() -> None:
    before_path = ROOT / "_audit_step2.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {}

    rag = RAGSystem()
    rag.initialize()
    doc_manager = DocumentManager(rag)

    pdf_count = len(list(Path(config.DOCS_DIR).glob("*.pdf")))
    if pdf_count:
        print(f"Re-extracting {pdf_count} PDF(s) from docs/ ...")
        doc_manager.reextract_from_docs()
    else:
        print("No PDFs in docs/ — re-indexing existing markdown only.")
        doc_manager.reindex_all()

    collection = rag.vector_db.get_collection(rag.collection_name)
    after = _retrieval_snapshot(collection, QUERY)

    report = {
        "query": QUERY,
        "pdf_extractor": config.PDF_EXTRACTOR,
        "before": {
            "top10": before.get("no_threshold", []),
            "tool_output": before.get("tool_output_limit_7"),
            "threshold_0_7_count": before.get("retrieval_with_threshold_0_7", {}).get("result_count"),
        },
        "after": after,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
