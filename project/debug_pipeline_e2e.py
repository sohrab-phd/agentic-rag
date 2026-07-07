"""
End-to-end RAG pipeline debugger.
Writes evidence to ../e2e_audit_report.json and ../e2e_audit_report.md
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from document_chunker import DocumentChuncker
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rag_agent.nodes import rewrite_query, orchestrator, compress_context, fallback_response, collect_answer
from rag_agent.prompts import get_orchestrator_prompt, get_rewrite_query_prompt
from rag_agent.tools import ToolFactory
from utils import estimate_context_tokens

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "e2e_audit_report.json"
OUT_MD = ROOT / "e2e_audit_report.md"

# Persian question grounded in the indexed document topic (information extraction / relation extraction).
USER_QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"


def _np_stats(vec: list[float]) -> dict:
    arr = np.array(vec, dtype=np.float64)
    return {
        "dimension": int(arr.size),
        "l2_norm": float(np.linalg.norm(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "first_5": [float(x) for x in arr[:5]],
    }


def _copy_qdrant_for_readonly() -> str:
    src = Path(config.QDRANT_DB_PATH)
    if not src.exists():
        raise FileNotFoundError(f"Qdrant path missing: {src}")
    tmp = tempfile.mkdtemp(prefix="qdrant_audit_")
    shutil.copytree(src, Path(tmp) / "qdrant_db")
    return str(Path(tmp) / "qdrant_db")


def _md_section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def main() -> None:
    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user_query": USER_QUERY,
        "config": {
            "dense_model": config.DENSE_MODEL,
            "sparse_model": config.SPARSE_MODEL,
            "llm_model": config.LLM_MODEL,
            "child_chunk_size": config.CHILD_CHUNK_SIZE,
            "child_chunk_overlap": config.CHILD_CHUNK_OVERLAP,
            "min_parent_size": config.MIN_PARENT_SIZE,
            "max_parent_size": config.MAX_PARENT_SIZE,
            "score_threshold_in_tools": 0.7,
            "retrieval_mode": "HYBRID (dense + BM25 sparse, RRF fusion)",
        },
    }

    # --- Markdown / chunking evidence ---
    md_files = sorted(Path(config.MARKDOWN_DIR).glob("*.md"))
    report["indexed_markdown_files"] = [f.name for f in md_files]

    chunker = DocumentChuncker()
    chunk_report = []
    for md in md_files:
        parents, children = chunker.create_chunks_single(md)
        chunk_report.append(
            {
                "file": md.name,
                "parent_count": len(parents),
                "child_count": len(children),
                "sample_children": [
                    {
                        "index": i,
                        "length": len(c.page_content),
                        "parent_id": c.metadata.get("parent_id"),
                        "preview": c.page_content[:300],
                    }
                    for i, c in enumerate(children[:5])
                ],
            }
        )
    report["chunking"] = chunk_report

    # --- Vector DB (readonly copy to avoid lock) ---
    qdrant_copy = _copy_qdrant_for_readonly()
    orig_path = config.QDRANT_DB_PATH
    config.QDRANT_DB_PATH = qdrant_copy
    try:
        vdb = VectorDbManager()
        coll = vdb.get_collection(config.CHILD_COLLECTION)
        client = vdb._VectorDbManager__client
        point_count = client.count(config.CHILD_COLLECTION).count
        report["vector_store"] = {
            "collection": config.CHILD_COLLECTION,
            "point_count": point_count,
            "document_count_markdown": len(md_files),
        }

        dense = vdb._VectorDbManager__dense_embeddings
        sparse = vdb._VectorDbManager__sparse_embeddings

        # --- Query rewrite (preprocessing) ---
        llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        rewrite_state = {
            "messages": [HumanMessage(content=USER_QUERY)],
            "conversation_summary": "",
        }
        rewrite_out = rewrite_query(rewrite_state, llm)
        rewritten = rewrite_out.get("rewrittenQuestions", [])
        report["query_preprocessing"] = {
            "rewrite_prompt_excerpt": get_rewrite_query_prompt()[:500] + "...",
            "is_clear": rewrite_out.get("questionIsClear"),
            "rewritten_questions": rewritten,
            "clarification": rewrite_out.get("messages", [{}])[0].content
            if not rewrite_out.get("questionIsClear")
            else None,
        }
        search_query = rewritten[0] if rewritten else USER_QUERY
        report["search_query_used"] = search_query

        # --- Embeddings ---
        dense_vec = dense.embed_query(search_query)
        sparse_vec = sparse.embed_query(search_query)
        report["query_embedding"] = {
            "dense_model": config.DENSE_MODEL,
            "dense_vector_stats": _np_stats(dense_vec),
            "sparse_token_count": len(sparse_vec.indices),
            "sparse_top_tokens": [
                {"index": int(i), "weight": float(v)}
                for i, v in zip(sparse_vec.indices[:15], sparse_vec.values[:15])
            ],
        }

        # --- Retrieval: top 10 with scores, no threshold ---
        top10 = coll.similarity_search_with_score(search_query, k=10, score_threshold=None)
        retrieved = []
        for rank, (doc, score) in enumerate(top10, 1):
            retrieved.append(
                {
                    "rank": rank,
                    "rrf_or_fusion_score": float(score),
                    "parent_id": doc.metadata.get("parent_id"),
                    "source": doc.metadata.get("source"),
                    "chunk_length": len(doc.page_content),
                    "chunk_text": doc.page_content,
                }
            )
        report["retrieval_top10_no_threshold"] = retrieved

        # --- Retrieval with production score_threshold=0.7 ---
        throttled = coll.similarity_search_with_score(search_query, k=10, score_threshold=0.7)
        report["retrieval_with_threshold_0_7"] = {
            "result_count": len(throttled),
            "scores": [float(s) for _, s in throttled],
            "chunks": [
                {
                    "score": float(s),
                    "parent_id": d.metadata.get("parent_id"),
                    "source": d.metadata.get("source"),
                    "text_preview": d.page_content[:200],
                }
                for d, s in throttled
            ],
        }

        # --- Tool path (what agent actually uses) ---
        tools = ToolFactory(coll).create_tools()
        search_tool = next(t for t in tools if t.name == "search_child_chunks")
        tool_output = search_tool.invoke({"query": search_query, "limit": 7})
        report["tool_search_output"] = tool_output

        # --- Parent retrieval for top hit ---
        parent_store = ParentStoreManager()
        if retrieved:
            top_parent = retrieved[0]["parent_id"]
            parent = parent_store.load_content(top_parent)
            report["top_parent_full_text"] = {
                "parent_id": top_parent,
                "length": len(parent.get("content", "")),
                "content": parent.get("content", ""),
            }

        # --- Full agent run for one rewritten question ---
        llm_tools = llm.bind_tools(tools)
        agent_state = {
            "question": search_query,
            "question_index": 0,
            "messages": [],
            "context_summary": "",
            "retrieval_keys": set(),
            "tool_call_count": 0,
            "iteration_count": 0,
        }

        prompts_trace = []
        final_messages = []

        for step in range(6):
            orch_out = orchestrator(agent_state, llm_tools)
            agent_state["messages"].extend(orch_out.get("messages", []))
            agent_state["tool_call_count"] = agent_state.get("tool_call_count", 0) + orch_out.get(
                "tool_call_count", 0
            )
            agent_state["iteration_count"] = agent_state.get("iteration_count", 0) + orch_out.get(
                "iteration_count", 0
            )

            last = agent_state["messages"][-1]
            prompts_trace.append(
                {
                    "step": step,
                    "orchestrator_system_prompt_excerpt": get_orchestrator_prompt()[:400] + "...",
                    "messages_to_llm": [
                        {
                            "type": type(m).__name__,
                            "content": (m.content[:2000] + "...") if m.content and len(m.content) > 2000 else m.content,
                            "tool_calls": getattr(m, "tool_calls", None),
                        }
                        for m in [SystemMessage(content=get_orchestrator_prompt())]
                        + ([HumanMessage(content=f"[زمینه فشرده]\n{agent_state.get('context_summary','')}" )] if agent_state.get("context_summary") else [])
                        + agent_state["messages"]
                    ],
                }
            )

            if not getattr(last, "tool_calls", None):
                break

            for tc in last.tool_calls:
                tool_fn = next(t for t in tools if t.name == tc["name"])
                result = tool_fn.invoke(tc["args"])
                from langchain_core.messages import ToolMessage

                agent_state["messages"].append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
                )

            if agent_state.get("iteration_count", 0) >= config.MAX_ITERATIONS:
                fb = fallback_response(agent_state, llm)
                agent_state["messages"].extend(fb.get("messages", []))
                break

        final_messages = agent_state["messages"]
        answer_out = collect_answer(agent_state)
        report["agent_trace"] = {
            "iterations": agent_state.get("iteration_count"),
            "tool_calls": agent_state.get("tool_call_count"),
            "prompts_trace": prompts_trace,
            "final_messages": [
                {
                    "type": type(m).__name__,
                    "content": m.content,
                    "tool_calls": getattr(m, "tool_calls", None),
                    "name": getattr(m, "name", None),
                }
                for m in final_messages
            ],
            "context_token_estimate": estimate_context_tokens(final_messages),
            "token_threshold": config.BASE_TOKEN_THRESHOLD,
        }
        report["final_answer"] = answer_out.get("final_answer")

        # Dense-only vs hybrid comparison on same query
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        dense_only = QdrantVectorStore(
            client=client,
            collection_name=config.CHILD_COLLECTION,
            embedding=dense,
            retrieval_mode=RetrievalMode.DENSE,
        )
        dense_hits = dense_only.similarity_search_with_score(search_query, k=5, score_threshold=None)
        report["dense_only_top5_cosine"] = [
            {
                "cosine_score": float(s),
                "parent_id": d.metadata.get("parent_id"),
                "preview": d.page_content[:150],
            }
            for d, s in dense_hits
        ]

    finally:
        config.QDRANT_DB_PATH = orig_path
        shutil.rmtree(Path(qdrant_copy).parent, ignore_errors=True)

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown summary
    md_lines = [
        "# E2E RAG Pipeline Audit",
        f"\n**Timestamp (UTC):** {report['timestamp_utc']}",
        f"\n**User query:** {USER_QUERY}",
    ]
    md_lines.append(_md_section("Query after preprocessing", json.dumps(report["query_preprocessing"], ensure_ascii=False, indent=2)))
    md_lines.append(_md_section("Search query used", report["search_query_used"]))
    md_lines.append(_md_section("Query embedding (dense)", json.dumps(report["query_embedding"], ensure_ascii=False, indent=2)))
    md_lines.append(_md_section("Vector store", json.dumps(report["vector_store"], ensure_ascii=False, indent=2)))
    md_lines.append(_md_section("Top 10 retrieved chunks (no threshold)", json.dumps(report["retrieval_top10_no_threshold"], ensure_ascii=False, indent=2)))
    md_lines.append(_md_section("Retrieval with score_threshold=0.7 (production)", json.dumps(report["retrieval_with_threshold_0_7"], ensure_ascii=False, indent=2)))
    md_lines.append(_md_section("Tool search output (production path)", report.get("tool_search_output", "")))
    md_lines.append(_md_section("Final answer", report.get("final_answer", "")))
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
