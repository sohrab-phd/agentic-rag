"""Trace answer-generation prompts for a Persian question."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core.rag_system import RAGSystem
from db.vector_db_manager import VectorDbManager
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from rag_agent.nodes import orchestrator, collect_answer, rewrite_query
from rag_agent.prompts import get_orchestrator_prompt
from rag_agent.tools import ToolFactory
from langgraph.prebuilt import ToolNode

QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
OUT = Path(__file__).resolve().parent.parent / "generation_trace.json"


def _msg_dict(m):
    return {
        "type": type(m).__name__,
        "content": m.content,
        "tool_calls": getattr(m, "tool_calls", None),
        "name": getattr(m, "name", None),
    }


def _copy_qdrant():
    src = Path(config.QDRANT_DB_PATH)
    tmp = Path(tempfile.mkdtemp()) / "qdrant_db"
    shutil.copytree(src, tmp)
    return tmp


def main():
    qdrant_copy = _copy_qdrant()
    orig = config.QDRANT_DB_PATH
    config.QDRANT_DB_PATH = str(qdrant_copy)
    report = {
        "query": QUERY,
        "model": config.LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "top_p": "not set in code (Ollama default, typically 0.9)",
    }
    try:
        rag = RAGSystem()
        rag.initialize()
        llm = rag.agent_graph.nodes["rewrite_query"].bound.func.__self__ if False else None
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        collection = rag.vector_db.get_collection(rag.collection_name)
        tools = ToolFactory(collection).create_tools()
        llm_tools = llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        # rewrite
        rewrite_out = rewrite_query({"messages": [HumanMessage(content=QUERY)], "conversation_summary": ""}, llm)
        search_q = rewrite_out.get("rewrittenQuestions", [QUERY])[0]
        report["rewritten_query"] = search_q

        # retrieval evidence
        search_tool = next(t for t in tools if t.name == "search_child_chunks")
        tool_raw = search_tool.invoke({"query": search_q, "limit": 7})
        top10 = collection.similarity_search_with_score(search_q, k=10, score_threshold=None)
        report["retrieved_chunks"] = [
            {"score": float(s), "parent_id": d.metadata.get("parent_id"), "source": d.metadata.get("source"), "text": d.page_content}
            for d, s in top10
        ]
        report["tool_output_to_llm"] = tool_raw

        # agent loop
        state = {
            "question": search_q,
            "question_index": 0,
            "messages": [],
            "context_summary": "",
            "retrieval_keys": set(),
            "tool_call_count": 0,
            "iteration_count": 0,
        }
        trace_steps = []
        for step in range(6):
            orch_out = orchestrator(state, llm_tools)
            state["messages"].extend(orch_out.get("messages", []))
            state["tool_call_count"] += orch_out.get("tool_call_count", 0)
            state["iteration_count"] += orch_out.get("iteration_count", 0)
            last = state["messages"][-1]

            final_prompt = [SystemMessage(content=get_orchestrator_prompt())]
            if state.get("context_summary"):
                final_prompt.append(HumanMessage(content=f"[زمینه فشرده از تحقیقات قبلی]\n\n{state['context_summary']}"))
            final_prompt.extend(state["messages"])

            step_info = {
                "step": step,
                "orchestrator_system_prompt": get_orchestrator_prompt(),
                "messages_sent_to_model": [_msg_dict(m) for m in final_prompt],
            }
            trace_steps.append(step_info)

            if not getattr(last, "tool_calls", None):
                break

            tool_msgs = tool_node.invoke({"messages": state["messages"]})
            state["messages"].extend(tool_msgs["messages"])

        report["generation_steps"] = trace_steps
        report["final_messages_in_state"] = [_msg_dict(m) for m in state["messages"]]
        report["final_answer"] = collect_answer(state).get("final_answer")

        # Ollama model options if available
        try:
            import ollama

            info = ollama.show(config.LLM_MODEL)
            report["ollama_model_info"] = {
                "modelfile_excerpt": str(info)[:500],
            }
        except Exception as e:
            report["ollama_model_info_error"] = str(e)

    finally:
        config.QDRANT_DB_PATH = orig
        shutil.rmtree(qdrant_copy.parent, ignore_errors=True)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
