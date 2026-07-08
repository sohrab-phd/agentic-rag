"""Direct test of generate_grounded_answer node (bypasses rewrite_query)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from rag_agent.nodes import generate_grounded_answer
from rag_agent.tools import ToolFactory
from db.vector_db_manager import VectorDbManager

QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
FORBIDDEN = [
    "شبکه اجتماعی", "شبکه‌های اجتماعی", "پزشک", "پزشکی",
    "دارو", "بیماری", "داروها", "بیماری‌ها",
]
OUT = Path(__file__).resolve().parent.parent / "grounded_answer_test.json"


def main() -> None:
    vdb = VectorDbManager()
    coll = vdb.get_collection(config.CHILD_COLLECTION)
    tools = ToolFactory(coll).create_tools()
    search_tool = next(t for t in tools if t.name == "search_child_chunks")
    tool_out = search_tool.invoke({"query": QUERY, "limit": 7})

    state = {
        "question": QUERY,
        "messages": [
            HumanMessage(content=QUERY),
            AIMessage(content="", tool_calls=[{
                "id": "call_test",
                "name": "search_child_chunks",
                "args": {"query": QUERY, "limit": 7},
            }]),
            ToolMessage(content=tool_out, tool_call_id="call_test", name="search_child_chunks"),
        ],
    }

    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    out = generate_grounded_answer(state, llm)
    answer = out["messages"][0].content

    forbidden_hits = {t: (t in answer) for t in FORBIDDEN}
    forbidden_in_chunks = {t: (t in tool_out) for t in FORBIDDEN}

    report = {
        "query": QUERY,
        "retrieved_chunks_tool_output": tool_out,
        "final_answer": answer,
        "forbidden_terms_in_answer": forbidden_hits,
        "forbidden_terms_in_retrieved_chunks": forbidden_in_chunks,
        "passed": not any(forbidden_hits.values()),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
