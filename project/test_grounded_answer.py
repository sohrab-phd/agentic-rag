"""Test strict grounded answer for a Persian query."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage
from core.rag_system import RAGSystem
from rag_agent.nodes import _collect_retrieved_chunks
from rag_agent.tools import ToolFactory

QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
FORBIDDEN = [
  "شبکه اجتماعی",
  "شبکه‌های اجتماعی",
  "پزشک",
  "پزشکی",
  "دارو",
  "بیماری",
  "داروها",
  "بیماری‌ها",
]
OUT = Path(__file__).resolve().parent.parent / "grounded_answer_test.json"


def main() -> None:
    rag = RAGSystem()
    rag.initialize()

    result = rag.agent_graph.invoke(
        {"messages": [HumanMessage(content=QUERY)]},
        config=rag.get_config(),
    )

    answer = ""
    for item in reversed(result.get("agent_answers", [])):
        if item.get("answer"):
            answer = item["answer"]
            break
    if not answer and result.get("messages"):
        answer = result["messages"][-1].content

    # retrieval snapshot from last agent state if available
    collection = rag.vector_db.get_collection(rag.collection_name)
    tools = ToolFactory(collection).create_tools()
    search_tool = next(t for t in tools if t.name == "search_child_chunks")
    tool_out = search_tool.invoke({"query": QUERY, "limit": 7})

    forbidden_hits = {term: (term in answer) for term in FORBIDDEN}
    forbidden_in_chunks = {term: (term in tool_out) for term in FORBIDDEN}

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
