import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rag_agent.nodes import rewrite_query, orchestrator, collect_answer
from rag_agent.prompts import get_orchestrator_prompt, get_rewrite_query_prompt
from rag_agent.schemas import QueryAnalysis

USER_QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
config.QDRANT_DB_PATH = str(Path(__file__).resolve().parent.parent / '_qdrant_audit_copy')
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory

llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
rewrite_state = {"messages": [HumanMessage(content=USER_QUERY)], "conversation_summary": ""}
rewrite_out = rewrite_query(rewrite_state, llm)
search_query = rewrite_out.get("rewrittenQuestions", [USER_QUERY])[0]

vdb = VectorDbManager()
coll = vdb.get_collection(config.CHILD_COLLECTION)
tools = ToolFactory(coll).create_tools()
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

trace = {"rewrite": rewrite_out, "search_query": search_query, "steps": []}

for step in range(5):
    orch_out = orchestrator(agent_state, llm_tools)
    agent_state["messages"].extend(orch_out.get("messages", []))
    agent_state["tool_call_count"] += orch_out.get("tool_call_count", 0)
    agent_state["iteration_count"] += orch_out.get("iteration_count", 0)
    last = agent_state["messages"][-1]
    step_info = {
        "step": step,
        "last_message_type": type(last).__name__,
        "tool_calls": getattr(last, "tool_calls", None),
        "content": getattr(last, "content", None),
    }
    if getattr(last, "tool_calls", None):
        from langchain_core.messages import ToolMessage
        for tc in last.tool_calls:
            tool_fn = next(t for t in tools if t.name == tc["name"])
            result = tool_fn.invoke(tc["args"])
            step_info.setdefault("tool_results", []).append({
                "name": tc["name"],
                "args": tc["args"],
                "result": result,
            })
            agent_state["messages"].append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )
    trace["steps"].append(step_info)
    if not getattr(last, "tool_calls", None):
        break

answer = collect_answer(agent_state)
trace["final_answer"] = answer.get("final_answer")
trace["all_messages"] = [
    {"type": type(m).__name__, "content": m.content, "tool_calls": getattr(m, "tool_calls", None), "name": getattr(m, "name", None)}
    for m in agent_state["messages"]
]
trace["orchestrator_system_prompt"] = get_orchestrator_prompt()
trace["rewrite_system_prompt"] = get_rewrite_query_prompt()

Path(__file__).resolve().parent.parent.joinpath('_audit_step3.json').write_text(
    json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print('done', 'answer_len', len(trace.get('final_answer') or ''))
