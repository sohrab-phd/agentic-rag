import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
config.QDRANT_DB_PATH = str(Path(__file__).resolve().parent.parent / '_qdrant_audit_copy')
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory
from rag_agent.prompts import get_orchestrator_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

USER_QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
SEARCH_QUERY = USER_QUERY

vdb = VectorDbManager()
coll = vdb.get_collection(config.CHILD_COLLECTION)
tools = ToolFactory(coll).create_tools()
search_tool = next(t for t in tools if t.name == 'search_child_chunks')
tool_result = search_tool.invoke({'query': SEARCH_QUERY, 'limit': 7})

sys_prompt = get_orchestrator_prompt()
human_msgs = [
    HumanMessage(content=SEARCH_QUERY),
    HumanMessage(content="برای پاسخ به این پرسش، اولین قدم باید فراخوانی 'search_child_chunks' باشد."),
]
# Simulate one orchestrator round: tool called, result returned, then answer
messages_for_answer = [
    SystemMessage(content=sys_prompt),
    *human_msgs,
    # pretend model called tool and got result
]
answer_prompt = f"""System prompt excerpt: {sys_prompt[:600]}...

User question: {SEARCH_QUERY}

Retrieved tool result (search_child_chunks):
{tool_result}

Based ONLY on the retrieved content above, answer the user question in Persian."""

llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
try:
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=answer_prompt)])
    answer = resp.content
except Exception as e:
    answer = f"OLLAMA_ERROR: {e}"

out = {
    'search_query': SEARCH_QUERY,
    'tool_result': tool_result,
    'orchestrator_system_prompt': sys_prompt,
    'synthetic_answer_prompt': answer_prompt,
    'final_answer': answer,
}
Path(__file__).resolve().parent.parent.joinpath('_audit_step4.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('done')
