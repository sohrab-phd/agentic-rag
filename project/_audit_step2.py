import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import config
config.QDRANT_DB_PATH = str(Path(__file__).resolve().parent.parent / '_qdrant_audit_copy')
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory

USER_QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
SEARCH_QUERY = USER_QUERY  # filled by step2 if rewrite works

vdb = VectorDbManager()
coll = vdb.get_collection(config.CHILD_COLLECTION)
dense = vdb._VectorDbManager__dense_embeddings

out = {"user_query": USER_QUERY, "search_query": SEARCH_QUERY}

vec = dense.embed_query(SEARCH_QUERY)
arr = np.array(vec)
out["embedding"] = {
    "model": config.DENSE_MODEL,
    "dimension": int(arr.size),
    "l2_norm": float(np.linalg.norm(arr)),
    "mean": float(arr.mean()),
    "first_5": [float(x) for x in arr[:5]],
}

for label, threshold in [("no_threshold", None), ("threshold_0_7", 0.7)]:
    hits = coll.similarity_search_with_score(SEARCH_QUERY, k=10, score_threshold=threshold)
    out[label] = [
        {
            "score": float(s),
            "parent_id": d.metadata.get("parent_id"),
            "source": d.metadata.get("source"),
            "len": len(d.page_content),
            "text": d.page_content,
        }
        for d, s in hits
    ]

tools = ToolFactory(coll).create_tools()
search_tool = next(t for t in tools if t.name == "search_child_chunks")
out["tool_output_limit_7"] = search_tool.invoke({"query": SEARCH_QUERY, "limit": 7})

from langchain_qdrant import QdrantVectorStore, RetrievalMode
client = vdb._VectorDbManager__client
dense_only = QdrantVectorStore(client=client, collection_name=config.CHILD_COLLECTION, embedding=dense, retrieval_mode=RetrievalMode.DENSE)
out["dense_only_top5"] = [
    {"cosine": float(s), "parent_id": d.metadata.get("parent_id"), "text": d.page_content[:250]}
    for d, s in dense_only.similarity_search_with_score(SEARCH_QUERY, k=5, score_threshold=None)
]

Path(__file__).resolve().parent.parent.joinpath('_audit_step2.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('done')
