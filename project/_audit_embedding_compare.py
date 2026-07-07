import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import config
config.QDRANT_DB_PATH = str(Path(__file__).resolve().parent.parent / '_qdrant_audit_copy')
from db.vector_db_manager import VectorDbManager
from langchain_qdrant import QdrantVectorStore, RetrievalMode

QUERY = "استخراج رابطه از متن چیست و چه کاربردی دارد؟"
CORRECT_SNIPPET = "استخراج روابط از منابع متنی یکی از کاربردهای برنامه‌های خودکار است. هدف استخراج رابطه، توسعه استخراج‌کننده‌هایی است که موجودیت‌ها و روابط را از متن شناسایی کنند."
CORRUPTED = ". استخراج روابط از منابع متنی ها کاربردتوسط برنامه خودکاری ها جی است. هدف استخراج رابطه، توسعه استخراج کنندهکی ریز مساله را"

vdb = VectorDbManager()
dense = vdb._VectorDbManager__dense_embeddings
coll = vdb.get_collection(config.CHILD_COLLECTION)
client = vdb._VectorDbManager__client

def cos(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

qv = dense.embed_query(QUERY)
out = {
    'query': QUERY,
    'cosine_query_vs_corrupted_top_chunk': cos(qv, dense.embed_query(CORRUPTED)),
    'cosine_query_vs_ideal_persian': cos(qv, dense.embed_query(CORRECT_SNIPPET)),
}
# optional second model if cached
try:
    alt = __import__('langchain_huggingface', fromlist=['HuggingFaceEmbeddings']).HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-large'
    )
    out['multilingual_e5'] = {
        'vs_corrupted': cos(alt.embed_query(QUERY), alt.embed_query(CORRUPTED)),
        'vs_ideal': cos(alt.embed_query(QUERY), alt.embed_query(CORRECT_SNIPPET)),
    }
except Exception as e:
    out['multilingual_e5_error'] = str(e)

Path(__file__).resolve().parent.parent.joinpath('_audit_embedding_compare.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('done')
