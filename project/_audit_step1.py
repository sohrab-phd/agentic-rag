import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
config.QDRANT_DB_PATH = str(Path(__file__).resolve().parent.parent / '_qdrant_audit_copy')
from db.vector_db_manager import VectorDbManager
vdb = VectorDbManager()
client = vdb._VectorDbManager__client
count = client.count(config.CHILD_COLLECTION).count
print('POINT_COUNT', count)
pts, _ = client.scroll(config.CHILD_COLLECTION, limit=3, with_payload=True, with_vectors=True)
for p in pts:
    vec = p.vector
    dim = None
    if isinstance(vec, dict):
        for k,v in vec.items():
            if k != config.SPARSE_VECTOR_NAME and isinstance(v, list):
                dim = len(v)
    elif isinstance(vec, list):
        dim = len(vec)
    pl = p.payload or {}
    text = pl.get('page_content', pl.get('text',''))[:120]
    print('POINT', p.id, 'dim', dim, 'parent', pl.get('metadata',{}).get('parent_id') if isinstance(pl.get('metadata'), dict) else pl.get('parent_id'), 'text', text)
