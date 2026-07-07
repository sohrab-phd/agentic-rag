import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from document_chunker import DocumentChuncker

md = list(Path(config.MARKDOWN_DIR).glob('*.md'))[0]
parents, children = DocumentChuncker().create_chunks_single(md)
out = {
    "file": md.name,
    "parent_count": len(parents),
    "child_count": len(children),
    "children": [{"i": i, "len": len(c.page_content), "text": c.page_content} for i, c in enumerate(children)],
}
Path(__file__).resolve().parent.parent.joinpath('_audit_chunks.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('children', len(children))
