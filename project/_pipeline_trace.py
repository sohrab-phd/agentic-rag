import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
md = (ROOT / "markdown_docs" / "چکیده فارسی.md").read_text(encoding="utf-8")
audit = json.loads((ROOT / "_audit_step2.json").read_text(encoding="utf-8"))

# Trace: only processing after extraction is surrogatepass in utils.py
lines = []
lines.append("=== PIPELINE TRACE ===")
lines.append(f"Markdown length: {len(md)}")
lines.append(f"Persian chars in markdown: {sum(1 for c in md if '\\u0600' <= c <= '\\u06FF')}")
lines.append("Markdown preview:")
lines.append(md[:600])
lines.append("")
lines.append("Known corruption patterns in stored markdown:")
patterns = ["فرآ ین", "دادهاستخراج", "کاربردتوسط", "کنندهکی", "ی ها ت", "استی ی", "پردازده"]
for p in patterns:
    lines.append(f"  {p}: {'FOUND' if re.search(p, md) else 'not found'}")
lines.append("")
lines.append("=== RETRIEVAL BEFORE (from _audit_step2.json) ===")
lines.append(f"Query: {audit['search_query']}")
for i, hit in enumerate(audit["no_threshold"][:5], 1):
    lines.append(f"Rank {i} score={hit['score']} len={hit['len']}")
    lines.append(hit["text"][:250])
lines.append("")
lines.append(f"Production tool (threshold 0.7) returned {audit['retrieval_with_threshold_0_7']['result_count']} chunk(s)")

(ROOT / "_pipeline_trace.txt").write_text("\n".join(lines), encoding="utf-8")
print("written")
