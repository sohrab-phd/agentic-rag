"""Recompute metrics for saved evaluation results without rerunning inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import compute_metrics
from eval.run_experiment import _render_markdown, _summarize

FORBIDDEN_DEFAULT = [
    "شبکه اجتماعی",
    "شبکه‌های اجتماعی",
    "پزشک",
    "پزشکی",
    "دارو",
    "بیماری",
]


def _gold_from_row(row: dict) -> dict:
    answerable = row["category"] not in {"unanswerable", "adversarial"}
    source_doc = ""
    if answerable and row.get("retrieved_chunks"):
        source_doc = row["retrieved_chunks"][0].get("source", "")
    return {
        "id": row["id"],
        "category": row["category"],
        "source_doc": source_doc,
        "gold_answer": row.get("gold_answer", ""),
        "gold_passages": [row.get("gold_answer", "")],
        "must_contain": [],
        "must_not_contain": FORBIDDEN_DEFAULT,
        "answerable": answerable,
    }


def regrade(path: Path) -> Path:
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["results"]:
        old_metrics = row.get("metrics", {})
        gold = _gold_from_row(row)
        chunk_text = "\n".join(c.get("text", "") for c in row.get("retrieved_chunks", []))
        metrics = compute_metrics(
            gold,
            row.get("final_answer", ""),
            row.get("retrieved_chunks", []),
            chunk_text,
            old_metrics.get("semantic_similarity"),
        )

        # Keep the original must-contain judgment because it used item-specific
        # required terms from the gold dataset.
        metrics["must_contain_pass"] = old_metrics.get("must_contain_pass")
        metrics["must_contain_missing"] = old_metrics.get("must_contain_missing", [])
        if gold["answerable"]:
            metrics["passed"] = bool(
                metrics["source_hit"]
                and metrics["passage_recall"]
                and metrics["must_contain_pass"]
                and not metrics["forbidden_violation"]
                and (
                    metrics["semantic_similarity"] is None
                    or metrics["semantic_similarity"] >= 0.30
                )
            )
            if metrics["passed"]:
                metrics["failure_class"] = "ok"
            elif not metrics["source_hit"]:
                metrics["failure_class"] = "retrieval_wrong_source"
            elif not metrics["passage_recall"]:
                metrics["failure_class"] = "retrieval_miss"
            elif not metrics["must_contain_pass"]:
                metrics["failure_class"] = "generation_incomplete_or_wrong"

        row["metrics_original"] = old_metrics
        row["metrics"] = metrics

    report["summary"] = _summarize(report["results"])
    out = path.with_name(f"{path.stem}_regraded.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = path.with_name(f"{path.stem.replace('results', 'summary', 1)}_regraded.md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    return out


def main() -> None:
    eval_dir = Path(__file__).resolve().parent
    for filename in ("results_resume_from_13.json", "results_20260711_191256.json"):
        path = eval_dir / filename
        if not path.exists():
            continue
        out = regrade(path)
        report = json.loads(out.read_text(encoding="utf-8"))
        print(out)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
