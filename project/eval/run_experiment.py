"""Run document-grounded evaluation experiment."""
from __future__ import annotations

import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.rag_system import RAGSystem
from eval.metrics import (
    compute_metrics,
    normalize_fa,
    parse_retrieved_chunks,
)
from langchain_core.messages import HumanMessage
from rag_agent.tools import ToolFactory

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold_dataset.json"
ROOT = EVAL_DIR.parent.parent


def _extract_answer(result: dict) -> str:
    for item in reversed(result.get("agent_answers", [])):
        if item.get("answer"):
            return item["answer"]
    msgs = result.get("messages", [])
    if msgs:
        return str(msgs[-1].content)
    return ""


def _load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)


def _semantic_similarity(embedder, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        import numpy as np

        va = embedder.embed_query(a)
        vb = embedder.embed_query(b)
        da = np.linalg.norm(va)
        db = np.linalg.norm(vb)
        if da == 0 or db == 0:
            return 0.0
        return float(np.dot(va, vb) / (da * db))
    except Exception:
        return 0.0


def _retrieval_with_scores(collection, query: str, k: int = 7):
    try:
        hits = collection.similarity_search_with_score(query, k=k, score_threshold=0.5)
    except Exception:
        hits = collection.similarity_search_with_score(query, k=k, score_threshold=None)
    rows = []
    for doc, score in hits:
        rows.append({
            "source": doc.metadata.get("source", ""),
            "parent_id": doc.metadata.get("parent_id", ""),
            "score": float(score),
            "text": doc.page_content.strip()[:400],
        })
    return rows


def run_experiment(
    limit: int | None = None,
    start_index: int = 0,
    out_prefix: str | None = None,
) -> dict:
    if not GOLD_PATH.exists():
        from eval.build_gold_dataset import main as build_main

        build_main()

    gold_items = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    total_items = len(gold_items)
    if limit:
        gold_items = gold_items[:limit]
        total_items = len(gold_items)
    if start_index:
        gold_items = gold_items[start_index:]

    rag = RAGSystem()
    rag.initialize()
    collection = rag.vector_db.get_collection(rag.collection_name)
    tools = ToolFactory(collection).create_tools()
    search_tool = next(t for t in tools if t.name == "search_child_chunks")

    embedder = _load_embeddings()
    results = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = out_prefix or f"results_{stamp}"
    partial_path = EVAL_DIR / f"{prefix}_partial.jsonl"

    for offset, gold in enumerate(gold_items, start_index + 1):
        q = gold["question"]
        print(f"[{offset}/{total_items}] {gold['id']}: {q[:50]}...", flush=True)

        rag.thread_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            state = rag.agent_graph.invoke(
                {"messages": [HumanMessage(content=q)]},
                config=rag.get_config(),
            )
            error = None
        except Exception as e:
            state = {}
            error = str(e)
        elapsed = time.perf_counter() - t0

        answer = _extract_answer(state) if not error else ""
        rewritten = state.get("rewrittenQuestions", [])

        # Retrieval snapshot using production search query (first rewritten or original)
        search_q = rewritten[0] if rewritten else q
        tool_out = search_tool.invoke({"query": search_q, "limit": 7})
        chunks = parse_retrieved_chunks(tool_out)
        chunk_text = "\n".join(c.get("text", "") for c in chunks)
        scored_hits = _retrieval_with_scores(collection, search_q, k=7)

        sim = None
        if gold.get("answerable") and gold.get("gold_answer"):
            sim = _semantic_similarity(embedder, answer, gold["gold_answer"])

        metrics = compute_metrics(gold, answer, chunks, chunk_text, sim)

        row = {
            "id": gold["id"],
            "category": gold["category"],
            "question": q,
            "rewritten_questions": rewritten,
            "rewrite_audit": state.get("rewrite_audit", {}),
            "search_query_used": search_q,
            "latency_s": round(elapsed, 2),
            "error": error,
            "retrieved_chunks": chunks,
            "retrieval_scores": scored_hits,
            "final_answer": answer,
            "gold_answer": gold.get("gold_answer", ""),
            "metrics": metrics,
        }
        results.append(row)
        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        status = "PASS" if metrics.get("passed") else f"FAIL ({metrics.get('failure_class')})"
        print(f"  -> {status} | {elapsed:.1f}s | chunks={len(chunks)}", flush=True)

    summary = _summarize(results)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "llm_model": config.LLM_MODEL,
            "dense_model": config.DENSE_MODEL,
            "score_threshold": 0.5,
        },
        "summary": summary,
        "results": results,
    }

    out_json = EVAL_DIR / f"{prefix}.json"
    out_md = EVAL_DIR / f"{prefix.replace('results', 'summary', 1)}.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return report


def _summarize(results: list[dict]) -> dict:
    by_cat: dict[str, list] = defaultdict(list)
    failures = Counter()
    for r in results:
        by_cat[r["category"]].append(r)
        if not r["metrics"].get("passed"):
            failures[r["metrics"].get("failure_class", "unknown")] += 1

    cat_summary = {}
    for cat, rows in by_cat.items():
        n = len(rows)
        passed = sum(1 for r in rows if r["metrics"].get("passed"))
        answerable = [r for r in rows if r.get("metrics", {}).get("source_hit") is not None]
        cat_summary[cat] = {
            "total": n,
            "passed": passed,
            "pass_rate": round(passed / n, 3) if n else 0,
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2) if n else 0,
            "source_hit_rate": round(
                sum(1 for r in answerable if r["metrics"].get("source_hit")) / len(answerable), 3
            ) if answerable else None,
            "passage_recall_rate": round(
                sum(1 for r in answerable if r["metrics"].get("passage_recall")) / len(answerable), 3
            ) if answerable else None,
            "must_contain_rate": round(
                sum(1 for r in answerable if r["metrics"].get("must_contain_pass")) / len(answerable), 3
            ) if answerable else None,
        }

    answerable_rows = [r for r in results if r["metrics"].get("source_hit") is not None]
    retrieval_pass = sum(1 for r in answerable_rows if r["metrics"].get("source_hit") and r["metrics"].get("passage_recall"))
    correctness_pass = sum(1 for r in answerable_rows if r["metrics"].get("must_contain_pass"))
    faith_pass = sum(1 for r in results if not r["metrics"].get("forbidden_violation"))

    n_ans = len(answerable_rows) or 1
    doc_relevance = round(
        0.4 * (retrieval_pass / n_ans)
        + 0.35 * (correctness_pass / n_ans)
        + 0.25 * (faith_pass / len(results)),
        3,
    )

    return {
        "total": len(results),
        "overall_passed": sum(1 for r in results if r["metrics"].get("passed")),
        "overall_pass_rate": round(sum(1 for r in results if r["metrics"].get("passed")) / len(results), 3) if results else 0,
        "doc_relevance_score": doc_relevance,
        "failure_classes": dict(failures),
        "by_category": cat_summary,
    }


def _render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Document-Grounded Evaluation Report",
        "",
        f"**Date:** {report['timestamp_utc']}",
        f"**Model:** {report['config']['llm_model']}",
        f"**Doc Relevance Score:** {s['doc_relevance_score']}",
        f"**Overall Pass Rate:** {s['overall_pass_rate']:.1%} ({s['overall_passed']}/{s['total']})",
        "",
        "## By Category",
        "",
        "| Category | Pass Rate | Source Hit | Passage Recall | Must Contain | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, cs in s["by_category"].items():
        lines.append(
            f"| {cat} | {cs['pass_rate']:.0%} | "
            f"{cs['source_hit_rate'] or 'N/A'} | {cs['passage_recall_rate'] or 'N/A'} | "
            f"{cs['must_contain_rate'] or 'N/A'} | {cs['avg_latency_s']}s |"
        )

    lines.extend(["", "## Failure Breakdown", ""])
    for fc, count in sorted(s["failure_classes"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{fc}**: {count}")

    lines.extend(["", "## Failed Cases", ""])
    for r in report["results"]:
        if r["metrics"].get("passed"):
            continue
        lines.append(f"### {r['id']} ({r['category']}) — {r['metrics'].get('failure_class')}")
        lines.append(f"**Q:** {r['question']}")
        lines.append(f"**Search query:** {r.get('search_query_used', '')}")
        lines.append(f"**Answer:** {r['final_answer'][:500]}...")
        if r.get("retrieval_scores"):
            top = r["retrieval_scores"][0]
            lines.append(f"**Top retrieval:** {top.get('source')} (score={top.get('score'):.3f})")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only first N questions")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based dataset index to start from")
    parser.add_argument("--out-prefix", default=None, help="Output filename prefix, without extension")
    args = parser.parse_args()
    run_experiment(limit=args.limit, start_index=args.start_index, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
