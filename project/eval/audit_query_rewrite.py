"""Phase 1 audit: query rewriting impact on paraphrase and multi-intent failures.

Read-only instrumentation — does not modify production code.
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.rag_system import RAGSystem
from eval.metrics import (
    compute_metrics,
    normalize_fa,
    overlap_ratio,
    parse_retrieved_chunks,
    passage_recall,
    source_hit,
)
from langchain_core.messages import HumanMessage, SystemMessage
from rag_agent.nodes import generate_grounded_answer, rewrite_query
from rag_agent.tools import ToolFactory

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold_dataset.json"
COMBINED_PATH = EVAL_DIR / "results_full_30_combined.json"
OUT_JSON = EVAL_DIR / "rewrite_audit_report.json"
OUT_MD = EVAL_DIR / "rewrite_audit_report.md"

TARGET_CATEGORIES = {"paraphrase", "multi_intent"}


def _keywords(text: str) -> set[str]:
    text = normalize_fa(text)
    return {t for t in re.split(r"[^\w\u0600-\u06ff]+", text) if len(t) >= 3}


def _analyze_drift(original: str, rewritten: list[str], gold: dict) -> dict:
    orig_kw = _keywords(original)
    rew_kw = set()
    for rq in rewritten:
        rew_kw |= _keywords(rq)

    must = set(normalize_fa(t) for t in gold.get("must_contain", []))
    missing_must = [t for t in gold.get("must_contain", []) if normalize_fa(t) not in rew_kw]

    dropped = sorted(orig_kw - rew_kw)
    added = sorted(rew_kw - orig_kw)
    preserved = sorted(orig_kw & rew_kw)

    issues: list[str] = []
    if missing_must:
        issues.append("missing_entities")
    if len(dropped) > len(preserved):
        issues.append("dropped_keywords")
    if any("مراحل" in rq or "فرآیند" in rq or "شامل" in rq for rq in rewritten) and "چیست" in original:
        issues.append("semantic_drift")
    if len(rewritten) == 1 and (" و " in original or "؟ و " in original):
        issues.append("multi_intent_loss")
    if any(len(rq) < len(original) * 0.4 for rq in rewritten):
        issues.append("intent_collapse")
    if any("رابطه" == normalize_fa(rq.replace(" ", ""))[:5] or rq.startswith("رابطه در") for rq in rewritten):
        issues.append("intent_collapse")

    return {
        "rewritten_count": len(rewritten),
        "missing_must_in_rewrite": missing_must,
        "dropped_keywords": dropped[:12],
        "added_keywords": added[:12],
        "preserved_keywords": preserved[:12],
        "issue_tags": issues,
    }


def _retrieval_metrics(gold: dict, chunks: list[dict]) -> dict:
    return {
        "source_hit": source_hit(gold.get("source_doc", ""), chunks),
        "passage_recall": passage_recall(
            gold.get("gold_passages", [gold.get("gold_answer", "")]),
            chunks,
        ),
        "top_score": None,
        "chunk_count": len(chunks),
    }


def _search(collection, query: str, k: int = 7) -> tuple[list[dict], list[dict]]:
    tool_factory = ToolFactory(collection)
    tool_out = tool_factory._search_child_chunks(query, k)
    chunks = parse_retrieved_chunks(tool_out)
    try:
        hits = collection.similarity_search_with_score(query, k=k, score_threshold=0.5)
    except Exception:
        hits = collection.similarity_search_with_score(query, k=k, score_threshold=None)
    scores = [
        {
            "source": d.metadata.get("source", ""),
            "parent_id": d.metadata.get("parent_id", ""),
            "score": float(s),
            "text": d.page_content.strip()[:300],
        }
        for d, s in hits
    ]
    return chunks, scores


def _grounded_answer(llm, question: str, chunks: list[dict]) -> str:
    chunk_text = "\n\n---\n\n".join(
        f"Parent ID: {c.get('parent_id', '')}\nFile Name: {c.get('source', '')}\nContent: {c.get('text', '')}"
        for c in chunks
    )
    state = {"messages": [], "question": question}
    # Reuse production grounded path with synthetic tool context in state
    from langchain_core.messages import ToolMessage

    state["messages"] = [ToolMessage(content=chunk_text or "NO_RELEVANT_CHUNKS", tool_call_id="audit")]
    out = generate_grounded_answer(state, llm)
    return out["messages"][-1].content


def run_audit() -> dict:
    gold_all = {g["id"]: g for g in json.loads(GOLD_PATH.read_text(encoding="utf-8"))}
    combined = json.loads(COMBINED_PATH.read_text(encoding="utf-8"))
    pipeline_a = {
        r["id"]: r
        for r in combined["results"]
        if r["category"] in TARGET_CATEGORIES
    }

    rag = RAGSystem()
    rag.initialize()
    collection = rag.vector_db.get_collection(rag.collection_name)
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    cases = []
    rewrite_caused_retrieval = 0
    rewrite_caused_answer = 0
    would_pass_with_original = 0

    for item_id, row in pipeline_a.items():
        gold = gold_all[item_id]
        original = row["question"]
        rewritten = row.get("rewritten_questions") or [original]
        drift = _analyze_drift(original, rewritten, gold)

        # A: production snapshot already in row
        a_chunks = row.get("retrieved_chunks", [])
        a_metrics = {
            "retrieval": _retrieval_metrics(gold, a_chunks),
            "passed": row["metrics"].get("passed"),
            "failure_class": row["metrics"].get("failure_class"),
            "latency_s": row.get("latency_s"),
            "answer": row.get("final_answer", ""),
        }
        if a_chunks and row.get("retrieval_scores"):
            a_metrics["retrieval"]["top_score"] = row["retrieval_scores"][0].get("score")

        # B: retrieval with original user query only
        t0 = time.perf_counter()
        b_chunks, b_scores = _search(collection, original, k=7)
        b_retrieval_latency = time.perf_counter() - t0
        b_ret_metrics = _retrieval_metrics(gold, b_chunks)
        if b_scores:
            b_ret_metrics["top_score"] = b_scores[0]["score"]

        # B answer via grounded path (isolated, not production pipeline)
        t1 = time.perf_counter()
        b_answer = _grounded_answer(llm, original, b_chunks)
        b_answer_latency = time.perf_counter() - t1
        b_chunk_text = "\n".join(c.get("text", "") for c in b_chunks)
        b_full_metrics = compute_metrics(gold, b_answer, b_chunks, b_chunk_text, None)

        rewrite_retrieval_regression = (
            not a_metrics["retrieval"]["passage_recall"]
            and b_ret_metrics["passage_recall"]
        )
        rewrite_answer_regression = (
            a_metrics["retrieval"]["passage_recall"]
            and not row["metrics"].get("passed")
            and b_full_metrics.get("passed")
        )
        original_would_pass = b_full_metrics.get("passed", False)

        if rewrite_retrieval_regression:
            rewrite_caused_retrieval += 1
        if rewrite_answer_regression:
            rewrite_caused_answer += 1
        if original_would_pass and not row["metrics"].get("passed"):
            would_pass_with_original += 1

        cases.append({
            "id": item_id,
            "category": row["category"],
            "passed_in_pipeline_a": row["metrics"].get("passed"),
            "original_question": original,
            "rewritten_questions": rewritten,
            "search_query_used_in_a": row.get("search_query_used"),
            "drift_analysis": drift,
            "pipeline_a": a_metrics,
            "retrieval_b_original_query": {
                "metrics": b_ret_metrics,
                "scores": b_scores[:3],
                "top_chunk_preview": b_chunks[0]["text"][:250] if b_chunks else "",
                "latency_s": round(b_retrieval_latency, 3),
            },
            "answer_b_original_query": {
                "answer": b_answer,
                "metrics": b_full_metrics,
                "latency_s": round(b_answer_latency, 2),
            },
            "ab_comparison": {
                "retrieval_passage_recall_delta": int(b_ret_metrics["passage_recall"]) - int(a_metrics["retrieval"]["passage_recall"]),
                "top_score_delta": round((b_ret_metrics.get("top_score") or 0) - (a_metrics["retrieval"].get("top_score") or 0), 3),
                "pass_rate_a": row["metrics"].get("passed"),
                "pass_rate_b_grounded_only": b_full_metrics.get("passed"),
                "rewrite_caused_retrieval_regression": rewrite_retrieval_regression,
                "rewrite_caused_answer_regression": rewrite_answer_regression,
                "likely_rewrite_root_cause": rewrite_retrieval_regression or (
                    drift["issue_tags"] and not row["metrics"].get("passed")
                ),
            },
        })

    failed = [c for c in cases if not c["passed_in_pipeline_a"]]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paraphrase + multi_intent (all 7 benchmark items)",
        "summary": {
            "total_cases": len(cases),
            "failed_in_pipeline_a": len(failed),
            "rewrite_caused_retrieval_regressions": rewrite_caused_retrieval,
            "rewrite_caused_answer_regressions": rewrite_caused_answer,
            "would_pass_if_original_query_used_for_grounded_answer": would_pass_with_original,
            "ab_retrieval_improved_with_original": sum(
                1 for c in cases if c["ab_comparison"]["retrieval_passage_recall_delta"] > 0
            ),
            "ab_answer_improved_with_original": sum(
                1 for c in cases if c["ab_comparison"]["pass_rate_b_grounded_only"] and not c["passed_in_pipeline_a"]
            ),
        },
        "failure_tags": _aggregate_tags(failed),
        "cases": cases,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_md(report), encoding="utf-8")
    return report


def _aggregate_tags(failed: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for c in failed:
        for tag in c["drift_analysis"]["issue_tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def _render_md(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Phase 1 — Query Rewrite Audit",
        "",
        f"**Cases audited:** {s['total_cases']} (paraphrase + multi-intent)",
        f"**Failed in pipeline A:** {s['failed_in_pipeline_a']}",
        f"**Retrieval regressions caused by rewrite:** {s['rewrite_caused_retrieval_regressions']}",
        f"**Answer regressions where rewrite retrieval was OK but B passed:** {s['rewrite_caused_answer_regressions']}",
        f"**Would pass with original-query grounded answer:** {s['would_pass_if_original_query_used_for_grounded_answer']}",
        f"**A/B retrieval improved with original query:** {s['ab_retrieval_improved_with_original']}/{s['total_cases']}",
        f"**A/B answer improved with original query:** {s['ab_answer_improved_with_original']}/{s['total_cases']}",
        "",
        "## Drift Tags (failed cases)",
        "",
    ]
    for tag, n in sorted(report["failure_tags"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{tag}**: {n}")

    lines.append("\n## Failed Case Details\n")
    for c in report["cases"]:
        if c["passed_in_pipeline_a"]:
            continue
        lines.append(f"### {c['id']} ({c['category']})")
        lines.append(f"**Original:** {c['original_question']}")
        lines.append(f"**Rewritten:** {', '.join(c['rewritten_questions'])}")
        lines.append(f"**Drift tags:** {', '.join(c['drift_analysis']['issue_tags']) or 'none'}")
        lines.append(f"**Pipeline A pass:** {c['pipeline_a']['passed']} ({c['pipeline_a']['failure_class']})")
        lines.append(f"**B retrieval passage_recall:** {c['retrieval_b_original_query']['metrics']['passage_recall']}")
        lines.append(f"**B grounded answer pass:** {c['answer_b_original_query']['metrics']['passed']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = run_audit()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
