"""Phase 2 read-only audit: aggregate_answers impact."""
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
from eval.metrics import compute_metrics, normalize_fa, parse_retrieved_chunks
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from rag_agent.nodes import aggregate_answers
from rag_agent.tools import ToolFactory

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold_dataset.json"
BENCHMARK_PATH = EVAL_DIR / "results_phase1a_rewrite_hardening.json"
OUT_JSON = EVAL_DIR / "aggregation_audit_report.json"
OUT_MD = EVAL_DIR / "aggregation_audit_report.md"


def _normalize_text(text: str) -> str:
    return normalize_fa(text or "")


def _combined_pre_answer(agent_answers: list[dict]) -> str:
    if not agent_answers:
        return ""
    parts = sorted(agent_answers, key=lambda x: x.get("index", 0))
    return "\n\n---\n\n".join(p.get("answer", "") for p in parts if p.get("answer"))


def _information_lost(pre_text: str, final_text: str, gold: dict) -> tuple[bool, list[str]]:
    lost_terms: list[str] = []
    pre_n = _normalize_text(pre_text)
    final_n = _normalize_text(final_text)
    for term in gold.get("must_contain", []):
        tn = _normalize_text(term)
        if tn in pre_n and tn not in final_n:
            lost_terms.append(term)
    return len(lost_terms) > 0, lost_terms


def _near_identical(a: str, b: str) -> bool:
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    # strip sources footer for comparison
    for sep in ("---", "منابع"):
        if sep in a:
            a = a.split(sep)[0]
        if sep in b:
            b = b.split(sep)[0]
    na, nb = _normalize_text(a), _normalize_text(b)
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return shorter in longer or len(set(shorter) & set(longer)) / max(len(set(longer)), 1) > 0.85


def run_audit() -> dict:
    gold_items = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold_map = {g["id"]: g for g in gold_items}
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    bench_map = {r["id"]: r for r in benchmark["results"]}

    rag = RAGSystem()
    rag.initialize()
    collection = rag.vector_db.get_collection(rag.collection_name)
    tools = ToolFactory(collection).create_tools()
    search_tool = next(t for t in tools if t.name == "search_child_chunks")
    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    rows = []
    agg_latencies = []
    single_ran_agg = 0
    single_identical = 0

    for i, gold in enumerate(gold_items, 1):
        q = gold["question"]
        print(f"[{i}/30] {gold['id']}", flush=True)
        rag.thread_id = str(uuid.uuid4())

        t0 = time.perf_counter()
        state = rag.agent_graph.invoke(
            {"messages": [HumanMessage(content=q)]},
            config=rag.get_config(),
        )
        pipeline_latency = time.perf_counter() - t0

        agent_answers = state.get("agent_answers", [])
        rewritten = state.get("rewrittenQuestions", [])
        pre_combined = _combined_pre_answer(agent_answers)
        final_answer = str(state["messages"][-1].content) if state.get("messages") else ""

        search_q = rewritten[0] if rewritten else q
        tool_out = search_tool.invoke({"query": search_q, "limit": 7})
        chunks = parse_retrieved_chunks(tool_out)
        chunk_text = "\n".join(c.get("text", "") for c in chunks)

        # isolated aggregation timing
        agg_latency = 0.0
        if agent_answers:
            agg_state = {
                "agent_answers": agent_answers,
                "originalQuery": state.get("originalQuery", q),
                "messages": [],
            }
            t1 = time.perf_counter()
            aggregate_answers(agg_state, llm)
            agg_latency = time.perf_counter() - t1
            agg_latencies.append(agg_latency)

        pre_metrics = compute_metrics(gold, pre_combined, chunks, chunk_text, None)
        final_metrics = compute_metrics(gold, final_answer, chunks, chunk_text, None)

        info_lost, lost_terms = _information_lost(pre_combined, final_answer, gold)
        failure_introduced = bool(pre_metrics.get("passed") and not final_metrics.get("passed"))
        failure_preexisted = bool(not pre_metrics.get("passed") and not final_metrics.get("passed"))
        aggregation_fixed = bool(not pre_metrics.get("passed") and final_metrics.get("passed"))

        is_single = len(agent_answers) == 1
        if is_single:
            single_ran_agg += 1
            if _near_identical(agent_answers[0].get("answer", ""), final_answer):
                single_identical += 1

        rows.append({
            "id": gold["id"],
            "category": gold["category"],
            "original_question": q,
            "rewritten_questions": rewritten,
            "rewrite_audit": state.get("rewrite_audit", {}),
            "retrieved_chunks_count": len(chunks),
            "retrieved_top_source": chunks[0].get("source") if chunks else "",
            "grounded_answers_before_aggregation": agent_answers,
            "grounded_combined_before": pre_combined,
            "final_answer_after_aggregation": final_answer,
            "benchmark_passed": bench_map.get(gold["id"], {}).get("metrics", {}).get("passed"),
            "pre_aggregation_passed": pre_metrics.get("passed"),
            "final_passed": final_metrics.get("passed"),
            "information_lost": info_lost,
            "information_lost_terms": lost_terms,
            "failure_introduced_by_aggregation": failure_introduced,
            "failure_preexisted_before_aggregation": failure_preexisted,
            "aggregation_fixed_failure": aggregation_fixed,
            "aggregation_ran": True,
            "single_grounded_answer": is_single,
            "aggregation_latency_s": round(agg_latency, 2),
            "pipeline_latency_s": round(pipeline_latency, 2),
            "pre_failure_class": pre_metrics.get("failure_class"),
            "final_failure_class": final_metrics.get("failure_class"),
        })

    introduced = [r for r in rows if r["failure_introduced_by_aggregation"]]
    fixed = [r for r in rows if r["aggregation_fixed_failure"]]
    info_loss_rows = [r for r in rows if r["information_lost"]]
    failed_final = [r for r in rows if not r["final_passed"] and gold_map[r["id"]].get("answerable", True)]

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    summary = {
        "total": len(rows),
        "failed_final_answerable": len(failed_final),
        "failures_introduced_by_aggregation": len(introduced),
        "failures_fixed_by_aggregation": len(fixed),
        "failures_preexisted_before_aggregation": sum(
            1 for r in rows if r["failure_preexisted_before_aggregation"] and gold_map[r["id"]].get("answerable")
        ),
        "information_loss_rate": round(len(info_loss_rows) / len(rows), 3),
        "avg_aggregation_latency_s": round(sum(agg_latencies) / len(agg_latencies), 2) if agg_latencies else 0,
        "avg_pipeline_latency_s": round(sum(r["pipeline_latency_s"] for r in rows) / len(rows), 2),
        "aggregation_latency_pct_of_pipeline": round(
            100 * (sum(agg_latencies) / sum(r["pipeline_latency_s"] for r in rows)), 1
        ) if agg_latencies else 0,
        "single_answer_count": single_ran_agg,
        "single_answer_aggregation_identical": single_identical,
        "single_answer_aggregation_changed": single_ran_agg - single_identical,
        "multi_intent_count": sum(1 for r in rows if len(r["grounded_answers_before_aggregation"]) > 1),
        "by_category": {
            cat: {
                "count": len(items),
                "introduced_by_aggregation": sum(1 for x in items if x["failure_introduced_by_aggregation"]),
                "information_lost": sum(1 for x in items if x["information_lost"]),
                "avg_agg_latency_s": round(
                    sum(x["aggregation_latency_s"] for x in items) / len(items), 2
                ),
            }
            for cat, items in by_cat.items()
        },
        "introduced_case_ids": [r["id"] for r in introduced],
        "fixed_case_ids": [r["id"] for r in fixed],
    }

    recommendations = _recommendations(summary, rows)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_source": BENCHMARK_PATH.name,
        "summary": summary,
        "items": rows,
        "recommended_production_change": recommendations,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_md(report), encoding="utf-8")
    return report


def _recommendations(summary: dict, rows: list[dict]) -> list[dict]:
    single_changed = summary["single_answer_aggregation_changed"]
    introduced = summary["failures_introduced_by_aggregation"]
    multi_items = [r for r in rows if len(r["grounded_answers_before_aggregation"]) > 1]

    recs = []

    if introduced > 0 or summary["information_loss_rate"] > 0.2:
        recs.append({
            "rank": 1,
            "change": "Bypass aggregate_answers when there is exactly one grounded sub-answer and rewrite produced a single intent",
            "expected_pass_rate_improvement": "Low–medium (prevents corruption on single-answer flows)",
            "latency_reduction": f"~{summary['avg_aggregation_latency_s']}s per single-intent query ({summary['aggregation_latency_pct_of_pipeline']}% of pipeline)",
            "implementation_risk": "Low",
            "evidence": f"{single_changed} single-answer runs altered by aggregation; {introduced} failures introduced",
        })

    if any(r["failure_introduced_by_aggregation"] for r in multi_items):
        recs.append({
            "rank": 2,
            "change": "For multi-intent, replace LLM aggregation with deterministic concatenation of grounded sub-answers",
            "expected_pass_rate_improvement": "Medium for multi-intent (currently 50% pass rate)",
            "latency_reduction": f"~{sum(r['aggregation_latency_s'] for r in multi_items)/max(len(multi_items),1):.1f}s per multi-intent query",
            "implementation_risk": "Medium",
            "evidence": "multi_001 previously failed when aggregation dropped first sub-answer",
        })
    else:
        recs.append({
            "rank": 2,
            "change": "For multi-intent, use deterministic merge template before optional light aggregation",
            "expected_pass_rate_improvement": "Medium if multi-intent information loss detected",
            "latency_reduction": "Partial aggregation skip",
            "implementation_risk": "Medium",
            "evidence": f"{len(multi_items)} multi-intent items; information loss rate {summary['information_loss_rate']}",
        })

    recs.append({
        "rank": 3,
        "change": "Tighten aggregation prompt to forbid dropping any sub-answer facts; require per-part coverage",
        "expected_pass_rate_improvement": "Low",
        "latency_reduction": "None",
        "implementation_risk": "Low",
        "evidence": "Prompt-only; does not remove aggregation LLM call",
    })

    return sorted(recs, key=lambda x: x["rank"])


def _render_md(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Phase 2 — Aggregation Audit",
        "",
        f"**Benchmark reference:** {report['benchmark_source']}",
        f"**Failures introduced by aggregation:** {s['failures_introduced_by_aggregation']}",
        f"**Failures pre-existing before aggregation:** {s['failures_preexisted_before_aggregation']}",
        f"**Information loss rate:** {s['information_loss_rate']:.1%}",
        f"**Avg aggregation latency:** {s['avg_aggregation_latency_s']}s ({s['aggregation_latency_pct_of_pipeline']}% of pipeline)",
        f"**Single-answer flows:** {s['single_answer_count']} ran aggregation; {s['single_answer_aggregation_identical']} near-identical; {s['single_answer_aggregation_changed']} changed",
        "",
        "## Per-item table",
        "",
        "| ID | Grounded Before (excerpt) | Final After (excerpt) | Info Lost? | Failure Introduced? |",
        "|---|---|---|---:|---:|",
    ]
    for r in report["items"]:
        pre = (r["grounded_combined_before"] or "")[:80].replace("\n", " ")
        fin = (r["final_answer_after_aggregation"] or "")[:80].replace("\n", " ")
        lines.append(
            f"| {r['id']} | {pre}... | {fin}... | {r['information_lost']} | {r['failure_introduced_by_aggregation']} |"
        )

    lines.extend(["", "## Recommended production change", ""])
    for rec in report["recommended_production_change"]:
        lines.append(f"### {rec['rank']}. {rec['change']}")
        lines.append(f"- **Pass-rate improvement:** {rec['expected_pass_rate_improvement']}")
        lines.append(f"- **Latency reduction:** {rec['latency_reduction']}")
        lines.append(f"- **Risk:** {rec['implementation_risk']}")
        lines.append(f"- **Evidence:** {rec['evidence']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = run_audit()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
