"""Compare Phase 1A benchmark against baseline."""
from __future__ import annotations

import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent
BASELINE = EVAL / "results_full_30_combined.json"
AFTER = EVAL / "results_phase1a_rewrite_hardening.json"


def main() -> None:
    before = json.loads(BASELINE.read_text(encoding="utf-8"))
    after = json.loads(AFTER.read_text(encoding="utf-8"))

    b_map = {r["id"]: r for r in before["results"]}
    a_map = {r["id"]: r for r in after["results"]}

    improved = []
    regressed = []
    for rid, a in a_map.items():
        b = b_map.get(rid)
        if not b:
            continue
        bp = b["metrics"].get("passed")
        ap = a["metrics"].get("passed")
        if ap and not bp:
            improved.append(rid)
        if bp and not ap:
            regressed.append(rid)

    fallbacks = [
        r for r in after["results"]
        if r.get("rewrite_audit", {}).get("fallback_triggered")
    ]
    prevented = [
        r for r in after["results"]
        if r.get("rewrite_audit", {}).get("fallback_triggered")
        and r.get("rewrite_audit", {}).get("proposed_rewrites") != r.get("rewrite_audit", {}).get("final_rewrites")
    ]

    report = {
        "baseline_pass_rate": before["summary"]["overall_pass_rate"],
        "after_pass_rate": after["summary"]["overall_pass_rate"],
        "baseline_passed": before["summary"]["overall_passed"],
        "after_passed": after["summary"]["overall_passed"],
        "pass_rate_delta": round(
            after["summary"]["overall_pass_rate"] - before["summary"]["overall_pass_rate"], 3
        ),
        "baseline_doc_relevance": before["summary"]["doc_relevance_score"],
        "after_doc_relevance": after["summary"]["doc_relevance_score"],
        "baseline_avg_latency": round(
            sum(r["latency_s"] for r in before["results"]) / len(before["results"]), 2
        ),
        "after_avg_latency": round(
            sum(r["latency_s"] for r in after["results"]) / len(after["results"]), 2
        ),
        "rewrite_fallbacks_triggered": len(fallbacks),
        "improved_cases": improved,
        "regressed_cases": regressed,
        "prevented_drift_examples": [
            {
                "id": r["id"],
                "original": r["question"],
                "proposed": r["rewrite_audit"].get("proposed_rewrites"),
                "final": r["rewrite_audit"].get("final_rewrites"),
                "reasons": r["rewrite_audit"].get("failure_reasons"),
            }
            for r in prevented
        ],
        "by_category_before": before["summary"]["by_category"],
        "by_category_after": after["summary"]["by_category"],
    }

    out = EVAL / "phase1a_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
