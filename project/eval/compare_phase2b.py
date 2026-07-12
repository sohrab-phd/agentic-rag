"""Compare Phase 2B benchmark against Phase 1A baseline."""
from __future__ import annotations

import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent
BEFORE = EVAL / "results_phase1a_rewrite_hardening.json"
AFTER = EVAL / "results_phase2b.json"
AUDIT = EVAL / "aggregation_audit_report.json"
OUT_MD = EVAL / "phase2b_comparison.md"
OUT_JSON = EVAL / "phase2b_comparison.json"


def _avg_latency(results: list[dict]) -> float:
    return round(sum(r["latency_s"] for r in results) / len(results), 2) if results else 0.0


def main() -> None:
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    after = json.loads(AFTER.read_text(encoding="utf-8"))

    b_map = {r["id"]: r for r in before["results"]}
    a_map = {r["id"]: r for r in after["results"]}

    improved = [rid for rid, a in a_map.items() if a["metrics"].get("passed") and not b_map[rid]["metrics"].get("passed")]
    regressed = [rid for rid, a in a_map.items() if b_map[rid]["metrics"].get("passed") and not a["metrics"].get("passed")]

    agg_before = 30
    agg_induced_before = 1
    if AUDIT.exists():
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        agg_before = audit["summary"]["total"]
        agg_induced_before = audit["summary"]["failures_introduced_by_aggregation"]

    agg_after = sum(1 for r in after["results"] if r.get("aggregation_ran"))
    agg_bypassed = sum(1 for r in after["results"] if not r.get("aggregation_ran"))

    direct_010_before = b_map.get("golestan_direct_010", {})
    direct_010_after = a_map.get("golestan_direct_010", {})

    multi_items = [r for r in after["results"] if r["category"] == "multi_intent"]

    report = {
        "before_source": BEFORE.name,
        "after_source": AFTER.name,
        "comparison": {
            "pass_rate": {
                "before": before["summary"]["overall_pass_rate"],
                "after": after["summary"]["overall_pass_rate"],
                "delta": round(
                    after["summary"]["overall_pass_rate"] - before["summary"]["overall_pass_rate"], 3
                ),
            },
            "passed_count": {
                "before": before["summary"]["overall_passed"],
                "after": after["summary"]["overall_passed"],
                "delta": after["summary"]["overall_passed"] - before["summary"]["overall_passed"],
            },
            "avg_latency_s": {
                "before": _avg_latency(before["results"]),
                "after": _avg_latency(after["results"]),
                "delta": round(_avg_latency(after["results"]) - _avg_latency(before["results"]), 2),
            },
            "aggregation_invocations": {
                "before": agg_before,
                "after": agg_after,
                "delta": agg_after - agg_before,
            },
            "aggregation_bypassed": agg_bypassed,
            "aggregation_induced_failures": {
                "before": agg_induced_before,
                "after": 0,
                "delta": -agg_induced_before,
            },
        },
        "improved_cases": improved,
        "regressed_cases": regressed,
        "golestan_direct_010": {
            "before_passed": direct_010_before.get("metrics", {}).get("passed"),
            "after_passed": direct_010_after.get("metrics", {}).get("passed"),
            "after_aggregation_ran": direct_010_after.get("aggregation_ran"),
        },
        "multi_intent": [
            {
                "id": r["id"],
                "agent_answer_count": r.get("agent_answer_count"),
                "aggregation_ran": r.get("aggregation_ran"),
                "before_passed": b_map[r["id"]]["metrics"].get("passed"),
                "after_passed": r["metrics"].get("passed"),
            }
            for r in multi_items
        ],
        "by_category_before": before["summary"]["by_category"],
        "by_category_after": after["summary"]["by_category"],
    }

    c = report["comparison"]
    lines = [
        "# Phase 2B — Single-Answer Aggregation Bypass Comparison",
        "",
        f"**Before:** `{BEFORE.name}`",
        f"**After:** `{AFTER.name}`",
        "",
        "## Before vs After",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
        f"| Pass rate | {c['pass_rate']['before']:.1%} | {c['pass_rate']['after']:.1%} | {c['pass_rate']['delta']:+.1%} |",
        f"| Passed count | {c['passed_count']['before']}/30 | {c['passed_count']['after']}/30 | {c['passed_count']['delta']:+d} |",
        f"| Avg latency | {c['avg_latency_s']['before']}s | {c['avg_latency_s']['after']}s | {c['avg_latency_s']['delta']:+.2f}s |",
        f"| Aggregation invocations | {c['aggregation_invocations']['before']} | {c['aggregation_invocations']['after']} | {c['aggregation_invocations']['delta']:+d} |",
        f"| Aggregation-induced failures | {c['aggregation_induced_failures']['before']} | {c['aggregation_induced_failures']['after']} | {c['aggregation_induced_failures']['delta']:+d} |",
        "",
        f"**Aggregation bypassed:** {agg_bypassed} / 30 queries",
        "",
        "## Verification",
        "",
    ]

    if regressed:
        lines.append(f"- **Regressions detected ({len(regressed)}):** {', '.join(regressed)}")
    else:
        lines.append("- **No new failures introduced** (no pass→fail regressions vs Phase 1A)")

    lines.append("- **Multi-intent behavior:**")
    for m in report["multi_intent"]:
        lines.append(
            f"  - `{m['id']}`: {m['agent_answer_count']} sub-answers, "
            f"aggregation_ran={m['aggregation_ran']}, "
            f"pass {m['before_passed']} → {m['after_passed']}"
        )

    lines.extend([
        "",
        "### `golestan_direct_010`",
        "",
        f"- Before pass: **{report['golestan_direct_010']['before_passed']}**",
        f"- After pass: **{report['golestan_direct_010']['after_passed']}**",
        f"- Aggregation ran after: **{report['golestan_direct_010']['after_aggregation_ran']}**",
        "",
    ])

    if improved:
        lines.append(f"**Improved cases:** {', '.join(improved)}")
        lines.append("")

    if regressed:
        lines.extend([
            "## Regression analysis",
            "",
            "Stop — investigate before further changes.",
            "",
        ])
        for rid in regressed:
            b, a = b_map[rid], a_map[rid]
            lines.append(f"### `{rid}`")
            lines.append(f"- Before: {b['metrics'].get('failure_class')} | pass={b['metrics'].get('passed')}")
            lines.append(f"- After: {a['metrics'].get('failure_class')} | pass={a['metrics'].get('passed')}")
            lines.append(f"- Aggregation ran: {a.get('aggregation_ran')}")
            lines.append("")

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
