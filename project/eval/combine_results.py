"""Combine segmented evaluation result files into one report."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_experiment import _render_markdown, _summarize  # noqa: E402


def main() -> None:
    eval_dir = Path(__file__).resolve().parent
    files = [
        eval_dir / "results_first_12_recovery.json",
        eval_dir / "results_resume_from_13_regraded.json",
    ]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    results = []
    for report in reports:
        results.extend(report["results"])

    combined = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": reports[0]["config"],
        "summary": _summarize(results),
        "results": results,
        "segments": [str(path.name) for path in files],
    }
    out_json = eval_dir / "results_full_30_combined.json"
    out_md = eval_dir / "summary_full_30_combined.md"
    out_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(combined), encoding="utf-8")
    print(out_json)
    print(out_md)
    print(json.dumps(combined["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
