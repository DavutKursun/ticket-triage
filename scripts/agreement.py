"""Silver-vs-human agreement on the hand-checked CSVs: accuracy and Cohen's kappa per task.

Usage: python scripts/agreement.py            # both dev_100 and test_100
Also writes the hand-check templates if they do not exist yet (--make-templates).
Acceptance: kappa ≥ 0.75 per task; below that, refine the rubric once and relabel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.config import HANDCHECK, RESULTS  # noqa: E402
from ticket_triage.data import read_handcheck, write_handcheck_template  # noqa: E402


def kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(list(a), list(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-templates", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if args.make_templates:
        for s in ("dev", "test"):
            print("wrote", write_handcheck_template(s, args.n))
        return 0

    report = {}
    for s in ("dev", "test"):
        p = HANDCHECK / f"{s}_{args.n}.csv"
        if not p.exists():
            print(f"missing {p} — run with --make-templates first")
            continue
        df = read_handcheck(s, args.n)
        if df.empty:
            print(f"{s}: nothing filled in yet")
            continue
        rep = {"n_checked": int(len(df))}
        for task in ("urgency", "sentiment"):
            h, m = df[task], df[f"silver_{task}"]
            rep[task] = {"accuracy": round(float((h == m).mean()), 3), "kappa": round(kappa(h, m), 3)}
            disagreements = df[h != m][["id", "category", f"silver_{task}", task, "note"]]
            rep[task]["n_disagree"] = int(len(disagreements))
        report[s] = rep
        print(s, json.dumps(rep, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = [f"{s}/{t}" for s, r in report.items() for t in ("urgency", "sentiment") if r.get(t, {}).get("kappa", 1) < 0.75]
    if bad:
        print("BELOW 0.75 kappa:", bad, "→ refine the rubric once and relabel (bump PROMPT_VERSION)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
