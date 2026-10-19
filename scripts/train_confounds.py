"""Confound check over the LABELLED train rows (urgency/sentiment present): does language or length predict the label?

Usage: python scripts/train_confounds.py   → eval/results/train_confounds.{json,md}
Sets: "M3-1" = Bitext silver + synthetic plan "minority"; "M3-2" = + plan "control" + translated rows; "M3-2b" =
everything train_rows() loads (+ plan "short"). Groups: language en/tr, length ≤16 / >16 words.
Targets: Turkish urgent+high share within 10 points of English; the >16-word group at least 40% low/medium.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.config import RESULTS  # noqa: E402
from ticket_triage.data import train_rows  # noqa: E402
from ticket_triage.schema import URGENCIES  # noqa: E402

LONG = 16


def _group_stats(rows: list[dict]) -> dict:
    n = len(rows)
    urg = {u: sum(r["urgency"] == u for r in rows) for u in URGENCIES}
    return {"n": n, "urgency": urg, "urgent_high": round((urg["urgent"] + urg["high"]) / n, 3) if n else 0.0,
            "low_medium": round((urg["low"] + urg["medium"]) / n, 3) if n else 0.0,
            "positive": round(sum(r["sentiment"] == "positive" for r in rows) / n, 3) if n else 0.0}


def table(rows: list[dict]) -> dict:
    lab = [r for r in rows if r["urgency"]]
    groups = {"en": [r for r in lab if r["language"] == "en"], "tr": [r for r in lab if r["language"] == "tr"],
              f"<={LONG} words": [r for r in lab if len(r["text"].split()) <= LONG], f">{LONG} words": [r for r in lab if len(r["text"].split()) > LONG]}
    out = {g: _group_stats(v) for g, v in groups.items()}
    out["all"] = _group_stats(lab)
    out["targets"] = {"tr_vs_en_urgent_high_gap_points": round(100 * abs(out["tr"]["urgent_high"] - out["en"]["urgent_high"]), 1),
                      "tr_gap_ok": abs(out["tr"]["urgent_high"] - out["en"]["urgent_high"]) <= 0.10,
                      "long_low_medium_ok": out[f">{LONG} words"]["low_medium"] >= 0.40}
    return out


def markdown(res: dict) -> str:
    lines = ["| set | group | n | low | medium | high | urgent | urgent+high | positive |", "|---|---|---|---|---|---|---|---|---|"]
    for name, t in res.items():
        for g in ("en", "tr", f"<={LONG} words", f">{LONG} words", "all"):
            s = t[g]
            u = s["urgency"]
            lines.append(f"| {name} | {g} | {s['n']} | {u['low']} | {u['medium']} | {u['high']} | {u['urgent']} | "
                         f"{100 * s['urgent_high']:.0f}% | {100 * s['positive']:.0f}% |")
    tgt = "\n".join(f"- {name}: Turkish vs English urgent+high gap {t['targets']['tr_vs_en_urgent_high_gap_points']} points "
                    f"({'ok' if t['targets']['tr_gap_ok'] else 'MISSED'}, target ≤ 10); >{LONG}-word rows low/medium "
                    f"{100 * t[f'>{LONG} words']['low_medium']:.0f}% ({'ok' if t['targets']['long_low_medium_ok'] else 'MISSED'}, target ≥ 40%)"
                    for name, t in res.items())
    return "# Train-set confounds (labelled rows only)\n\n" + "\n".join(lines) + "\n\n" + tgt + "\n"


def main() -> int:
    def plans(*names):
        return lambda r: r.get("plan", "minority") in names

    res = {"M3-1": table(train_rows(translated=False, synth_filter=plans("minority"))),
           "M3-2": table(train_rows(synth_filter=plans("minority", "control"))),
           "M3-2b": table(train_rows())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_confounds.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = markdown(res)
    (RESULTS / "train_confounds.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
