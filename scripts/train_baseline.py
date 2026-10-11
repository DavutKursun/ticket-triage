"""Approach D: fit TF-IDF + logistic regression on train (category: gold; urgency/sentiment: silver
where available) and report dev metrics. Saves models/baseline.joblib.

Usage: python scripts/train_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.baseline import TASKS, BaselineEngine, accuracy, macro_f1  # noqa: E402
from ticket_triage.config import RESULTS  # noqa: E402
from ticket_triage.data import labelled, load_hard, load_split  # noqa: E402


def main() -> int:
    train = load_split("train")
    silver_train = labelled("train")
    # category on all of train; urgency/sentiment only on rows that have silver labels
    frame = train.assign(urgency="", sentiment="")
    if not silver_train.empty:
        frame = frame.set_index("id")
        frame.loc[silver_train["id"], ["urgency", "sentiment"]] = silver_train.set_index("id")[["urgency", "sentiment"]]
        frame = frame.reset_index()
    eng = BaselineEngine().fit(frame)
    eng.save()

    report = {}
    for name, df in (("dev", labelled("dev") if not labelled("dev").empty else load_split("dev")), ("hard_dev", load_hard("dev"))):
        if df.empty:
            continue
        pred = eng.predict_frame(df["text"].tolist())
        rep = {"n": int(len(df))}
        for task in TASKS:
            if task in df.columns and task in pred.columns and df[task].astype(str).str.len().gt(0).all():
                rep[task] = {"accuracy": round(accuracy(df[task], pred[task]), 4), "macro_f1": round(macro_f1(df[task], pred[task]), 4)}
        report[name] = rep
        print(name, json.dumps(rep))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baseline_dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
