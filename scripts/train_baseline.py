"""Approach D: fit TF-IDF + logistic regression on train (category: gold; urgency/sentiment: silver
where available) and report dev metrics. Saves models/baseline.joblib.

Usage: python scripts/train_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
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
    t0 = time.perf_counter()
    eng = BaselineEngine().fit(frame)
    train_s = time.perf_counter() - t0
    path = eng.save()
    n_labels = {t: int((frame[t].astype(str).str.len() > 0).sum()) for t in TASKS}
    report = {"train_seconds": round(train_s, 1), "model_bytes": path.stat().st_size, "n_train_rows": n_labels}
    print(f"trained in {train_s:.1f}s on {n_labels}; model {path.stat().st_size / 1e6:.1f} MB -> {path}")

    dev_silver = labelled("dev")
    # dev_silver: the 495 silver-labelled rows (all three tasks, same rows the LLM engines are scored on);
    # dev_full: every dev row, category only (gold labels exist for all of them)
    for name, df in (("dev_silver", dev_silver), ("dev_full", load_split("dev")), ("hard_dev", load_hard("dev"))):
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
