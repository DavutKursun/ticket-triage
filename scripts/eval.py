"""Evaluate engines on a split's silver-labelled rows and write metrics, latency and cost per 1,000 tickets.

Usage:
  python scripts/eval.py --split dev --engines llm_zero,llm_few,baseline --delay 0.5
  python scripts/eval.py --split dev --engines baseline --dry-run          # 20 rows, no API
  python scripts/eval.py --split dev --engine finetuned --ft-model DavutKursun/ticket-triage-xlmr   # ONNX, $0 marginal

Rows: the split's silver-labelled messages (gold category + silver urgency/sentiment, data.labelled()).
Per engine, predictions stream to eval/results/<split>-<engine>.predictions.jsonl (resumable: done ids are
skipped) and metrics go to <split>-<engine>.json; <split>-summary.{json,md} compares the engines.
LLM calls go through with_backoff (per-minute 429 / 5xx retried); a daily quota stops with exit 3.
Guard: the test split is touched once, at the end — it needs data/hard/hard.csv to exist and --final.
Exit codes: 0 ok, 2 not configured / guard, 3 daily quota, 4 persistent API error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from ticket_triage.baseline import BaselineEngine  # noqa: E402
from ticket_triage.config import HARD, RESULTS  # noqa: E402
from ticket_triage.costs import cost_per_1000, cost_usd, format_cost, is_priced  # noqa: E402
from ticket_triage.data import append_jsonl, few_shot_examples, labelled, read_jsonl  # noqa: E402
from ticket_triage.llm_engine import LLMEngine  # noqa: E402
from ticket_triage.providers import ProviderError, QuotaError, with_backoff  # noqa: E402
from ticket_triage.rubric import PROMPT_VERSION  # noqa: E402

TASKS = ("category", "urgency", "sentiment")
LLM_ENGINES = {"llm_zero": "zero", "llm_few": "few"}


def build_engine(name: str, ft_model: str | None = None):
    if name == "baseline":
        return BaselineEngine.load()
    if name == "finetuned":
        from ticket_triage.ft_engine import FTEngine

        return FTEngine(ft_model)
    if name in LLM_ENGINES:
        mode = LLM_ENGINES[name]
        return LLMEngine(mode=mode, examples=few_shot_examples(2) if mode == "few" else None)
    raise SystemExit(f"unknown engine {name!r}; choose from baseline, finetuned, llm_zero, llm_few")


def predict_rows(name: str, engine, df: pd.DataFrame, pred_path: Path, delay: float) -> int:
    """Append one prediction per row to pred_path; skip ids already there. Returns 0, 3 (quota) or 4 (API error)."""
    done = {r["id"] for r in read_jsonl(pred_path)}
    todo = df[~df["id"].isin(done)]
    print(f"{name}: {len(df)} rows, {len(done)} done, {len(todo)} to do", flush=True)
    is_llm = name in LLM_ENGINES
    t_start = time.time()
    n = 0
    for n, r in enumerate(todo.itertuples()):  # noqa: B007 — n is also read in the except branches
        row: dict = {"id": r.id}
        t0 = time.perf_counter()
        try:
            if is_llm:
                call = with_backoff(partial(engine.classify, r.text))
                row.update(call.triage.model_dump(), latency_s=round(call.elapsed, 4), input_tokens=call.input_tokens,
                           output_tokens=call.output_tokens, cache_read_tokens=call.cache_read_tokens,
                           cache_write_tokens=call.cache_write_tokens, repaired=call.repaired, n_calls=len(call.results))
            else:
                t = engine.classify(r.text)
                row.update(t.model_dump(), latency_s=round(time.perf_counter() - t0, 6))
        except QuotaError as e:
            print(f"\nDaily quota reached after {n} new rows: {e}\nRerun later; progress is saved.")
            return 3
        except ProviderError as e:
            print(f"\nAPI error after {n} new rows: {e}")
            return 4
        except Exception as e:  # noqa: BLE001 — invalid twice / bad input: score as a miss, keep going
            row.update(error=f"{type(e).__name__}: {str(e)[:200]}", latency_s=round(time.perf_counter() - t0, 4))
            print(f"  failed {r.id}: {row['error'][:100]}")
        append_jsonl(pred_path, row)
        if is_llm:
            if (n + 1) % 25 == 0:
                rate = (n + 1) / (time.time() - t_start) * 60
                print(f"  {n + 1}/{len(todo)}  {rate:.0f}/min", flush=True)
            time.sleep(delay)
    return 0


def _macro_f1(y_true, y_pred) -> float:
    labels = sorted(set(y_true))  # classes present in the truth; a predicted class that never occurs still costs precision
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def score(name: str, engine, df: pd.DataFrame, pred_path: Path, split: str, delay: float) -> dict:
    preds = pd.DataFrame(read_jsonl(pred_path))
    m = df.merge(preds, on="id", how="left", suffixes=("", "_pred"))
    failed = m["category_pred"].isna() if "category_pred" in m else pd.Series(True, index=m.index)
    out: dict = {"engine": name, "split": split, "n": len(m), "n_failed": int(failed.sum()),
                 "timestamp": datetime.now(UTC).isoformat(timespec="seconds")}
    for task in TASKS:
        y_true = m[task].astype(str)
        y_pred = m.get(f"{task}_pred", pd.Series("", index=m.index)).fillna("").astype(str)
        out[task] = {"accuracy": round(float((y_true == y_pred).mean()), 4), "macro_f1": round(_macro_f1(y_true, y_pred), 4),
                     "labels_scored": sorted(set(y_true))}
    lat = m["latency_s"].dropna().astype(float) * 1000
    out["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                         "mean": round(float(lat.mean()), 1)}
    if name in LLM_ENGINES:
        model = engine.model
        ok = m[~failed]
        tin, tout = ok["input_tokens"].sum(), ok["output_tokens"].sum()
        cr, cw = ok["cache_read_tokens"].sum(), ok["cache_write_tokens"].sum()
        per = {k: round(float(v) / max(len(ok), 1), 1) for k, v in (("input", tin), ("output", tout), ("cache_read", cr), ("cache_write", cw))}
        out["model"] = model
        out["prompt_version"] = PROMPT_VERSION
        out["mode"] = engine.mode
        out["cache_control"] = engine.mode == "few"
        out["delay_s"] = delay
        out["tokens"] = {"per_call": per, "total_input": int(tin), "total_output": int(tout), "total_cache_read": int(cr),
                         "total_cache_write": int(cw), "repaired": int(ok["repaired"].sum()), "calls": int(ok["n_calls"].sum())}
        out["measured_cost_usd"] = round(cost_usd(model, int(tin), int(tout), int(cr), int(cw)), 4)
        out["cost_per_1000_usd"] = {
            "sync_no_cache": round(cost_per_1000(model, per["input"], per["output"]), 4),
            "sync_cached": round(cost_per_1000(model, per["input"], per["output"], per["cache_read"], cache_write_per_call=per["cache_write"]), 4),
            "batch_no_cache": round(cost_per_1000(model, per["input"], per["output"], batch=True), 4),
            "batch_cached": round(cost_per_1000(model, per["input"], per["output"], per["cache_read"], batch=True, cache_write_per_call=per["cache_write"]), 4),
            "priced": is_priced(model),
        }
        out["cache_hit_rate"] = round(float((ok["cache_read_tokens"] > 0).mean()), 3) if len(ok) else 0.0
    else:
        out["model"] = getattr(engine, "model_name", "tfidf_logreg")
        out["cost_per_1000_usd"] = {"sync_no_cache": 0.0, "sync_cached": 0.0, "batch_no_cache": 0.0, "batch_cached": 0.0, "priced": False}
    return out


def summary_markdown(results: list[dict], split: str) -> str:
    lines = ["| engine | category acc | category macro-F1 | urgency macro-F1 | sentiment macro-F1 | p50 / p95 ms | tokens in / out per call | $ / 1,000 sync | $ / 1,000 cached | $ / 1,000 batch |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["cost_per_1000_usd"]
        if r["engine"] in LLM_ENGINES:
            per = r["tokens"]["per_call"]
            tok = f"{per['input']:.0f} / {per['output']:.0f}" + (f" ({per['cache_read']:.0f} cached)" if per["cache_read"] else "")
            cost = (format_cost(r["model"], c["sync_no_cache"]), format_cost(r["model"], c["sync_cached"]) if r["cache_control"] else "–", format_cost(r["model"], c["batch_no_cache"]))
        else:
            tok, cost = "–", ("$0 marginal", "–", "–")
        lines.append(f"| {r['engine']} | {r['category']['accuracy']:.3f} | {r['category']['macro_f1']:.3f} | {r['urgency']['macro_f1']:.3f} | "
                     f"{r['sentiment']['macro_f1']:.3f} | {r['latency_ms']['p50']:.0f} / {r['latency_ms']['p95']:.0f} | {tok} | {cost[0]} | {cost[1]} | {cost[2]} |")
    n = results[0]["n"] if results else 0
    return (f"# {split} results — {n} silver-labelled rows (preliminary)\n\n" + "\n".join(lines) +
            "\n\nCategory is scored against Bitext gold labels; urgency and sentiment against silver labels from claude-haiku-5-5 "
            "(their own quality is measured against the 200-row hand-check). Latency is per ticket on this machine, batch size 1; "
            "LLM latency includes the network round trip. Cost is measured tokens × list prices; 'cached' = measured cache "
            "reads at 0.1× and writes at 1.25×; 'batch' = Batch API at 50% off, uncached.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--engines", "--engine", default="baseline", help="comma list of baseline,finetuned,llm_zero,llm_few")
    ap.add_argument("--ft-model", default=None, help="finetuned engine: exported folder or Hub repo id (default FT_MODEL_ID, else models/ft/onnx)")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between LLM calls")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="20 rows, no LLM engines")
    ap.add_argument("--final", action="store_true", help="required for --split test (one run at the end)")
    args = ap.parse_args()
    if args.split == "test" and not (args.final and (HARD / "hard.csv").exists()):
        print("the test split is evaluated once, at the end: needs data/hard/hard.csv and --final")
        return 2
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if args.dry_run:
        engines = [e for e in engines if e not in LLM_ENGINES]
        args.limit = args.limit or 20
    df = labelled(args.split)
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print(f"no silver-labelled rows for {args.split}")
        return 2
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}" + (f"-limit{args.limit}" if args.limit else "")
    results = []
    for name in engines:
        try:
            engine = build_engine(name, args.ft_model)
        except (ProviderError, FileNotFoundError) as e:
            print(f"{name}: not available: {e}")
            return 2
        pred_path = RESULTS / f"{tag}-{name}.predictions.jsonl"
        rc = predict_rows(name, engine, df, pred_path, args.delay)
        if rc:
            return rc
        res = score(name, engine, df, pred_path, args.split, args.delay)
        (RESULTS / f"{tag}-{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        results.append(res)
        extra = f", cost {format_cost(res['model'], res['measured_cost_usd'])}, cache hit rate {res['cache_hit_rate']}" if name in LLM_ENGINES else ""
        print(f"{name}: category acc {res['category']['accuracy']} F1 {res['category']['macro_f1']} | urgency F1 {res['urgency']['macro_f1']} | "
              f"sentiment F1 {res['sentiment']['macro_f1']} | p50 {res['latency_ms']['p50']} ms p95 {res['latency_ms']['p95']} ms | failed {res['n_failed']}{extra}")
    (RESULTS / f"{tag}-summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = summary_markdown(results, args.split)
    (RESULTS / f"{tag}-summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
