"""Export a saved TriageModel to ONNX: fp32 (stays local) and int8 dynamic quantisation (uploaded and served), with a
parity report.

Usage (.venv-train): python scripts/export_onnx.py MODEL_DIR [--opset 17] [--limit N] [--threads 2] [--latency-n 200]
Writes
  MODEL_DIR/onnx/              model.int8.onnx, tokenizer.json, labels.json, ft_config.json  (self-contained, what serves)
  MODEL_DIR/fp32/model.onnx    local only, never uploaded
  MODEL_DIR/export_report.json max |Δlogit| PyTorch vs ONNX fp32, argmax agreement per head (fp32 and int8 vs PyTorch),
                               dev macro-F1 PyTorch vs int8, file sizes, tokenizer parity, CPU latency (batch 1, this machine)
Exporter: torch.onnx.export, opset 17, dynamic batch and sequence axes; the default exporter first, dynamo=False when it
fails or its graph does not match PyTorch on two input shapes. Quantisation: onnxruntime quantize_dynamic with QInt8
weights; if int8 category agreement with PyTorch is below 98% it tries per_channel=True, then QUInt8, and keeps the
first that passes (else the best). Reads the dev split, the silver dev rows and the Turkish train rows only.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone  # timezone.utc, not datetime.UTC: Kaggle may run Python 3.10
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ticket_triage.ft_model import LABELS, TASKS, load_model  # noqa: E402

INPUTS = ["input_ids", "attention_mask"]
OUTPUTS = list(TASKS)
AXES = {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, **{t: {0: "batch"} for t in TASKS}}
QUANT_LADDER = (("QInt8", False), ("QInt8", True), ("QUInt8", False))
AGREEMENT_TARGET = 0.98
SHAPE_TEXTS = (["ok"], ["refund please", "where is my parcel, it has been two weeks and nobody answers my emails at all",
                        "i was charged twice for order 48213 and i want the duplicate payment back today"])


def encode(tok, texts: list[str], max_len: int):
    enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


@torch.no_grad()
def torch_logits(model, tok, texts: list[str], max_len: int, bs: int = 32) -> dict[str, np.ndarray]:
    out: dict[str, list] = {t: [] for t in TASKS}
    for b in range(0, len(texts), bs):
        ids, mask = encode(tok, texts[b : b + bs], max_len)
        for t, x in zip(TASKS, model(torch.from_numpy(ids), torch.from_numpy(mask)), strict=True):
            out[t].append(x.float().numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


def ort_session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(sess, tok, texts: list[str], max_len: int, bs: int = 32) -> dict[str, np.ndarray]:
    out: dict[str, list] = {t: [] for t in TASKS}
    names = [o.name for o in sess.get_outputs()]
    for b in range(0, len(texts), bs):
        ids, mask = encode(tok, texts[b : b + bs], max_len)
        for name, x in zip(names, sess.run(None, {"input_ids": ids, "attention_mask": mask}), strict=True):
            out[name].append(np.asarray(x, dtype=np.float32))
    return {t: np.concatenate(v) for t, v in out.items()}


def max_abs(a: dict, b: dict) -> dict[str, float]:
    d = {t: float(np.abs(a[t] - b[t]).max()) for t in TASKS}
    return {**d, "all": max(d.values())}


def agreement(a: dict, b: dict) -> dict[str, float]:
    return {t: round(float((a[t].argmax(1) == b[t].argmax(1)).mean()), 4) for t in TASKS}


def _clean(path: Path) -> None:
    for p in path.parent.glob(path.name + "*"):
        p.unlink()


def export_fp32(model, tok, path: Path, opset: int, max_len: int) -> dict:
    """Default exporter first, then dynamo=False; each result must run on two shapes and match PyTorch within 1e-3."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, mask = encode(tok, SHAPE_TEXTS[1][:2], max_len)
    args = (torch.from_numpy(ids), torch.from_numpy(mask))
    errors: dict[str, str] = {}
    for name, kw in (("default", {}), ("torchscript (dynamo=False)", {"dynamo": False})):
        _clean(path)
        try:
            torch.onnx.export(model, args, str(path), input_names=INPUTS, output_names=OUTPUTS, opset_version=opset, dynamic_axes=AXES, **kw)
            sess = ort_session(path)
            worst = max(max_abs(torch_logits(model, tok, t, max_len), ort_logits(sess, tok, t, max_len))["all"] for t in SHAPE_TEXTS)
            if worst > 1e-3:
                raise RuntimeError(f"graph does not match PyTorch on the shape check (max |Δlogit| {worst:.2e})")
            return {"exporter": name, "errors": errors, "shape_check_max_abs_diff": worst}
        except Exception as e:  # noqa: BLE001 — try the next exporter
            errors[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
            print(f"  exporter {name!r} failed: {errors[name]}", flush=True)
    raise RuntimeError(f"every exporter failed: {errors}")


def quantize(fp32: Path, out: Path, weight_type: str, per_channel: bool) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.unlink(missing_ok=True)
    quantize_dynamic(str(fp32), str(out), weight_type=getattr(QuantType, weight_type), per_channel=per_channel)


def macro_f1s(logits: dict, cat_idx: list[int], cat_true: list[str], us_idx: list[int], us_true: dict[str, list[str]]) -> dict:
    from sklearn.metrics import f1_score

    def f1(y, p):
        return round(float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)), 4)

    out = {"category": f1(cat_true, [LABELS["category"][k] for k in logits["category"][cat_idx].argmax(1)])}
    for t in ("urgency", "sentiment"):
        out[t] = f1(us_true[t], [LABELS[t][k] for k in logits[t][us_idx].argmax(1)])
    return out


def _spread(xs: list, k: int) -> list:
    return xs[:: max(1, len(xs) // k)][:k] if k else xs


def _lat(fn, texts: list[str], warmup: int = 10) -> dict:
    for t in texts[:warmup]:
        fn(t)
    ms = []
    for t in texts:
        t0 = time.perf_counter()
        fn(t)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"n": len(ms), "p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0, help="dev rows to score (0 = all 2,455 + 495 silver)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for the latency measurement")
    ap.add_argument("--latency-n", type=int, default=200)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    from ticket_triage.data import labelled, load_split, train_rows
    from ticket_triage.ft_engine import FTEngine

    md = Path(args.model_dir)
    model, tok, _labels, cfg = load_model(md, "cpu")
    model.float().eval()
    max_len = int(cfg["max_len"])
    onnx_dir, fp32 = md / "onnx", md / "fp32" / "model.onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"exporting {md} ({cfg['model_name']}, max_len {max_len}, opset {args.opset})", flush=True)
    exp = export_fp32(model, tok, fp32, args.opset, max_len)
    print(f"  fp32 exported with the {exp['exporter']} exporter (shape check max |Δlogit| {exp['shape_check_max_abs_diff']:.2e})", flush=True)

    # ---- dev texts: category on the dev split, urgency/sentiment on the silver dev rows ----
    dev, silver = load_split("dev"), labelled("dev")
    if args.limit:
        dev, silver = dev.iloc[_spread(list(range(len(dev))), args.limit)], silver.iloc[_spread(list(range(len(silver))), args.limit)]
    texts = list(dict.fromkeys(dev["text"].tolist() + silver["text"].tolist()))
    pos = {t: k for k, t in enumerate(texts)}
    cat_idx, us_idx = [pos[t] for t in dev["text"]], [pos[t] for t in silver["text"]]
    cat_true, us_true = dev["category"].tolist(), {t: silver[t].tolist() for t in ("urgency", "sentiment")}
    ref = torch_logits(model, tok, texts, max_len)
    f32 = ort_logits(ort_session(fp32), tok, texts, max_len)
    report: dict = {"model_dir": str(md), "base_model": cfg["model_name"], "max_len": max_len, "opset": args.opset, **exp,
                    "dev_rows": {"category": len(cat_idx), "urgency_sentiment": len(us_idx), "texts_scored": len(texts)},
                    "fp32": {"max_abs_logit_diff": max_abs(ref, f32), "argmax_agreement": agreement(ref, f32)}}
    print(f"  fp32 vs PyTorch on {len(texts)} dev texts: max |Δlogit| {report['fp32']['max_abs_logit_diff']['all']:.2e}, "
          f"agreement {report['fp32']['argmax_agreement']}", flush=True)

    # ---- int8: QInt8, then per_channel, then QUInt8 until category agreement >= 98% ----
    int8 = onnx_dir / "model.int8.onnx"
    attempts, best = [], None
    for wt, pc in QUANT_LADDER:
        cand = onnx_dir / f"model.int8.{wt}{'.pc' if pc else ''}.onnx"
        quantize(fp32, cand, wt, pc)
        q = ort_logits(ort_session(cand), tok, texts, max_len)
        att = {"weight_type": wt, "per_channel": pc, "argmax_agreement": agreement(ref, q), "max_abs_logit_diff": max_abs(ref, q)["all"],
               "bytes": cand.stat().st_size}
        attempts.append(att)
        print(f"  int8 {wt} per_channel={pc}: agreement {att['argmax_agreement']}, {att['bytes'] / 1e6:.2f} MB", flush=True)
        if best is None or att["argmax_agreement"]["category"] > best[0]["argmax_agreement"]["category"]:
            best = (att, cand, q)
        if att["argmax_agreement"]["category"] >= AGREEMENT_TARGET:
            break
    kept, cand, q8 = best
    shutil.move(str(cand), int8)
    for p in onnx_dir.glob("model.int8.*.onnx"):
        p.unlink()
    report["int8"] = {"kept": {"weight_type": kept["weight_type"], "per_channel": kept["per_channel"],
                               "meets_target": kept["argmax_agreement"]["category"] >= AGREEMENT_TARGET},
                      "target_category_agreement": AGREEMENT_TARGET, "argmax_agreement": kept["argmax_agreement"], "attempts": attempts}
    report["dev_macro_f1"] = {"pytorch": macro_f1s(ref, cat_idx, cat_true, us_idx, us_true),
                              "onnx_fp32": macro_f1s(f32, cat_idx, cat_true, us_idx, us_true),
                              "onnx_int8": macro_f1s(q8, cat_idx, cat_true, us_idx, us_true)}

    # ---- self-contained serving folder ----
    shutil.copy(md / "tokenizer" / "tokenizer.json", onnx_dir / "tokenizer.json")
    shutil.copy(md / "labels.json", onnx_dir / "labels.json")
    serve_cfg = {**cfg, "max_len": max_len, "onnx_file": "model.int8.onnx", "inputs": INPUTS, "outputs": OUTPUTS, "opset": args.opset,
                 "exporter": exp["exporter"], "quantization": {**report["int8"]["kept"], "method": "onnxruntime quantize_dynamic"},
                 "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id}
    (onnx_dir / "ft_config.json").write_text(json.dumps(serve_cfg, indent=2), encoding="utf-8")

    # ---- tokenizer parity: tokenizers.Tokenizer(tokenizer.json) vs the transformers tokenizer ----
    tk = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
    tk.enable_truncation(max_length=max_len)
    turkish = [r["text"] for r in train_rows() if r["language"] == "tr"]
    parity_texts = _spread(dev["text"].tolist(), 200) + turkish + ["refund " * 715]
    mism = [k for k, t in enumerate(parity_texts) if tk.encode(t).ids != tok(t, truncation=True, max_length=max_len)["input_ids"]]
    report["tokenizer_parity"] = {"n": len(parity_texts), "dev": min(200, len(dev)), "turkish": len(turkish), "long_5000_chars": 1,
                                  "mismatches": len(mism), "first_mismatch_index": mism[0] if mism else None}
    print(f"  tokenizer parity: {len(parity_texts) - len(mism)}/{len(parity_texts)} identical ({len(turkish)} Turkish rows)", flush=True)

    # ---- sizes and CPU latency (batch 1, end to end: tokenize + run + softmax) ----
    sizes = {p.relative_to(md).as_posix(): p.stat().st_size for p in sorted(md.rglob("*")) if p.is_file() and p.name != "export_report.json"}
    report["sizes_bytes"] = {**sizes, "onnx/ (total)": sum(v for k, v in sizes.items() if k.startswith("onnx/"))}
    eng = FTEngine(onnx_dir, threads=args.threads)
    long_texts = [r["text"] for r in train_rows() if r["source"] != "bitext"]
    half = max(1, args.latency_n // 2)
    short, long_ = _spread(dev["text"].tolist(), half), _spread(long_texts, half)
    torch.set_num_threads(args.threads)

    def pt(text):
        return torch_logits(model, tok, [text], max_len)

    report["latency_ms"] = {"machine": f"this machine: {platform.platform()}, {platform.processor() or platform.machine()}",
                            "threads": args.threads, "batch": 1, "measured": "end to end per ticket: tokenize + run + softmax",
                            "onnx_int8": {"dev_short": _lat(eng.classify, short), "long_synthetic": _lat(eng.classify, long_)},
                            "pytorch_fp32": {"dev_short": _lat(pt, short), "long_synthetic": _lat(pt, long_)}}
    report["seconds"] = round(time.time() - t_start, 1)
    report["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
    (md / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    L = report["latency_ms"]
    print(f"  int8 kept: {report['int8']['kept']} | dev macro-F1 PyTorch {report['dev_macro_f1']['pytorch']} vs int8 {report['dev_macro_f1']['onnx_int8']}")
    print(f"  sizes: int8 {int8.stat().st_size / 1e6:.2f} MB, onnx/ total {report['sizes_bytes']['onnx/ (total)'] / 1e6:.2f} MB, fp32 {fp32.stat().st_size / 1e6:.2f} MB")
    print(f"  latency p50/p95 ms ({args.threads} threads, batch 1): int8 short {L['onnx_int8']['dev_short']['p50']}/{L['onnx_int8']['dev_short']['p95']}, "
          f"long {L['onnx_int8']['long_synthetic']['p50']}/{L['onnx_int8']['long_synthetic']['p95']} | PyTorch short "
          f"{L['pytorch_fp32']['dev_short']['p50']}/{L['pytorch_fp32']['dev_short']['p95']}")
    print(f"wrote {onnx_dir}/ and {md / 'export_report.json'} in {report['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
