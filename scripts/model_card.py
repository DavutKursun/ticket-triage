"""Render the Hugging Face model card (README.md) for a trained + exported model folder.

Usage: python scripts/model_card.py MODEL_DIR [--repo-id DavutKursun/ticket-triage-xlmr] [--github https://github.com/DavutKursun/ticket-triage]
Reads ft_config.json, labels.json, dev_metrics.json and, when present, train_meta.json and export_report.json.
Writes MODEL_DIR/README.md. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.rubric import CATEGORY_DEFINITIONS, PROMPT_VERSION  # noqa: E402

URGENCY_MEANING = {"urgent": "service down, money being lost right now, or a hard deadline today",
                   "high": "the customer is blocked (cannot log in, cannot pay, order stuck)",
                   "medium": "an inconvenience or correction that can wait a few days",
                   "low": "a question, information request or feedback with no time pressure"}
SENTIMENT_MEANING = {"negative": "frustration, anger, disappointment", "neutral": "plain request or question",
                     "positive": "thanks, praise, friendly tone"}
DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def _n(v) -> str:
    return f"{v:,}" if isinstance(v, int) else "?"


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _metrics_table(m: dict) -> str:
    rows = ["| task | macro-F1 | accuracy | rows | labels scored |", "|---|---|---|---|---|"]
    for t in ("category", "urgency", "sentiment"):
        if t in m:
            rows.append(f"| {t} | {m[t]['macro_f1']:.3f} | {m[t]['accuracy']:.3f} | {m[t]['n']} | {', '.join(m[t]['labels_scored'])} |")
    return "\n".join(rows)


def render(md: Path, repo_id: str, github: str) -> str:
    cfg, labels, dev = _load(md / "ft_config.json"), _load(md / "labels.json"), _load(md / "dev_metrics.json")
    meta, exp = _load(md / "train_meta.json") or {}, _load(md / "export_report.json") or {}
    base = cfg["model_name"]
    smoke = bool(cfg.get("tiny")) or "tiny" in base
    by_src = dev.get("rows_by_source", {})
    lab_src = dev.get("labelled_by_source", {})
    best = dev["best"]
    yaml = "\n".join(["---", "language:", "- en", "- tr", "license: mit", f"base_model: {base}", "datasets:", f"- {DATASET}",
                      "pipeline_tag: text-classification", "tags:", "- customer-support", "- multi-task", "- onnx", "---", ""])
    out = [yaml, "# ticket-triage-xlmr — category, urgency and sentiment for support tickets\n"]
    if smoke:
        out.append("> **Smoke-test build.** This card was rendered from a tiny random model that only proves the pipeline. "
                   "Its numbers mean nothing.\n")
    out.append(f"`{base}` fine-tuned with three classification heads on one shared encoder, exported to int8 ONNX for CPU serving. "
               f"Part of [ticket-triage]({github}), which compares this model with an LLM (Claude Haiku 5.5) on accuracy, latency "
               "and cost per 1,000 tickets.\n")
    out.append("## Heads\n")
    out.append("| head | labels | meaning |\n|---|---|---|")
    out.append(f"| category | {len(labels['category'])} | where to route the ticket: " + "; ".join(f"`{k}` {v}" for k, v in CATEGORY_DEFINITIONS.items()) + " |")
    out.append("| urgency | 4 | judged by consequence, not tone: " + "; ".join(f"`{k}` {v}" for k, v in URGENCY_MEANING.items()) + " |")
    out.append("| sentiment | 3 | tone only: " + "; ".join(f"`{k}` {v}" for k, v in SENTIMENT_MEANING.items()) + " |\n")
    out.append("`confidence` is the category head's top probability. A two-issue message takes the category of the issue that needs action first.\n")
    out.append("## Training data\n")
    out.append(f"- **Category**: gold labels from [{DATASET}](https://huggingface.co/datasets/{DATASET}) (CDLA-Sharing-1.0), "
               f"{_n(by_src.get('bitext'))} deduplicated train rows, 11 categories.")
    out.append(f"- **Urgency and sentiment**: silver labels from `claude-haiku-5-5` with a written rubric (prompt {PROMPT_VERSION}) on "
               f"{_n(lab_src.get('bitext'))} Bitext train rows; their agreement with 200 hand-checked rows is reported in the GitHub README.")
    out.append(f"- **Synthetic, train only**: {_n(by_src.get('synthetic', dev.get('synthetic_rows')))} messages written by Haiku from the rubric and kept only when the "
               "silver labeller agreed blind. They add the classes Bitext lacks (urgent, high, positive) and an equal-style control set of "
               "long, calm low/medium tickets, so neither language nor length predicts the label.")
    out.append(f"- **Translated, train only**: {_n(by_src.get('translated', dev.get('translated_rows')))} Bitext rows translated into customer-style Turkish "
               "(labels carried over; ~20% typed without Turkish characters).")
    out.append("- Synthetic and translated rows never enter dev, test or the hand-written hard set.\n")
    out.append("## Training\n")
    c = dev["config"]
    out.append(f"Multi-task cross-entropy (missing urgency/sentiment labels masked, inverse-sqrt class weights), labelled rows oversampled "
               f"×{c['oversample']}, AdamW lr {c['lr']}, warmup {c['warmup']:.0%}, batch {c['batch_size']}, {c['epochs']} epochs, "
               f"max_len {c['max_len']}, seed {c['seed']}; best epoch ({dev['best_epoch']}) chosen by the mean dev macro-F1."
               + (f" Trained on {meta.get('gpu')} in {meta.get('wall_seconds', 0) / 60:.0f} min (torch {meta.get('torch')}, "
                  f"transformers {meta.get('transformers')}, commit `{(meta.get('git_commit') or '')[:7]}`)." if meta.get("gpu") else "") + "\n")
    out.append("## Dev results (silver labels, preliminary)\n")
    out.append("Category against Bitext gold; urgency and sentiment against the silver labels, which come from an LLM. Final numbers on the "
               "held-out test set and the hand-labelled hard set are published in the GitHub README.\n")
    out.append(_metrics_table(best) + "\n")
    if dev.get("best_hard_dev"):
        out.append("Hard set, dev half (hand labels, report only, not used for model selection):\n")
        out.append(_metrics_table(dev["best_hard_dev"]) + "\n")
    if exp:
        q = exp["int8"]
        size = exp["sizes_bytes"].get("onnx/model.int8.onnx", 0) / 1e6
        lat = exp.get("latency_ms", {}).get("onnx_int8", {})
        out.append("## ONNX\n")
        out.append(f"`onnx/model.int8.onnx` ({size:.0f} MB, onnxruntime dynamic quantisation, {q['kept']['weight_type']} weights"
                   f"{', per channel' if q['kept']['per_channel'] else ''}). Against the PyTorch model on dev: fp32 ONNX max |Δlogit| "
                   f"{exp['fp32']['max_abs_logit_diff']['all']:.1e}; int8 argmax agreement category {q['argmax_agreement']['category']:.1%}, "
                   f"urgency {q['argmax_agreement']['urgency']:.1%}, sentiment {q['argmax_agreement']['sentiment']:.1%}."
                   + (f" CPU latency per ticket (batch 1, {exp['latency_ms']['threads']} threads, export machine): p50 "
                      f"{lat['dev_short']['p50']:.0f} ms on short tickets, {lat['long_synthetic']['p50']:.0f} ms on long ones." if lat else "") + "\n")
    out.append("```python\nimport json, numpy as np, onnxruntime as ort\nfrom huggingface_hub import snapshot_download\nfrom tokenizers import Tokenizer\n\n"
               f'path = snapshot_download("{repo_id}", allow_patterns=["onnx/*"]) + "/onnx"\n'
               'cfg, labels = (json.load(open(f"{path}/{f}")) for f in ("ft_config.json", "labels.json"))\n'
               'tok = Tokenizer.from_file(f"{path}/tokenizer.json"); tok.enable_truncation(cfg["max_len"])\n'
               'sess = ort.InferenceSession(f"{path}/model.int8.onnx", providers=["CPUExecutionProvider"])\n\n'
               'enc = tok.encode("I was charged twice for order #48213 and need the duplicate refunded today")\n'
               'feeds = {"input_ids": np.array([enc.ids], dtype=np.int64), "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}\n'
               'for task, logits in zip(("category", "urgency", "sentiment"), sess.run(None, feeds)):\n'
               '    p = np.exp(logits[0] - logits[0].max()); p /= p.sum()\n'
               '    print(task, labels[task][int(p.argmax())], round(float(p.max()), 3))\n```\n')
    out.append("The full PyTorch weights (`model.safetensors`, `encoder/`, `tokenizer/`) are in this repo too; "
               f"`ticket_triage.ft_model.load_model` in the [GitHub repo]({github}) rebuilds the module.\n")
    out.append("## Limitations\n")
    out.append("- Bitext messages are templated one-liners, so dev scores overstate real-world accuracy; trust the hard-set numbers in the "
               "GitHub README.\n- Urgency and sentiment were learned from LLM labels, not human ones, and inherit their biases; Bitext has "
               "almost no urgent or positive messages, so those classes come mostly from synthetic rows.\n- Turkish comes from synthetic "
               "messages and machine translations, not real Turkish tickets.\n- Inputs are truncated to "
               f"{c['max_len']} tokens; the start of a long message decides.\n- int8 dynamic quantisation scales activations per batch, "
               "so batched outputs can differ slightly from single-message ones.\n- Built to route and prioritise tickets for human agents, "
               "not to make decisions about customers without review.\n")
    out.append(f"## License\n\nModel and code: MIT. Training data derived from Bitext: CDLA-Sharing-1.0. Source: {github}\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--repo-id", default="DavutKursun/ticket-triage-xlmr")
    ap.add_argument("--github", default="https://github.com/DavutKursun/ticket-triage")
    a = ap.parse_args()
    md = Path(a.model_dir)
    text = render(md, a.repo_id, a.github)
    (md / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {md / 'README.md'} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
