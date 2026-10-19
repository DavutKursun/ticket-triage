"""Approach C: fine-tune the multi-task encoder (thin wrapper around ticket_triage.ft_model; the Kaggle notebook
calls the same module).

Usage: python scripts/train.py [--model-name xlm-roberta-base] [--epochs 3] [--max-len 128] [--batch-size 32] [--lr 2e-5]
                               [--limit N] [--out models/ft] [--no-synth] [--no-class-weights] [--oversample 4] [--seed 7]
Smoke (CPU, seconds): python scripts/train.py --tiny --limit 64 --epochs 1 --out models/smoke
  --tiny builds a 2-layer random XLM-R from the tokenizer-only repo hf-internal-testing/tiny-random-xlm-roberta (no weights download).
Needs the training venv (requirements-train.txt). Never reads the test split or the hard set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.ft_model import TINY_TOKENIZER, TrainConfig, train  # noqa: E402


def main() -> int:
    d = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=d.model_name)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--max-len", type=int, default=d.max_len)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--limit", type=int, default=0, help="train/dev rows for smoke tests (labelled + unlabelled mix)")
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--no-synth", action="store_true", help="leave the synthetic train rows out")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--oversample", type=int, default=d.oversample)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--fp16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--tiny", action="store_true", help="smoke test: tiny random encoder built locally from the tiny tokenizer")
    a = ap.parse_args()
    if a.tiny and a.model_name == d.model_name:
        a.model_name = TINY_TOKENIZER
    cfg = TrainConfig(model_name=a.model_name, epochs=a.epochs, max_len=a.max_len, batch_size=a.batch_size, lr=a.lr, limit=a.limit,
                      out=a.out, no_synth=a.no_synth, class_weights=not a.no_class_weights, oversample=a.oversample, seed=a.seed,
                      fp16=None if a.fp16 == "auto" else a.fp16 == "on", tiny=a.tiny)
    summary = train(cfg)
    print(json.dumps({"best_epoch": summary["best_epoch"], "best": summary["best"], "out": cfg.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
