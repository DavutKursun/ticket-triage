"""Approach C: fine-tune the multi-task encoder (thin wrapper around ticket_triage.ft_model; the Kaggle notebook
calls the same module).

Usage: python scripts/train.py [--model-name xlm-roberta-base] [--epochs 3] [--max-len N] [--batch-size 32] [--lr 2e-5]
                               [--limit N] [--out models/ft] [--no-synth] [--no-translated] [--no-class-weights] [--oversample 4] [--seed 7]
Writes <out>/train_meta.json next to the model: git commit, GPU, torch/transformers/CUDA versions, wall time, peak GPU memory.
Smoke (CPU, seconds): python scripts/train.py --tiny --limit 64 --epochs 1 --out models/smoke
  --tiny builds a 2-layer random XLM-R from the tokenizer-only repo hf-internal-testing/tiny-random-xlm-roberta (no weights download).
Needs the training venv (requirements-train.txt). Never reads the test split or the hard set.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone  # timezone.utc, not datetime.UTC: Kaggle may run Python 3.10
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
    ap.add_argument("--no-translated", action="store_true", help="leave the translated Turkish train rows out")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--oversample", type=int, default=d.oversample)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--fp16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--tiny", action="store_true", help="smoke test: tiny random encoder built locally from the tiny tokenizer")
    a = ap.parse_args()
    if a.tiny and a.model_name == d.model_name:
        a.model_name = TINY_TOKENIZER
    cfg = TrainConfig(model_name=a.model_name, epochs=a.epochs, max_len=a.max_len, batch_size=a.batch_size, lr=a.lr, limit=a.limit,
                      out=a.out, no_synth=a.no_synth, no_translated=a.no_translated, class_weights=not a.no_class_weights, oversample=a.oversample, seed=a.seed,
                      fp16=None if a.fp16 == "auto" else a.fp16 == "on", tiny=a.tiny)
    started = datetime.now(timezone.utc)  # noqa: UP017
    t0 = time.time()
    summary = train(cfg)
    write_train_meta(Path(cfg.out), started, time.time() - t0)
    print(json.dumps({"best_epoch": summary["best_epoch"], "best": summary["best"], "out": cfg.out}, indent=2))
    return 0


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_train_meta(out: Path, started: datetime, wall_s: float) -> Path:
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    meta = {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "started_utc": started.isoformat(timespec="seconds"), "wall_seconds": round(wall_s, 1),
            "gpu": torch.cuda.get_device_name(0) if cuda else None, "gpu_count": torch.cuda.device_count() if cuda else 0,
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda if cuda else None,
            "cudnn": torch.backends.cudnn.version() if cuda else None, "python": platform.python_version(), "platform": platform.platform()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"train_meta.json: {meta['wall_seconds']} s, gpu {meta['gpu']}, peak GPU memory {meta['peak_gpu_memory_mb']} MB, commit {meta['git_commit']}", flush=True)
    return out / "train_meta.json"


if __name__ == "__main__":
    raise SystemExit(main())
