"""Token lengths of the train rows with the real tokenizer, per source and language → the default max_len.

Usage (.venv-train): python scripts/token_lengths.py [--tokenizer xlm-roberta-base]  → eval/results/train_token_lengths.{json,md}
Counts include the two special tokens, without truncation. Recommended max_len = the smallest of 128 / 192 / 256 that
covers p99 of the non-Bitext rows (synthetic + translated), the long ones that look like the hard set.
Reads train rows only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from ticket_triage.config import RESULTS  # noqa: E402
from ticket_triage.data import train_rows  # noqa: E402

CHOICES = (128, 192, 256)


def stats(lengths: list[int]) -> dict:
    a = np.asarray(lengths)
    return {"n": len(a), "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)), "p99": int(np.percentile(a, 99)), "max": int(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="xlm-roberta-base")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = train_rows()
    lengths = [len(x) for x in tok([r["text"] for r in rows], add_special_tokens=True)["input_ids"]]
    groups: dict[str, list[int]] = {}
    for r, n in zip(rows, lengths, strict=True):
        groups.setdefault(f"{r['source']} / {r['language']}", []).append(n)
    non_bitext = [n for r, n in zip(rows, lengths, strict=True) if r["source"] != "bitext"]
    res = {"tokenizer": args.tokenizer, "groups": {g: stats(v) for g, v in sorted(groups.items())},
           "non_bitext": stats(non_bitext), "all": stats(lengths)}
    p99 = res["non_bitext"]["p99"]
    res["recommended_max_len"] = next((c for c in CHOICES if c >= p99), CHOICES[-1])
    res["truncated_at_recommended"] = {g: int(sum(n > res["recommended_max_len"] for n in v)) for g, v in sorted(groups.items())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_token_lengths.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [f"# Train token lengths ({args.tokenizer}, incl. special tokens)\n", "| source / language | n | p50 | p95 | p99 | max | > max_len |",
             "|---|---|---|---|---|---|---|"]
    for g, s in res["groups"].items():
        lines.append(f"| {g} | {s['n']} | {s['p50']} | {s['p95']} | {s['p99']} | {s['max']} | {res['truncated_at_recommended'][g]} |")
    nb = res["non_bitext"]
    lines.append(f"| non-Bitext (synthetic + translated) | {nb['n']} | {nb['p50']} | {nb['p95']} | {nb['p99']} | {nb['max']} | "
                 f"{sum(n > res['recommended_max_len'] for n in non_bitext)} |")
    lines.append(f"\nRecommended max_len: **{res['recommended_max_len']}** (smallest of {'/'.join(map(str, CHOICES))} covering the non-Bitext p99 of {p99}).\n")
    md = "\n".join(lines)
    (RESULTS / "train_token_lengths.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
