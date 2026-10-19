"""Token lengths of the train rows with the real tokenizer, per source and language → the default max_len.

Usage (.venv-train): python scripts/token_lengths.py [--tokenizer xlm-roberta-base]  → eval/results/train_token_lengths.{json,md}
Counts include the two special tokens, without truncation. Reports the rows truncated at 128 / 192 / 256, the smallest
of those covering p99 of the non-Bitext rows (synthetic + translated), and the configured default (TrainConfig.max_len,
192 since M3-2b: headroom for six-sentence hard-set tickets at inference; collate pads per batch, so training barely pays).
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
from ticket_triage.ft_model import TrainConfig  # noqa: E402

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
    res["smallest_covering_p99"] = next((c for c in CHOICES if c >= p99), CHOICES[-1])
    res["configured_max_len"] = TrainConfig().max_len
    res["truncated_at"] = {c: {g: int(sum(n > c for n in v)) for g, v in sorted(groups.items())} | {"all": int(sum(n > c for n in lengths))}
                           for c in CHOICES}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_token_lengths.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    head = " | ".join(f"> {c}" for c in CHOICES)
    lines = [f"# Train token lengths ({args.tokenizer}, incl. special tokens)\n", f"| source / language | n | p50 | p95 | p99 | max | {head} |",
             "|---|---|---|---|---|---|" + "---|" * len(CHOICES)]
    for g, s in res["groups"].items():
        cut = " | ".join(str(res["truncated_at"][c][g]) for c in CHOICES)
        lines.append(f"| {g} | {s['n']} | {s['p50']} | {s['p95']} | {s['p99']} | {s['max']} | {cut} |")
    nb = res["non_bitext"]
    cut = " | ".join(str(sum(n > c for n in non_bitext)) for c in CHOICES)
    lines.append(f"| non-Bitext (synthetic + translated) | {nb['n']} | {nb['p50']} | {nb['p95']} | {nb['p99']} | {nb['max']} | {cut} |")
    lines.append(f"\nThe smallest of {'/'.join(map(str, CHOICES))} covering the non-Bitext p99 of {p99} is {res['smallest_covering_p99']}; "
                 f"the configured default is **{res['configured_max_len']}** (headroom for six-sentence hard-set tickets at inference). "
                 f"Train rows truncated: " + ", ".join(f"{res['truncated_at'][c]['all']} at {c}" for c in CHOICES) + ".\n")
    md = "\n".join(lines)
    (RESULTS / "train_token_lengths.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
