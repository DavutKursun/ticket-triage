"""Turkish translations of silver-labelled Bitext TRAIN rows (train only; labels carry over from the source row).

Usage: python scripts/translate_train.py [--n 300] [--seed 7] [--ascii-share 0.2] [--max-usd 0.08] [--out data/silver/train_translated.jsonl]

Selection (deterministic from --seed): a stratified --n of the silver-labelled train rows, ~n/11 per category (the
remainder goes to randomly chosen categories), urgency in proportion within each category. Haiku translates 10 rows
per call into natural customer-style Turkish that keeps meaning, request, urgency and tone, and keeps {{placeholders}}.
A seeded --ascii-share of the rows is then typed without Turkish characters (ç→c, ğ→g, ı→i, ö→o, ş→s, ü→u), as people
do on phones. Rows: id, text, category, urgency, sentiment, language="tr", source="translated", source_id,
ascii_folded, model, translate_version, silver_prompt_version. Never part of dev, test or the hard set.
Resumable via <out stem>.progress.json (gitignored).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from ticket_triage.config import SEED, settings  # noqa: E402
from ticket_triage.costs import cost_usd, format_cost  # noqa: E402
from ticket_triage.data import labelled, stable_id  # noqa: E402
from ticket_triage.providers import ProviderError, complete_validated, get_provider, with_backoff  # noqa: E402
from ticket_triage.schema import CATEGORIES  # noqa: E402

TRANSLATE_VERSION = "2026-10-19.1"
BATCH = 10
MAX_TOKENS = 8000
FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
PLACEHOLDER = re.compile(r"\{\{[^{}]+\}\}")


class Item(BaseModel):
    i: int
    text: str


class Batch(BaseModel):
    items: list[Item] = Field(min_length=1)


SYSTEM = """You translate English customer-support messages into Turkish for a ticket classifier's training data.
For each numbered message write the Turkish a real customer would type into a support chat or e-mail:
- natural everyday Turkish, not a literal or formal translation; casual or terse when the English is casual or terse;
- keep the meaning, the request, the urgency and the tone exactly: add no complaint, politeness, urgency or detail,
  and drop none;
- keep placeholders such as {{Order Number}} or {{Account Type}} exactly as written, untranslated;
- roughly the same length; proper Turkish characters (ç ğ ı ö ş ü).
Return JSON {"items": [{"i": <number>, "text": "<Turkish>"}, ...]}: one item per message, same order, echo i."""


def select(n: int, seed: int):
    """~n/11 per category (extras to random categories), urgency in proportion within each category."""
    df = labelled("train")
    rng = random.Random(seed)
    per = {c: n // len(CATEGORIES) for c in CATEGORIES}
    for c in rng.sample(list(CATEGORIES), n - sum(per.values())):
        per[c] += 1
    picked = []
    for c in CATEGORIES:
        g = df[df["category"] == c]
        shares = g["urgency"].value_counts(normalize=True).to_dict()
        raw = {u: per[c] * v for u, v in shares.items()}
        counts = {u: int(x) for u, x in raw.items()}
        for u in sorted(raw, key=lambda u: raw[u] - counts[u], reverse=True)[: per[c] - sum(counts.values())]:
            counts[u] += 1
        for u, k in sorted(counts.items()):
            sub = g[g["urgency"] == u]
            picked.append(sub.sample(min(k, len(sub)), random_state=rng.randint(0, 10**6)))
    import pandas as pd

    sel = pd.concat(picked).reset_index(drop=True)
    fold_ids = set(sel["id"].sample(round(len(sel) * ARGS.ascii_share), random_state=seed))
    return sel, fold_ids


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ascii-share", type=float, default=0.2)
    ap.add_argument("--max-usd", type=float, default=0.08)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="data/silver/train_translated.jsonl")
    ARGS = args = ap.parse_args()
    try:
        provider, model = get_provider(settings)
    except ProviderError as e:
        print("not configured:", e)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = out.with_name(out.stem + ".progress.json")
    sel, fold_ids = select(args.n, args.seed)
    src = {r.id: r for r in sel.itertuples()}
    state = {"n": args.n, "seed": args.seed, "done": {}, "tokens_in": 0, "tokens_out": 0, "calls": 0, "repaired": 0, "rejected": 0}
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if (saved.get("n"), saved.get("seed")) == (args.n, args.seed):
            state = saved
            print(f"resuming: {len(state['done'])} translated")

    def spent() -> float:
        return cost_usd(model, state["tokens_in"], state["tokens_out"])

    order = list(src)
    for round_ in range(args.rounds):
        pending = [i for i in order if i not in state["done"]]
        if not pending:
            break
        for b in range(0, len(pending), BATCH):
            if spent() >= args.max_usd:
                print(f"budget ${args.max_usd} reached — stopping (progress saved)")
                break
            ids = pending[b : b + BATCH]
            user = "Translate these messages.\n" + "\n".join(f"{k}. i={k} {src[sid].text}" for k, sid in enumerate(ids))
            try:
                obj, results, rep = with_backoff(partial(complete_validated, provider, SYSTEM, user, model, Batch, max_tokens=MAX_TOKENS))
            except ProviderError as e:
                print("API error (progress saved):", e)
                progress.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                return 4
            except Exception as e:  # noqa: BLE001 — invalid twice: these rows come back next round
                print(f"  batch skipped: {type(e).__name__}: {str(e)[:120]}")
                continue
            state["calls"] += len(results)
            state["repaired"] += int(rep)
            state["tokens_in"] += sum(r.input_tokens for r in results)
            state["tokens_out"] += sum(r.output_tokens for r in results)
            for it in obj.items:
                if not 0 <= it.i < len(ids):
                    continue
                sid = ids[it.i]
                tr = " ".join(it.text.split())
                english = src[sid].text
                if (sid in state["done"] or len(tr) < 3 or tr.strip().lower() == english.strip().lower()
                        or sorted(PLACEHOLDER.findall(tr)) != sorted(PLACEHOLDER.findall(english))):
                    state["rejected"] += 1
                    continue
                state["done"][sid] = tr
            progress.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            print(f"  round {round_ + 1}: {len(state['done'])}/{len(order)} translated | tokens in {state['tokens_in']} / out {state['tokens_out']}, "
                  f"cost {format_cost(model, spent())}", flush=True)

    rows, seen = [], set()
    for sid in order:
        if sid not in state["done"]:
            continue
        s = src[sid]
        text = state["done"][sid].translate(FOLD) if sid in fold_ids else state["done"][sid]
        rid = stable_id(text)
        if rid in seen:
            continue
        seen.add(rid)
        rows.append({"id": rid, "text": text, "category": s.category, "urgency": s.urgency, "sentiment": s.sentiment, "language": "tr",
                     "source": "translated", "source_id": sid, "ascii_folded": sid in fold_ids, "model": model,
                     "translate_version": TRANSLATE_VERSION, "silver_prompt_version": s.prompt_version if hasattr(s, "prompt_version") else ""})
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}: {len(rows)} rows ({len(order) - len(rows)} missing); {state['calls']} calls, {state['repaired']} repaired, "
          f"{state['rejected']} translations rejected and retried; tokens in {state['tokens_in']} / out {state['tokens_out']}, cost {format_cost(model, spent())}")
    print("per category:", dict(sorted(collections.Counter(r["category"] for r in rows).items())))
    print("urgency:", dict(collections.Counter(r["urgency"] for r in rows)), "| sentiment:", dict(collections.Counter(r["sentiment"] for r in rows)),
          "| ascii-folded:", sum(r["ascii_folded"] for r in rows))
    return 0


ARGS: argparse.Namespace


if __name__ == "__main__":
    raise SystemExit(main())
