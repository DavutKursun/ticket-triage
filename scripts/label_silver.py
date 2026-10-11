"""Silver labels (urgency + sentiment) with the LLM. Resumable, quota-aware, incremental.

Usage:
  python scripts/label_silver.py --split dev  --n 1000 --delay 1
  python scripts/label_silver.py --split test --n 1000 --delay 1
  python scripts/label_silver.py --split train --n 24 --delay 1      # few-shot pool (2 per category)

Writes one line per message to data/silver/<split>.jsonl as it goes; rerunning skips ids already
done. Stops with exit code 3 on a daily-quota error (Groq 200k tokens/day) — rerun tomorrow.
Selection is stratified by category and deterministic (SEED), so partial runs stay comparable.
Exit codes: 0 ok, 2 not configured, 3 daily quota hit, 4 other API error.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.config import SEED, SILVER, settings  # noqa: E402
from ticket_triage.data import append_jsonl, load_split, read_jsonl  # noqa: E402
from ticket_triage.providers import ProviderError, QuotaError, complete_validated, get_provider  # noqa: E402
from ticket_triage.rubric import PROMPT_VERSION, SILVER_INSTRUCTION, SILVER_SYSTEM  # noqa: E402
from ticket_triage.schema import CATEGORIES, SilverLabel  # noqa: E402


def select(df, n: int, seed: int = SEED):
    rng = random.Random(seed)
    per = max(1, n // len(CATEGORIES))
    parts = []
    for c in CATEGORIES:
        g = df[df["category"] == c]
        parts.append(g.sample(min(per, len(g)), random_state=rng.randint(0, 10**6)))
    import pandas as pd

    return pd.concat(parts).sample(frac=1, random_state=seed).head(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between calls (rate limits)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many NEW labels this run (0 = no limit)")
    args = ap.parse_args()

    try:
        provider, model = get_provider(settings)
    except ProviderError as e:
        print(f"not configured: {e}")
        return 2

    df = select(load_split(args.split), args.n)
    out = SILVER / f"{args.split}.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    todo = df[~df["id"].isin(done)]
    print(f"{args.split}: {len(df)} selected, {len(done)} already labelled, {len(todo)} to do, model {model}, prompt {PROMPT_VERSION}")

    new = repaired = 0
    tokens_in = tokens_out = 0
    t_start = time.time()
    for r in todo.itertuples():
        user = SILVER_INSTRUCTION.format(category=r.category, text=r.text)
        try:
            obj, results, rep = complete_validated(provider, SILVER_SYSTEM, user, model, SilverLabel)
        except QuotaError as e:
            print(f"\nDaily quota reached after {new} new labels: {e}\nRerun tomorrow; progress is saved.")
            return 3
        except ProviderError as e:
            print(f"\nAPI error after {new} new labels: {e}")
            return 4
        except Exception as e:  # validation failed twice → record as unlabelled, continue
            print(f"  skip {r.id}: {type(e).__name__}: {str(e)[:120]}")
            continue
        append_jsonl(out, {"id": r.id, "urgency": obj.urgency, "sentiment": obj.sentiment, "reason": obj.reason,
                           "model": model, "prompt_version": PROMPT_VERSION})
        new += 1
        repaired += int(rep)
        tokens_in += sum(x.input_tokens for x in results)
        tokens_out += sum(x.output_tokens for x in results)
        if new % 25 == 0:
            rate = new / (time.time() - t_start)
            print(f"  {new}/{len(todo)}  {rate*60:.0f}/min  tokens in {tokens_in} out {tokens_out}  repaired {repaired}", flush=True)
        if args.limit and new >= args.limit:
            break
        time.sleep(args.delay)
    print(f"done: {new} new labels, {repaired} repaired, tokens in {tokens_in} / out {tokens_out}, total labelled {len(done) + new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
