"""Draft the hard set: realistic multi-issue / typo / Turkish tickets for Davut to edit and label.

Usage: python scripts/draft_hard.py --n 300 --out data/hard/hard_draft.csv
Generates in batches of 10 per LLM call from a written brief (so ~30 calls), stratified across
categories and with 20% Turkish. Davut then edits every row and fills labels → hard.csv.
Resumable: appends to the output file and skips when it already has --n rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from ticket_triage.config import SEED, settings  # noqa: E402
from ticket_triage.data import stable_id  # noqa: E402
from ticket_triage.providers import ProviderError, complete_validated, get_provider  # noqa: E402
from ticket_triage.rubric import CATEGORY_DEFINITIONS, SENTIMENT_RUBRIC, URGENCY_RUBRIC  # noqa: E402
from ticket_triage.schema import CATEGORIES, Sentiment, Urgency  # noqa: E402


class Draft(BaseModel):
    text: str = Field(min_length=40)
    category: str
    urgency: Urgency
    sentiment: Sentiment


class DraftBatch(BaseModel):
    tickets: list[Draft] = Field(min_length=1, max_length=12)


SYSTEM = f"""You write realistic customer-support tickets for a dataset. Each ticket: 2–6 sentences, written like
a real person (some typos, inconsistent capitalisation, order numbers, dates, product names), sometimes
mentioning TWO issues where only one is the main one. Never templated, never polite boilerplate only.
Categories:\n""" + "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items()) + f"""

Urgency rubric:\n{URGENCY_RUBRIC}\n\nSentiment rubric:\n{SENTIMENT_RUBRIC}
Return JSON: {{"tickets": [{{"text", "category", "urgency", "sentiment"}}, ...]}}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/hard/hard_draft.csv")
    args = ap.parse_args()
    try:
        provider, model = get_provider(settings)
    except ProviderError as e:
        print("not configured:", e)
        return 2
    out = Path(args.out)
    rows = []
    if out.exists():
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
    rng = random.Random(SEED + len(rows))
    while len(rows) < args.n:
        cats = rng.sample(CATEGORIES, 5) * 2
        lang = "tr" if rng.random() < 0.2 else "en"
        mix = rng.choice(["mostly angry", "mostly neutral", "mixed, include one positive"])
        user = (f"Write 10 tickets in {'Turkish' if lang == 'tr' else 'English'}, main categories in this order: {cats}. "
                f"Spread urgency across all four levels; sentiment {mix}; at least 3 tickets mention a second, minor issue.")
        try:
            batch, _, _ = complete_validated(provider, SYSTEM, user, model, DraftBatch)
        except ProviderError as e:
            print("API error:", e)
            break
        for d in batch.tickets:
            if d.category not in CATEGORIES:
                continue
            rows.append({"id": stable_id(d.text), "text": d.text, "category": d.category, "urgency": d.urgency,
                         "sentiment": d.sentiment, "language": lang, "split": ""})
        print(f"{len(rows)}/{args.n}", flush=True)
    # assign split alternating per category so dev/test stay balanced
    for i, r in enumerate(sorted(rows, key=lambda r: (r["category"], r["id"]))):
        r["split"] = "dev" if i % 2 == 0 else "test"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text", "category", "urgency", "sentiment", "language", "split"])
        w.writeheader()
        w.writerows(rows[: args.n])
    print("wrote", out, "— edit every row by hand, then save as data/hard/hard.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
